# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer-mapped buffers and a cross-rank grid barrier for Triton kernels.

Every rank of the TP group allocates one uncached buffer and maps every peer's.
A kernel addresses rank ``r``'s buffer through :func:`peer` and separates its
phases with :func:`grid_barrier`: every block of every rank arrives on every
rank's counter, then waits until its own counter reaches a cumulative target.
The barrier needs the whole grid resident, so kernels launch a fixed, small
grid of ``NUM_BLOCKS``.
"""

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_WORLD_SIZE = 8
NUM_BLOCKS = 64


@triton.jit
def peer(P, r, DTYPE: tl.constexpr):
    """Base of rank ``r``'s buffer from the tuple of 8 scalar peer addresses.

    Scalar kernel arguments stay uniform and carry their 16-byte divisibility,
    so the peer accesses vectorize; a base loaded from memory does not.
    """
    base = P[0]
    for i in tl.static_range(1, 8):  # MAX_WORLD_SIZE
        base = tl.where(r == i, P[i], base)
    return tl.multiple_of(base.to(tl.pointer_type(DTYPE)), 16)


@triton.jit
def grid_barrier(P, rank, count_off, target, W: tl.constexpr, NUM_WAVES: tl.constexpr):
    """Arrive on every rank's counter, then wait for ``target`` arrivals."""
    mine = peer(P, rank, tl.int32) + count_off
    # A release by one lane of every wave: ``debug_barrier`` does not wait for
    # the other waves' outstanding stores (no ``s_waitcnt vmcnt(0)``).
    lanes = tl.arange(0, NUM_WAVES * 64)
    tl.atomic_add(mine + lanes * 0, 0, mask=lanes % 64 == 0, sem="release", scope="sys")
    tl.debug_barrier()
    for r in tl.static_range(W):
        tl.atomic_add(peer(P, r, tl.int32) + count_off, 1, sem="release", scope="sys")
    seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    # Wrapping difference: the counters overflow after ~2^31 arrivals.
    while seen - target < 0:
        seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    tl.debug_barrier()


class PeerBuffer:
    """``nbytes`` of zeroed uncached memory per rank, mapped by every rank.

    ``peers`` is the 8-tuple of buffer addresses a kernel takes as ``P``.
    """

    def __init__(self, nbytes: int, group: dist.ProcessGroup) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        assert self.world_size <= MAX_WORLD_SIZE
        # hipExtMallocWithFlags(hipDeviceMallocUncached), zeroed.
        self._pointers = CustomAllreduce.create_shared_buffer(nbytes, group=group)
        self.peers = tuple(self._pointers) + (0,) * (MAX_WORLD_SIZE - self.world_size)

    def close(self) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        if self._pointers:
            CustomAllreduce.free_shared_buffer(self._pointers, rank=self.rank)
            self._pointers = []


def all_tp_ranks(flag: bool) -> bool:
    """Whether ``flag`` holds on every TP rank; collective."""
    value = torch.tensor(int(flag))
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(value.item())


def tp_group_supports_peer_kernels() -> bool:
    """Whether every TP rank is a CDNA3/4 GPU in one fully connected node, with
    a TP size the kernels are built for; collective over the TP group."""
    tp_group = get_tp_group()
    if not current_platform.is_rocm() or tp_group.world_size not in (2, 4, 8):
        return False
    from vllm.platforms.rocm import get_cdna_version

    if not all_tp_ranks(get_cdna_version() in (3, 4)):
        return False
    if not all(in_the_same_node_as(tp_group.cpu_group, source_rank=0)):
        return False
    physical_id = current_platform.visible_device_id_to_physical_device_id(
        torch.accelerator.current_device_index()
    )
    physical_ids: list[int | None] = [None] * tp_group.world_size
    dist.all_gather_object(physical_ids, physical_id, group=tp_group.cpu_group)
    return current_platform.is_fully_connected(physical_ids)
