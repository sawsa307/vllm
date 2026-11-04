# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer-mapped buffers and a cross-rank grid barrier for Triton kernels.

Every rank of the TP group allocates one uncached buffer and maps every peer's.
A kernel addresses rank ``r``'s buffer through :func:`peer` and separates its
phases with :func:`grid_barrier`: block ``b`` of rank ``s`` writes the call's
target into flag ``[s][b]`` of every rank, then waits until all ``W * NB``
flags of its own rank carry it. One writer per flag and vector polls keep the
barrier free of contended atomics (on MI355X the earlier shared counter with a
system-scope release per wave cost 28 us, against 4 us for a custom
all-reduce). The barrier needs the whole grid resident, so kernels launch a
fixed, small grid of ``NUM_BLOCKS``; a workspace reserves
:func:`barrier_words` int32 words per barrier and :func:`wave_words` once.
"""

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_WORLD_SIZE = 8
NUM_BLOCKS = 64
MAX_WAVES = 16


def barrier_words(world_size: int) -> int:
    """int32 words one barrier's flags take: ``[world_size][NUM_BLOCKS]``."""
    return world_size * NUM_BLOCKS


def wave_words() -> int:
    """int32 words of the per-(block, wave) release scratch a workspace holds."""
    return NUM_BLOCKS * MAX_WAVES


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
def _wait_stores(x):
    """``s_waitcnt vmcnt(0)``: this wave's outstanding stores have completed
    (GFX9 counts stores in vmcnt). Neither ``debug_barrier`` nor a
    workgroup-scope release waits for them on CDNA (no tgsplit), so the
    cross-rank flag could otherwise be published before the data lands. Side
    effects plus a memory clobber keep the stores on the near side of it."""
    return tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\nv_mov_b32 $0, $1",
        "=v,v,~{memory}",
        [x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grid_barrier(
    P,
    rank,
    flags_off,
    wave_off,
    target,
    W: tl.constexpr,
    NB: tl.constexpr,
    NUM_WAVES: tl.constexpr,
):
    """Publish ``target`` on flag ``[rank][block]`` of every rank, then wait
    until every flag of this rank has reached it. ``wave_off`` is reserved."""
    b = tl.program_id(0)
    mine = peer(P, rank, tl.int32)
    # Every wave waits for its own stores, then the block synchronizes, then
    # one lane publishes the flag with a system-scope release.
    _wait_stores(tl.arange(0, NUM_WAVES * 64))
    tl.debug_barrier()
    for r in tl.static_range(W):
        tl.atomic_xchg(
            peer(P, r, tl.int32) + flags_off + rank * NB + b,
            target,
            sem="release",
            scope="sys",
        )
    flags = mine + flags_off + tl.arange(0, W * NB)
    # Wrapping difference: the targets overflow after ~2^31 arrivals. The loop
    # carries the scalar, not the flag vector (a vector carry fails to compile
    # at one wave).
    behind = tl.min(tl.load(flags, volatile=True) - target, axis=0)
    while behind < 0:
        behind = tl.min(tl.load(flags, volatile=True) - target, axis=0)
    # Acquire after the successful poll, so the data reads that follow are not
    # reordered before it.
    tl.atomic_add(mine + flags_off + rank * NB + b, 0, sem="acquire", scope="sys")
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
