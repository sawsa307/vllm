# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP all-reduce of the GEMM2 route rows fused with the post-expert finalize.

Under pure TP every rank holds a partial sum of each ``[T, top_k, H]`` route
row, and the post-expert RMSNorm needs the full sum. The unfused path
all-reduces all ``T * top_k`` rows and then runs ``fused_post_expert_finalize``.
This kernel does both in one launch over peer-mapped uncached buffers:

A. every rank pushes each row to the token's owner rank (``t % W``);
B. the owner sums the ``W`` partials in fp32 in rank order, applies the norm,
   router weights and top-k sum, and pushes the ``[H]`` output row to every
   rank;
C. every rank copies the gathered ``[T, H]`` output out.

Only ``1 / top_k`` of the all-reduce volume is gathered, the partials are summed
in fp32 whatever ``VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE`` is, and every rank gets the
same output bits. Phases are separated by grid barriers across all blocks of all
ranks (cumulative arrival counters), so the launch needs every block resident.
"""

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Replaces the all-reduce branch of the finalize, which runs below the
# token-sharded reduce-scatter threshold. Keeps every offset in int32.
_MAX_TOKENS = 255
_NUM_BLOCKS = 64
_NUM_WARPS = 8


@triton.jit
def _peer(P, r, DTYPE: tl.constexpr):
    """Base of rank ``r``'s buffer from the tuple of 8 scalar peer addresses.

    Scalar kernel arguments stay uniform and carry their 16-byte divisibility,
    so the peer accesses vectorize; a base loaded from memory does not.
    """
    base = P[0]
    for i in tl.static_range(1, 8):
        base = tl.where(r == i, P[i], base)
    return tl.multiple_of(base.to(tl.pointer_type(DTYPE)), 16)


@triton.jit
def _grid_barrier(P, rank, count_off, target, W: tl.constexpr, NUM_WAVES: tl.constexpr):
    """Arrive on every rank's counter, then wait for ``target`` arrivals."""
    mine = _peer(P, rank, tl.int32) + count_off
    # A release by one lane of every wave: ``debug_barrier`` does not wait for
    # the other waves' outstanding stores (no ``s_waitcnt vmcnt(0)``).
    lanes = tl.arange(0, NUM_WAVES * 64)
    tl.atomic_add(mine + lanes * 0, 0, mask=lanes % 64 == 0, sem="release", scope="sys")
    tl.debug_barrier()
    for r in tl.static_range(W):
        tl.atomic_add(_peer(P, r, tl.int32) + count_off, 1, sem="release", scope="sys")
    seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    # Wrapping difference: the counters overflow after ~2^31 arrivals.
    while seen - target < 0:
        seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    tl.debug_barrier()


@triton.jit
def _finalize_all_reduce_kernel(
    rows_ptr,
    stride_rt,
    stride_rk,
    tw_ptr,
    stride_twt,
    w_ptr,
    out_ptr,
    stride_out,
    P,
    rank,
    T,
    eps,
    gain_center,
    slots_off,
    gathered_off,
    count_a_off,
    count_b_off,
    target_off,
    W: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    NUM_WAVES: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    my_bf16 = _peer(P, rank, tl.bfloat16)
    my_i32 = _peer(P, rank, tl.int32)
    target = tl.load(my_i32 + target_off + b) + W * NB
    groups = tl.cdiv(T, W)

    # A: row (t, k) of owner r's token t = g * W + r goes to r's slot
    # [rank][g][k].
    for r in tl.static_range(W):
        dst = _peer(P, r, tl.bfloat16) + slots_off + rank * G * K * H
        for i in range(b, groups * K, NB):
            g = i // K
            k = i % K
            t = g * W + r
            if t < T:
                x = tl.load(rows_ptr + t * stride_rt + k * stride_rk + offs, mask)
                tl.store(dst + i * H + offs, x, mask=mask)
    _grid_barrier(P, rank, count_a_off, target, W, NUM_WAVES)

    # B: finalize this rank's tokens and push each output row to every rank.
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + gain_center
    for g in range(b, groups, NB):
        t = g * W + rank
        if t < T:
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for k in range(K):
                x = tl.zeros([BLOCK_H], dtype=tl.float32)
                for s in tl.static_range(W):
                    slot = ((s * G + g) * K + k) * H
                    x += tl.load(my_bf16 + slots_off + slot + offs, mask, 0.0).to(
                        tl.float32
                    )
                if HAS_WEIGHT and WEIGHT_FIRST:
                    x = x * w
                var = tl.sum(x * x, axis=0) / H
                y = x * (1.0 / tl.sqrt(var + eps))
                if HAS_WEIGHT and not WEIGHT_FIRST:
                    y = y * w
                tw = tl.load(tw_ptr + t * stride_twt + k).to(tl.float32)
                acc += y * tw
            res = acc.to(tl.bfloat16)
            for r in tl.static_range(W):
                dst = _peer(P, r, tl.bfloat16) + gathered_off
                tl.store(dst + t * H + offs, res, mask=mask)
    _grid_barrier(P, rank, count_b_off, target, W, NUM_WAVES)

    # C: copy the gathered rows out.
    for t in range(b, T, NB):
        x = tl.load(my_bf16 + gathered_off + t * H + offs, mask)
        tl.store(out_ptr + t * stride_out + offs, x, mask=mask)
    tl.store(my_i32 + target_off + b, target)


class MuseSparkFinalizeAllReduce:
    """Peer-mapped workspace for the fused finalize, shared by every MoE layer.

    Layout of each rank's buffer, in bytes: the route slots
    ``[W][G][top_k][H]`` bf16 (``G = ceil(max_tokens / W)``), the gathered
    output ``[max_tokens][H]`` bf16, then two arrival counters and one barrier
    target per block (int32).
    """

    def __init__(
        self,
        max_tokens: int,
        top_k: int,
        hidden_size: int,
        group: dist.ProcessGroup,
    ) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.slots_per_rank = triton.cdiv(max_tokens, self.world_size)

        row_bytes = hidden_size * 2
        self.slots_off = 0
        slots_bytes = self.world_size * self.slots_per_rank * top_k * row_bytes
        gathered_off = slots_bytes
        sync_off = gathered_off + max_tokens * row_bytes
        nbytes = sync_off + (2 + _NUM_BLOCKS) * 4
        # Element offsets for the kernel's typed base pointers.
        self.gathered_off = gathered_off // 2
        self.count_a_off = sync_off // 4
        self.count_b_off = self.count_a_off + 1
        self.target_off = self.count_a_off + 2
        # hipExtMallocWithFlags(hipDeviceMallocUncached), zeroed.
        self._pointers = CustomAllreduce.create_shared_buffer(nbytes, group=group)
        self.peers = tuple(self._pointers) + (0,) * (8 - self.world_size)

    def supports(self, routed: torch.Tensor, out: torch.Tensor) -> bool:
        num_tokens, top_k, hidden = routed.shape
        return (
            0 < num_tokens <= self.max_tokens
            and top_k == self.top_k
            and hidden == self.hidden_size
            and routed.dtype == torch.bfloat16
            and routed.stride(-1) == 1
            and out.dtype == torch.bfloat16
            and out.stride(-1) == 1
        )

    def __call__(
        self,
        routed: torch.Tensor,
        weight: torch.Tensor | None,
        topk_weights: torch.Tensor,
        eps: float,
        gain_center: float,
        weight_first: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """``out = finalize(all_reduce(routed))``; collective over the TP group."""
        num_tokens, top_k, hidden = routed.shape
        assert topk_weights.shape == (num_tokens, top_k)
        assert topk_weights.stride(1) == 1
        assert out.shape == (num_tokens, hidden)
        _finalize_all_reduce_kernel[(_NUM_BLOCKS,)](
            routed,
            routed.stride(0),
            routed.stride(1),
            topk_weights,
            topk_weights.stride(0),
            weight if weight is not None else routed,
            out,
            out.stride(0),
            self.peers,
            self.rank,
            num_tokens,
            eps,
            gain_center,
            self.slots_off,
            self.gathered_off,
            self.count_a_off,
            self.count_b_off,
            self.target_off,
            W=self.world_size,
            K=top_k,
            H=hidden,
            G=self.slots_per_rank,
            NB=_NUM_BLOCKS,
            BLOCK_H=triton.next_power_of_2(hidden),
            HAS_WEIGHT=weight is not None,
            WEIGHT_FIRST=weight_first,
            NUM_WAVES=_NUM_WARPS,
            num_warps=_NUM_WARPS,
        )
        return out

    def close(self) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        if self._pointers:
            CustomAllreduce.free_shared_buffer(self._pointers, rank=self.rank)
            self._pointers = []


_WORKSPACE: MuseSparkFinalizeAllReduce | None = None


def get_finalize_all_reduce() -> MuseSparkFinalizeAllReduce | None:
    return _WORKSPACE


def _all_tp_ranks(flag: bool) -> bool:
    value = torch.tensor(int(flag))
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(value.item())


def maybe_create_finalize_all_reduce(
    vllm_config: VllmConfig, top_k: int, hidden_size: int
) -> MuseSparkFinalizeAllReduce | None:
    """Collectively create the shared workspace once; every TP rank gets one or
    none. Call from every TP rank in the same order (model construction)."""
    global _WORKSPACE
    if _WORKSPACE is not None:
        return _WORKSPACE
    import vllm.envs as envs

    pc = vllm_config.parallel_config
    tp_group = get_tp_group()
    if (
        not envs.VLLM_ROCM_MUSE_FUSED_FINALIZE_ALL_REDUCE
        or not current_platform.is_rocm()
        or tp_group.world_size not in (2, 4, 8)
        or pc.enable_expert_parallel
        or pc.use_sequence_parallel_moe
        or pc.use_ubatching
        or envs.VLLM_BATCH_INVARIANT
    ):
        return None
    from vllm.platforms.rocm import get_cdna_version

    eligible = get_cdna_version() in (3, 4) and hidden_size * 2 % 16 == 0
    if not _all_tp_ranks(eligible):
        return None
    if not all(in_the_same_node_as(tp_group.cpu_group, source_rank=0)):
        return None
    physical_id = current_platform.visible_device_id_to_physical_device_id(
        torch.accelerator.current_device_index()
    )
    physical_ids: list[int | None] = [None] * tp_group.world_size
    dist.all_gather_object(physical_ids, physical_id, group=tp_group.cpu_group)
    if not current_platform.is_fully_connected(physical_ids):
        return None
    try:
        workspace = MuseSparkFinalizeAllReduce(
            min(_MAX_TOKENS, vllm_config.scheduler_config.max_num_batched_tokens),
            top_k,
            hidden_size,
            tp_group.cpu_group,
        )
    except RuntimeError as exc:
        logger.warning("Muse Spark fused MoE finalize + all-reduce failed: %s", exc)
        workspace = None
    if not _all_tp_ranks(workspace is not None):
        if workspace is not None:
            workspace.close()
        return None
    assert workspace is not None
    _WORKSPACE = workspace
    logger.info_once(
        "Muse Spark fused MoE finalize + all-reduce enabled (TP%d, up to %d tokens).",
        tp_group.world_size,
        workspace.max_tokens,
    )
    return workspace
