# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark MoE post-expert finalize fused with its TP all-reduce.

Two deployments reduce the routed experts' output over the TP group:

* Pure TP (``PRE_NORM``): every rank holds a partial sum of each
  ``[T, top_k, H]`` route row, and the post-expert RMSNorm needs the full sum.
  The unfused path all-reduces all ``T * top_k`` rows, then finalizes.
* TP + EP over the same group: every route is complete on the rank that owns
  its expert, so each rank finalizes its local routes and the MoE runner then
  all-reduces the ``[T, H]`` partial outputs.

This kernel does either in one launch over peer-mapped uncached buffers:

A. every rank pushes its contribution for token ``t`` to the owner rank
   ``t % W``: the ``top_k`` route rows (TP), or its fp32 finalized partial row
   (EP);
B. the owner sums the ``W`` contributions in fp32 in rank order (TP: then
   applies the norm, router weights and top-k sum) and pushes the ``[H]``
   output row to every rank;
C. every rank copies the gathered ``[T, H]`` output out.

Contributions are summed in fp32 whatever ``VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE``
is, and every rank gets the same output bits. Phases are separated by grid
barriers across all blocks of all ranks (cumulative arrival counters), so the
launch needs every block resident.
"""

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from .peer_sync import (
    NUM_BLOCKS,
    PeerBuffer,
    all_tp_ranks,
    barrier_words,
    grid_barrier,
    peer,
    tp_group_supports_peer_kernels,
    wave_words,
)

logger = init_logger(__name__)

# Replaces the all-reduce branch of the finalize, which runs below the
# token-sharded reduce-scatter threshold. Keeps every offset in int32.
_MAX_TOKENS = 255
_NUM_WARPS = 8


@triton.jit
def _norm_route(x, w, eps, H: tl.constexpr, HAS_WEIGHT: tl.constexpr, WF: tl.constexpr):
    """MuseSpark's post-expert RMSNorm of one fp32 route row."""
    if HAS_WEIGHT and WF:
        x = x * w
    y = x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / H + eps))
    if HAS_WEIGHT and not WF:
        y = y * w
    return y


@triton.jit
def _finalize_all_reduce_kernel(
    rows_ptr,
    stride_rt,
    stride_rk,
    tw_ptr,
    stride_twt,
    ids_ptr,
    stride_ids,
    expert_map_ptr,
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
    flags_a_off,
    flags_b_off,
    wave_off,
    target_off,
    W: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PRE_NORM: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    NUM_WAVES: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    my_bf16 = peer(P, rank, tl.bfloat16)
    my_f32 = peer(P, rank, tl.float32)
    my_i32 = peer(P, rank, tl.int32)
    target = tl.load(my_i32 + target_off + b) + W * NB
    groups = tl.cdiv(T, W)
    w = tl.zeros([BLOCK_H], dtype=tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + gain_center

    # A: owner r's token t = g * W + r goes to r's slot [rank][g]: its
    # [top_k][H] bf16 route rows (TP), or its [H] fp32 local finalize (EP).
    for r in tl.static_range(W):
        if PRE_NORM:
            rows_dst = peer(P, r, tl.bfloat16) + slots_off + rank * G * K * H
            for i in range(b, groups * K, NB):
                g = i // K
                k = i % K
                t = g * W + r
                if t < T:
                    x = tl.load(rows_ptr + t * stride_rt + k * stride_rk + offs, mask)
                    tl.store(rows_dst + i * H + offs, x, mask=mask)
        else:
            partial_dst = peer(P, r, tl.float32) + slots_off + rank * G * H
            for g in range(b, groups, NB):
                t = g * W + r
                if t < T:
                    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
                    for k in range(K):
                        local = tl.full((), True, tl.int1)
                        if HAS_EXPERT_MAP:
                            e = tl.load(ids_ptr + t * stride_ids + k)
                            local = (
                                tl.load(expert_map_ptr + e, mask=e >= 0, other=-1) >= 0
                            )
                        # Skip non-local routes: never written, nothing to add.
                        if local:
                            x = tl.load(
                                rows_ptr + t * stride_rt + k * stride_rk + offs, mask
                            ).to(tl.float32)
                            tw = tl.load(tw_ptr + t * stride_twt + k).to(tl.float32)
                            y = _norm_route(x, w, eps, H, HAS_WEIGHT, WEIGHT_FIRST)
                            acc += y * tw
                    tl.store(partial_dst + g * H + offs, acc, mask=mask)
    grid_barrier(P, rank, flags_a_off, wave_off, target, W, NB, NUM_WAVES)

    # B: reduce this rank's tokens and push each output row to every rank.
    for g in range(b, groups, NB):
        t = g * W + rank
        if t < T:
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            if PRE_NORM:
                for k in range(K):
                    x = tl.zeros([BLOCK_H], dtype=tl.float32)
                    for s in tl.static_range(W):
                        slot = ((s * G + g) * K + k) * H
                        x += tl.load(my_bf16 + slots_off + slot + offs, mask, 0.0).to(
                            tl.float32
                        )
                    tw = tl.load(tw_ptr + t * stride_twt + k).to(tl.float32)
                    y = _norm_route(x, w, eps, H, HAS_WEIGHT, WEIGHT_FIRST)
                    acc += y * tw
            else:
                for s in tl.static_range(W):
                    slot = (s * G + g) * H
                    acc += tl.load(my_f32 + slots_off + slot + offs, mask, 0.0)
            res = acc.to(tl.bfloat16)
            for r in tl.static_range(W):
                gathered = peer(P, r, tl.bfloat16) + gathered_off
                tl.store(gathered + t * H + offs, res, mask=mask)
    grid_barrier(P, rank, flags_b_off, wave_off, target, W, NB, NUM_WAVES)

    # C: copy the gathered rows out.
    for t in range(b, T, NB):
        x = tl.load(my_bf16 + gathered_off + t * H + offs, mask)
        tl.store(out_ptr + t * stride_out + offs, x, mask=mask)
    tl.store(my_i32 + target_off + b, target)


class MuseSparkFinalizeAllReduce:
    """Peer-mapped workspace for the fused finalize, shared by every MoE layer.

    Layout of each rank's buffer, in bytes: the slots, ``[W][G][top_k][H]``
    bf16 route rows (``pre_norm``) or ``[W][G][H]`` fp32 partial outputs, with
    ``G = ceil(max_tokens / W)``; the gathered output ``[max_tokens][H]`` bf16;
    then two arrival counters and one barrier target per block (int32).
    """

    def __init__(
        self,
        max_tokens: int,
        top_k: int,
        hidden_size: int,
        group: dist.ProcessGroup,
        *,
        pre_norm: bool,
    ) -> None:
        self.group = group
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.pre_norm = pre_norm
        self.world_size = dist.get_world_size(group=group)
        self.slots_per_rank = triton.cdiv(max_tokens, self.world_size)
        # Whether the last finalize returned the reduced output (EP mode); read
        # by the MoE runner to skip its own all-reduce.
        self.output_reduced = False

        slot_bytes = top_k * hidden_size * 2 if pre_norm else hidden_size * 4
        self.slots_off = 0
        gathered_off = self.world_size * self.slots_per_rank * slot_bytes
        sync_off = gathered_off + max_tokens * hidden_size * 2
        # Element offsets for the kernel's typed base pointers.
        self.gathered_off = gathered_off // 2
        self.flags_a_off = sync_off // 4
        self.flags_b_off = self.flags_a_off + barrier_words(self.world_size)
        self.wave_off = self.flags_b_off + barrier_words(self.world_size)
        self.target_off = self.wave_off + wave_words()
        nbytes = sync_off + (self.target_off - self.flags_a_off + NUM_BLOCKS) * 4
        self.buffer = PeerBuffer(nbytes, group)

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
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out = all_reduce(finalize(routed))`` (EP) or
        ``finalize(all_reduce(routed))`` (TP); collective over the TP group.

        In EP mode, routes whose ``expert_map`` entry is negative were never
        written and are masked before the read.
        """
        num_tokens, top_k, hidden = routed.shape
        assert topk_weights.shape == (num_tokens, top_k)
        assert topk_weights.stride(1) == 1
        assert out.shape == (num_tokens, hidden)
        assert expert_map is None or not self.pre_norm
        ids, stride_ids = topk_weights, 0
        if expert_map is not None:
            assert topk_ids is not None and topk_ids.stride(1) == 1
            ids, stride_ids = topk_ids, topk_ids.stride(0)
        _finalize_all_reduce_kernel[(NUM_BLOCKS,)](
            routed,
            routed.stride(0),
            routed.stride(1),
            topk_weights,
            topk_weights.stride(0),
            ids,
            stride_ids,
            expert_map if expert_map is not None else topk_weights,
            weight if weight is not None else routed,
            out,
            out.stride(0),
            self.buffer.peers,
            self.buffer.rank,
            num_tokens,
            eps,
            gain_center,
            self.slots_off,
            self.gathered_off,
            self.flags_a_off,
            self.flags_b_off,
            self.wave_off,
            self.target_off,
            W=self.world_size,
            K=top_k,
            H=hidden,
            G=self.slots_per_rank,
            NB=NUM_BLOCKS,
            BLOCK_H=triton.next_power_of_2(hidden),
            PRE_NORM=self.pre_norm,
            HAS_EXPERT_MAP=expert_map is not None,
            HAS_WEIGHT=weight is not None,
            WEIGHT_FIRST=weight_first,
            NUM_WAVES=_NUM_WARPS,
            num_warps=_NUM_WARPS,
        )
        return out

    def close(self) -> None:
        self.buffer.close()


_WORKSPACE: MuseSparkFinalizeAllReduce | None = None


def get_finalize_all_reduce() -> MuseSparkFinalizeAllReduce | None:
    return _WORKSPACE


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
        or hidden_size * 2 % 16
        # EP mode reduces over the TP group, which must be the EP group.
        or (pc.enable_expert_parallel and pc.data_parallel_size != 1)
        or pc.enable_eplb
        or pc.use_sequence_parallel_moe
        or pc.use_ubatching
        or envs.VLLM_BATCH_INVARIANT
        or not tp_group_supports_peer_kernels()
    ):
        return None
    try:
        workspace = MuseSparkFinalizeAllReduce(
            min(_MAX_TOKENS, vllm_config.scheduler_config.max_num_batched_tokens),
            top_k,
            hidden_size,
            tp_group.cpu_group,
            pre_norm=not pc.enable_expert_parallel,
        )
    except RuntimeError as exc:
        logger.warning("Muse Spark fused MoE finalize + all-reduce failed: %s", exc)
        workspace = None
    if not all_tp_ranks(workspace is not None):
        if workspace is not None:
            workspace.close()
        return None
    assert workspace is not None
    _WORKSPACE = workspace
    logger.info_once(
        "Muse Spark fused MoE finalize + all-reduce enabled (TP%d%s, up to %d tokens).",
        tp_group.world_size,
        "" if workspace.pre_norm else " + EP",
        workspace.max_tokens,
    )
    return workspace
