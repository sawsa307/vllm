# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention output projection's TP all-reduce fused with the residual
boundary that follows it.

At decode sizes ``o_proj`` runs without its reduction, and one launch reduces
the partial ``[T, D]`` rows and applies the whole boundary (post-attention norm,
gated FP32 residual, pre-FFN norm) through ``norm_residual_norm_tile``:

* one-shot (small batches): every rank pushes all its partial rows to every
  rank; after one grid barrier each rank sums the ``W`` partials of every row;
* two-shot (larger batches): every rank pushes each partial row to the row's
  owner (``t % W``); the owner sums them and pushes the reduced row to every
  rank; after a second barrier each rank reads the reduced rows. Half the link
  traffic of one-shot, one more barrier.

Both sum the partials in fp32 in rank order and round the sum to the activation
dtype once (the unfused all-reduce's output), so they give the same bits, on
every rank, and the replicated residual stream stays identical across ranks.

The workspace is this fusion's own, separate from the MoE gathers and the fused
finalize. The two variants use disjoint buffer regions and their own counters
and barrier targets. One-shot calls alternate between two buffers by the parity
of the one-shot call count, as in :mod:`.column_gather`; two-shot calls need no
parity, since a rank only reuses a region after a barrier every peer reaches
after reading it.
"""

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import norm_residual_norm_tile
from vllm.triton_utils import tl, triton

from .column_gather import parity_buffer
from .peer_sync import (
    NUM_BLOCKS,
    PeerBuffer,
    all_tp_ranks,
    grid_barrier,
    peer,
    tp_group_supports_peer_kernels,
)

logger = init_logger(__name__)

MAX_TOKENS = 256
# One-shot pushes (W - 1) full partials per rank; above this two-shot moves less.
ONE_SHOT_MAX_TOKENS = 32


@triton.jit
def _boundary_row(
    x,
    t,
    d,
    dm,
    stream_ptr,
    new_stream_ptr,
    out_ptr,
    post_w_ptr,
    pre_w_ptr,
    alpha_ptr,
    beta_ptr,
    router_ptr,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    D: tl.constexpr,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    ROUTER_OUT: tl.constexpr,
):
    """The boundary of reduced row ``t`` (fp32 values of the bf16 sum)."""
    norm_residual_norm_tile(
        x[None, :],
        stream_ptr,
        new_stream_ptr,
        out_ptr,
        post_w_ptr,
        pre_w_ptr,
        alpha_ptr,
        beta_ptr,
        t * D + d[None, :],
        d,
        dm,
        dm[None, :],
        D,
        post_eps,
        pre_eps,
        post_gain_center,
        pre_gain_center,
        alpha_scalar,
        POST_HAS_WEIGHT,
        POST_INPUT_SCALE,
        PRE_HAS_WEIGHT,
        PRE_INPUT_SCALE,
        GATE_STREAM,
        PER_CHANNEL,
        router_ptr,
        ROUTER_OUT,
    )


@triton.jit
def _all_reduce_boundary_kernel(
    partial_ptr,
    stride_partial,
    stream_ptr,
    new_stream_ptr,
    out_ptr,
    post_w_ptr,
    pre_w_ptr,
    alpha_ptr,
    beta_ptr,
    router_ptr,
    P,
    rank,
    T,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    one_shot_elems,
    slots_off,
    gathered_off,
    count1_off,
    count2a_off,
    count2b_off,
    target1_off,
    target2_off,
    W: tl.constexpr,
    D: tl.constexpr,
    MAX_T1: tl.constexpr,
    G: tl.constexpr,
    NB: tl.constexpr,
    LOG2_ARRIVALS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TWO_SHOT: tl.constexpr,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    ROUTER_OUT: tl.constexpr,
    NUM_WAVES: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dm = d < D
    mine = peer(P, rank, tl.bfloat16)
    dtype = partial_ptr.dtype.element_ty
    if TWO_SHOT:
        target = tl.load(peer(P, rank, tl.int32) + target2_off + b) + W * NB
        groups = tl.cdiv(T, W)
        # A: owner r's partial row t = g * W + r goes to r's slot [rank][g].
        for r in tl.static_range(W):
            dst = peer(P, r, tl.bfloat16) + slots_off + rank * G * D
            for g in range(b, groups, NB):
                t = g * W + r
                if t < T:
                    x = tl.load(partial_ptr + t * stride_partial + d, mask=dm)
                    tl.store(dst + g * D + d, x, mask=dm)
        grid_barrier(P, rank, count2a_off, target, W, NUM_WAVES)
        # B: sum this rank's rows in rank order, round once, push to every rank.
        for g in range(b, groups, NB):
            t = g * W + rank
            if t < T:
                x = tl.zeros([BLOCK_D], dtype=tl.float32)
                for s in tl.static_range(W):
                    row = mine + slots_off + (s * G + g) * D
                    x += tl.load(row + d, mask=dm, other=0.0).to(tl.float32)
                res = x.to(dtype)
                for r in tl.static_range(W):
                    dst = peer(P, r, tl.bfloat16) + gathered_off + t * D
                    tl.store(dst + d, res, mask=dm)
        grid_barrier(P, rank, count2b_off, target, W, NUM_WAVES)
        # C: the boundary of every reduced row.
        for t in range(b, T, NB):
            row = mine + gathered_off + t * D
            x = tl.load(row + d, mask=dm, other=0.0).to(tl.float32)
            _boundary_row(
                x, t, d, dm, stream_ptr, new_stream_ptr, out_ptr, post_w_ptr,
                pre_w_ptr, alpha_ptr, beta_ptr, router_ptr, post_eps, pre_eps,
                post_gain_center, pre_gain_center, alpha_scalar, D,
                POST_HAS_WEIGHT, POST_INPUT_SCALE, PRE_HAS_WEIGHT,
                PRE_INPUT_SCALE, GATE_STREAM, PER_CHANNEL, ROUTER_OUT,
            )  # fmt: skip
        tl.store(peer(P, rank, tl.int32) + target2_off + b, target)
    else:
        target, buf_off = parity_buffer(
            P, rank, target1_off, one_shot_elems, LOG2_ARRIVALS
        )
        # A: this rank's partial row t goes to slot [rank][t] of every rank.
        for t in range(b, T, NB):
            x = tl.load(partial_ptr + t * stride_partial + d, mask=dm)
            for r in tl.static_range(W):
                dst = peer(P, r, tl.bfloat16) + buf_off + (rank * MAX_T1 + t) * D
                tl.store(dst + d, x, mask=dm)
        grid_barrier(P, rank, count1_off, target, W, NUM_WAVES)
        # B: sum the partials in rank order, round once, run the boundary.
        for t in range(b, T, NB):
            x = tl.zeros([BLOCK_D], dtype=tl.float32)
            for s in tl.static_range(W):
                row = mine + buf_off + (s * MAX_T1 + t) * D
                x += tl.load(row + d, mask=dm, other=0.0).to(tl.float32)
            x = x.to(dtype).to(tl.float32)
            _boundary_row(
                x, t, d, dm, stream_ptr, new_stream_ptr, out_ptr, post_w_ptr,
                pre_w_ptr, alpha_ptr, beta_ptr, router_ptr, post_eps, pre_eps,
                post_gain_center, pre_gain_center, alpha_scalar, D,
                POST_HAS_WEIGHT, POST_INPUT_SCALE, PRE_HAS_WEIGHT,
                PRE_INPUT_SCALE, GATE_STREAM, PER_CHANNEL, ROUTER_OUT,
            )  # fmt: skip
        tl.store(peer(P, rank, tl.int32) + target1_off + b, target)


class MuseSparkAllReduceBoundary:
    """Peer-mapped workspace of the fused attention all-reduce + boundary.

    Each rank's buffer holds, in bf16: two one-shot ``[W][one_shot_max][D]``
    buffers, the two-shot slots ``[W][ceil(max_tokens / W)][D]`` and the
    two-shot reduced rows ``[max_tokens][D]``; then, in int32, one arrival
    counter for one-shot and two for two-shot, and one barrier target per block
    for each variant.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden: int,
        group: dist.ProcessGroup,
        one_shot_max_tokens: int = ONE_SHOT_MAX_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.one_shot_max_tokens = min(one_shot_max_tokens, max_tokens)
        self.hidden = hidden
        self.world_size = world = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)
        self.slots_per_rank = triton.cdiv(max_tokens, world)
        self.one_shot_elems = world * max(self.one_shot_max_tokens, 1) * hidden
        self.slots_off = 2 * self.one_shot_elems
        self.gathered_off = self.slots_off + world * self.slots_per_rank * hidden
        sync_off = (self.gathered_off + max_tokens * hidden) * 2
        self.count1_off = sync_off // 4
        self.count2a_off = self.count1_off + 1
        self.count2b_off = self.count1_off + 2
        self.target1_off = self.count1_off + 3
        self.target2_off = self.target1_off + NUM_BLOCKS
        self.log2_arrivals = (world * NUM_BLOCKS).bit_length() - 1
        assert 1 << self.log2_arrivals == world * NUM_BLOCKS
        self.buffer = PeerBuffer(sync_off + (3 + 2 * NUM_BLOCKS) * 4, group)

    def supports(
        self,
        dtype: torch.dtype,
        stream: torch.Tensor,
        post_norm: MuseSpark1xRMSNorm,
        connection: object,
        pre_norm: MuseSpark1xRMSNorm,
    ) -> bool:
        """Whether :meth:`__call__` serves this boundary for ``dtype`` partials;
        decided before the projection runs, so it can skip its reduction."""
        return (
            isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
            and getattr(connection, "_coefficients_ready", False)
            and dtype == torch.bfloat16
            and stream.dim() == 2
            and 0 < stream.shape[0] <= self.max_tokens
            and stream.shape[1] == self.hidden
            and stream.dtype == torch.float32
            and stream.is_contiguous()
            and post_norm.variance_size_override is None
            and pre_norm.variance_size_override is None
            and post_norm.hidden_size == self.hidden == pre_norm.hidden_size
        )

    def __call__(
        self,
        partial: torch.Tensor,
        stream: torch.Tensor,
        post_norm: MuseSpark1xRMSNorm,
        connection: MuseSpark1xNormPreservingResidualConnection,
        pre_norm: MuseSpark1xRMSNorm,
        out_dtype: torch.dtype,
        router_out: torch.Tensor | None = None,
        two_shot: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fused_norm_residual_norm_fwd`` of ``all_reduce(partial)``; returns
        ``(out, new_stream)``. Collective over TP. ``router_out`` (fp32, if
        given) receives the unrounded pre-norm result. ``two_shot`` defaults to
        the batch size's variant; both give the same bits."""
        assert self.supports(partial.dtype, stream, post_norm, connection, pre_norm)
        assert partial.shape == stream.shape and partial.stride(-1) == 1
        if router_out is not None:
            assert router_out.shape == stream.shape
            assert router_out.dtype == torch.float32 and router_out.is_contiguous()
        num_tokens, hidden = stream.shape
        if two_shot is None:
            two_shot = num_tokens > self.one_shot_max_tokens
        assert two_shot or num_tokens <= self.one_shot_max_tokens
        alpha, beta = connection.alpha, connection.beta
        gate_stream = isinstance(alpha, torch.Tensor)
        assert beta.is_contiguous() and beta.numel() in (1, hidden)
        if isinstance(alpha, torch.Tensor):
            assert alpha.shape == beta.shape and alpha.is_contiguous()
        post_w = post_norm.weight if post_norm.has_weight else None
        pre_w = pre_norm.weight if pre_norm.has_weight else None
        post_scale = post_w is not None and post_norm.applies_weight_before_norm
        pre_scale = pre_w is not None and pre_norm.applies_weight_before_norm
        out = torch.empty_like(stream, dtype=out_dtype)
        new_stream = torch.empty_like(stream)
        num_waves = 16 if hidden > 4096 else 8
        _all_reduce_boundary_kernel[(NUM_BLOCKS,)](
            partial,
            partial.stride(0),
            stream,
            new_stream,
            out,
            post_w if post_w is not None else stream,
            pre_w if pre_w is not None else stream,
            alpha if gate_stream else beta,
            beta,
            router_out if router_out is not None else stream,
            self.buffer.peers,
            self.rank,
            num_tokens,
            post_norm.variance_epsilon,
            pre_norm.variance_epsilon,
            post_norm.fused_gain_center,
            pre_norm.fused_gain_center,
            1.0 if gate_stream else float(alpha),
            self.one_shot_elems,
            self.slots_off,
            self.gathered_off,
            self.count1_off,
            self.count2a_off,
            self.count2b_off,
            self.target1_off,
            self.target2_off,
            W=self.world_size,
            D=hidden,
            MAX_T1=max(self.one_shot_max_tokens, 1),
            G=self.slots_per_rank,
            NB=NUM_BLOCKS,
            LOG2_ARRIVALS=self.log2_arrivals,
            BLOCK_D=triton.next_power_of_2(hidden),
            TWO_SHOT=two_shot,
            POST_HAS_WEIGHT=post_w is not None,
            POST_INPUT_SCALE=post_scale,
            PRE_HAS_WEIGHT=pre_w is not None,
            PRE_INPUT_SCALE=pre_scale,
            GATE_STREAM=gate_stream,
            PER_CHANNEL=beta.numel() > 1,
            ROUTER_OUT=router_out is not None,
            NUM_WAVES=num_waves,
            num_warps=num_waves,
        )
        return out, new_stream

    def close(self) -> None:
        self.buffer.close()


_WORKSPACE: MuseSparkAllReduceBoundary | None = None


def maybe_create_all_reduce_boundary(
    vllm_config: VllmConfig, hidden: int
) -> MuseSparkAllReduceBoundary | None:
    """Collectively create the workspace once; every TP rank gets one or none.
    Call from every TP rank in the same order (model construction)."""
    global _WORKSPACE
    if _WORKSPACE is not None:
        return _WORKSPACE
    import vllm.envs as envs

    pc = vllm_config.parallel_config
    if (
        not envs.VLLM_ROCM_MUSE_FUSED_ATTN_ALL_REDUCE
        or hidden > 8192
        or hidden % 8
        or pc.use_ubatching
        or vllm_config.lora_config is not None
        or envs.VLLM_BATCH_INVARIANT
        or not tp_group_supports_peer_kernels()
    ):
        return None
    tp_group = get_tp_group()
    try:
        workspace = MuseSparkAllReduceBoundary(
            min(MAX_TOKENS, vllm_config.scheduler_config.max_num_batched_tokens),
            hidden,
            tp_group.cpu_group,
            envs.VLLM_ROCM_MUSE_ATTN_ALL_REDUCE_ONE_SHOT_MAX_TOKENS,
        )
    except RuntimeError as exc:
        logger.warning("Muse Spark fused attention all-reduce failed: %s", exc)
        workspace = None
    if not all_tp_ranks(workspace is not None):
        if workspace is not None:
            workspace.close()
        return None
    assert workspace is not None
    _WORKSPACE = workspace
    logger.info_once(
        "Muse Spark attention all-reduce fused into the residual boundary "
        "(TP%d, up to %d tokens; one-shot up to %d, two-shot above).",
        tp_group.world_size,
        workspace.max_tokens,
        workspace.one_shot_max_tokens,
    )
    return workspace
