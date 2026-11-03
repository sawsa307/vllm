# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention output projection's TP all-reduce fused with the residual
boundary that follows it.

At decode sizes ``o_proj`` runs without its reduction; one launch then pushes
every rank's partial ``[T, D]`` rows into every rank's buffer and, after one
grid barrier, sums the ``W`` partials in fp32 in rank order, rounds the sum to
the activation dtype (the unfused all-reduce's output), and applies the whole
boundary (post-attention norm, gated FP32 residual, pre-FFN norm) through
``norm_residual_norm_tile``. Every rank computes the same rows from the same
bits, so the replicated residual stream stays identical across ranks.

The workspace is this fusion's own, separate from the MoE gathers and the
fused finalize. Calls alternate between two buffers by call parity, as in
:mod:`.column_gather`.
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
    P,
    rank,
    T,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    buf_elems,
    count_off,
    target_off,
    W: tl.constexpr,
    D: tl.constexpr,
    MAX_T: tl.constexpr,
    NB: tl.constexpr,
    LOG2_ARRIVALS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    NUM_WAVES: tl.constexpr,
):
    target, buf_off = parity_buffer(P, rank, target_off, buf_elems, LOG2_ARRIVALS)
    d = tl.arange(0, BLOCK_D)
    dm = d < D

    # A: this rank's partial row t goes to slot [rank][t] of every rank.
    for t in range(tl.program_id(0), T, NB):
        x = tl.load(partial_ptr + t * stride_partial + d, mask=dm)
        for r in tl.static_range(W):
            dst = peer(P, r, tl.bfloat16) + buf_off + (rank * MAX_T + t) * D
            tl.store(dst + d, x, mask=dm)
    grid_barrier(P, rank, count_off, target, W, NUM_WAVES)

    # B: sum the partials in rank order, round as the all-reduce does, and run
    # the boundary on the reduced row.
    mine = peer(P, rank, tl.bfloat16) + buf_off
    for t in range(tl.program_id(0), T, NB):
        x = tl.zeros([BLOCK_D], dtype=tl.float32)
        for s in tl.static_range(W):
            row = mine + (s * MAX_T + t) * D
            x += tl.load(row + d, mask=dm, other=0.0).to(tl.float32)
        x = x.to(partial_ptr.dtype.element_ty).to(tl.float32)
        idx = t * D + d[None, :]
        norm_residual_norm_tile(
            x[None, :],
            stream_ptr,
            new_stream_ptr,
            out_ptr,
            post_w_ptr,
            pre_w_ptr,
            alpha_ptr,
            beta_ptr,
            idx,
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
        )
    tl.store(peer(P, rank, tl.int32) + target_off + tl.program_id(0), target)


class MuseSparkAllReduceBoundary:
    """Peer-mapped workspace of the fused attention all-reduce + boundary.

    Each rank's buffer holds two ``[W][max_tokens][hidden]`` bf16 partial-row
    buffers, then one arrival counter and one barrier target per block.
    """

    def __init__(self, max_tokens: int, hidden: int, group: dist.ProcessGroup) -> None:
        self.max_tokens = max_tokens
        self.hidden = hidden
        self.world_size = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)
        self.buf_elems = self.world_size * max_tokens * hidden
        sync_off = 2 * self.buf_elems * 2
        self.count_off = sync_off // 4
        self.target_off = self.count_off + 1
        self.log2_arrivals = (self.world_size * NUM_BLOCKS).bit_length() - 1
        assert 1 << self.log2_arrivals == self.world_size * NUM_BLOCKS
        self.buffer = PeerBuffer(sync_off + (1 + NUM_BLOCKS) * 4, group)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fused_norm_residual_norm_fwd`` of ``all_reduce(partial)``; returns
        ``(out, new_stream)``. Collective over TP."""
        assert self.supports(partial.dtype, stream, post_norm, connection, pre_norm)
        assert partial.shape == stream.shape and partial.stride(-1) == 1
        num_tokens, hidden = stream.shape
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
            self.buffer.peers,
            self.rank,
            num_tokens,
            post_norm.variance_epsilon,
            pre_norm.variance_epsilon,
            post_norm.fused_gain_center,
            pre_norm.fused_gain_center,
            1.0 if gate_stream else float(alpha),
            self.buf_elems,
            self.count_off,
            self.target_off,
            W=self.world_size,
            D=hidden,
            MAX_T=self.max_tokens,
            NB=NUM_BLOCKS,
            LOG2_ARRIVALS=self.log2_arrivals,
            BLOCK_D=triton.next_power_of_2(hidden),
            POST_HAS_WEIGHT=post_w is not None,
            POST_INPUT_SCALE=post_scale,
            PRE_HAS_WEIGHT=pre_w is not None,
            PRE_INPUT_SCALE=pre_scale,
            GATE_STREAM=gate_stream,
            PER_CHANNEL=beta.numel() > 1,
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
        "(TP%d, up to %d tokens).",
        tp_group.world_size,
        workspace.max_tokens,
    )
    return workspace
