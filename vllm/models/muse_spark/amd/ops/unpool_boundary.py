# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE unpooler GEMM fused with the FFN residual boundary (decode widths).

Two launches: the FP32 split-K partials of ``experts_out @ W_unpool.T``
(:mod:`.small_m_gemm`), then one program per row that sums the splits, rounds
to BF16 as the GEMM store would, and runs the boundary through
``norm_residual_norm_tile`` (post-FFN norm, gated FP32 residual, the next
layer's pre-attention norm). Both BF16 rounding points of the unfused
sequence are kept, so the result matches ``fused_norm_residual_norm_fwd`` on
the hipBLASLt product up to the K summation order.

Design from SGLang's ROCm Muse Spark ``unpool_gemm_norm_residual``
(Apache-2.0); the boundary math is vLLM's own tile helper.
"""

import torch

from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import norm_residual_norm_tile
from vllm.triton_utils import tl, triton

from .small_m_gemm import _rowmajor_partial, block_m

MAX_TOKENS = 128
# (N, K, BM) -> (BN, BK, SK, warps, stages) of the partials launch; SK >= 2 so
# the epilogue always sums FP32 partials (SGLang: 64 / 128 / 4 at <= 64 rows).
_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
    (8192, 4096, 16): (64, 128, 4, 4, 2),
    (8192, 4096, 32): (64, 128, 4, 4, 2),
    (8192, 4096, 64): (64, 128, 4, 4, 2),
    (8192, 4096, 128): (64, 128, 2, 8, 3),
}


@triton.jit
def _unpool_boundary_kernel(
    part_ptr,
    stream_ptr,
    new_stream_ptr,
    out_ptr,
    post_w_ptr,
    pre_w_ptr,
    alpha_ptr,
    beta_ptr,
    router_ptr,
    T,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    D: tl.constexpr,
    SK: tl.constexpr,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    ROUTER_OUT: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, D)
    dm = d < D
    s = tl.arange(0, SK)
    part = tl.load(part_ptr + s[:, None] * T * D + t * D + d[None, :])
    x = tl.sum(part, axis=0).to(tl.bfloat16).to(tl.float32)
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


def unpool_boundary_supports(
    x: torch.Tensor,
    weight: torch.Tensor,
    stream: torch.Tensor,
    post_norm: MuseSpark1xRMSNorm,
    connection: object,
    pre_norm: MuseSpark1xRMSNorm,
) -> bool:
    m, k = x.shape
    n = weight.shape[0]
    return (
        isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and 0 < m <= MAX_TOKENS
        and (n, k, block_m(m)) in _CONFIGS
        and stream.dtype == torch.float32
        and stream.is_contiguous()
        and stream.shape == (m, n)
        and post_norm.variance_size_override is None
        and pre_norm.variance_size_override is None
        and post_norm.hidden_size == n == pre_norm.hidden_size
    )


def unpool_boundary(
    x: torch.Tensor,
    weight: torch.Tensor,
    stream: torch.Tensor,
    post_norm: MuseSpark1xRMSNorm,
    connection: MuseSpark1xNormPreservingResidualConnection,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype,
    router_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fused_norm_residual_norm_fwd(x @ weight.T, stream, ...)``; returns
    ``(out, new_stream)``. ``router_out`` (fp32, if given) receives the
    unrounded pre-norm result."""
    assert unpool_boundary_supports(x, weight, stream, post_norm, connection, pre_norm)
    m, k = x.shape
    n = weight.shape[0]
    bn, bk, sk, warps, stages = _CONFIGS[(n, k, block_m(m))]
    part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    _rowmajor_partial[(n // bn, sk)](
        x,
        weight,
        part,
        m,
        N=n,
        K=k,
        BM=block_m(m),
        BN=bn,
        BK=bk,
        SK=sk,
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    alpha, beta = connection.alpha, connection.beta
    gate_stream = isinstance(alpha, torch.Tensor)
    post_w = post_norm.weight if post_norm.has_weight else None
    pre_w = pre_norm.weight if pre_norm.has_weight else None
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    new_stream = torch.empty_like(stream)
    _unpool_boundary_kernel[(m,)](
        part,
        stream,
        new_stream,
        out,
        post_w if post_w is not None else stream,
        pre_w if pre_w is not None else stream,
        alpha if gate_stream else beta,
        beta,
        router_out if router_out is not None else stream,
        m,
        post_norm.variance_epsilon,
        pre_norm.variance_epsilon,
        post_norm.fused_gain_center,
        pre_norm.fused_gain_center,
        1.0 if gate_stream else float(alpha),
        D=n,
        SK=sk,
        POST_HAS_WEIGHT=post_w is not None,
        POST_INPUT_SCALE=post_w is not None and post_norm.applies_weight_before_norm,
        PRE_HAS_WEIGHT=pre_w is not None,
        PRE_INPUT_SCALE=pre_w is not None and pre_norm.applies_weight_before_norm,
        GATE_STREAM=gate_stream,
        PER_CHANNEL=beta.numel() > 1,
        ROUTER_OUT=router_out is not None,
        num_warps=16,
        num_stages=1,
    )
    return out, new_stream
