# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE routing front fused: FP32-exact router GEMM, sigmoid top-k routing,
pooler GEMM and pre-expert RMSNorm with one shared epilogue (decode widths).

Three launches for ``x [M, 8192]`` BF16: the router's split-K partials
(``router_bf16x3``: three BF16 planes that sum exactly to the FP32 weight), the
pooler's FP32 split-K partials (:mod:`.small_m_gemm`), and one epilogue whose
programs either route a row (sum the router splits, sigmoid, bias-corrected
top-k, normalized weights, as ``fused_musespark1x_routing``) or normalize a
row (sum the pooler splits, round to BF16 as the GEMM store would, RMSNorm,
written ``repeats`` times per row in the layout the AITER MXFP4 experts take,
as ``fused_rmsnorm_expand``). Replaces router GEMM + split reduce + routing +
pooler GEMM + norm (five launches) with three.

Design from SGLang's ROCm Muse Spark ``router_pooler_topk`` (Apache-2.0).
"""

import torch

from vllm.triton_utils import tl, triton

from .fused_norm_expand import _EXPANDED_ATTR
from .router_bf16x3 import _config as _router_config
from .router_bf16x3 import _router_bf16x3_kernel
from .small_m_gemm import _rowmajor_partial, block_m

MAX_TOKENS = 128
NUM_EXPERTS = 256
TOP_K = 8
# (BM) -> (BN, BK, SK, warps, stages) of the pooler [4096, 8192] partials.
_POOLER_CONFIGS: dict[int, tuple[int, int, int, int, int]] = {
    16: (64, 128, 8, 8, 2),
    32: (64, 128, 8, 8, 2),
    64: (64, 128, 8, 8, 2),
    128: (64, 128, 4, 8, 3),
}


@triton.jit
def _route_row(
    part_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    t,
    M,
    multiplier,
    route_scale,
    division_eps,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    K: tl.constexpr,
):
    n = tl.arange(0, N)
    logits = tl.zeros((N,), tl.float32)
    for s in tl.static_range(SPLITS):
        logits += tl.load(part_ptr + (s * M + t) * N + n)
    scores = tl.sigmoid(logits * multiplier)
    biased = scores + tl.load(bias_ptr + n).to(tl.float32)
    ks = tl.arange(0, K)
    ids = tl.zeros((K,), tl.int32)
    selected = tl.zeros((K,), tl.float32)
    denom = tl.zeros((), tl.float32)
    for k in tl.static_range(K):
        idx = tl.argmax(biased, axis=0)
        hit = n == idx
        score = tl.sum(tl.where(hit, scores, 0.0), axis=0)
        denom += score
        biased = tl.where(hit, float("-inf"), biased)
        ids = tl.where(ks == k, idx, ids)
        selected = tl.where(ks == k, score, selected)
    weights = selected / (denom + division_eps) * route_scale
    tl.store(weights_ptr + t * K + ks, weights)
    tl.store(ids_ptr + t * K + ks, ids)


@triton.jit
def _norm_row(
    part_ptr,
    w_ptr,
    out_ptr,
    t,
    M,
    eps,
    gain_center,
    D: tl.constexpr,
    SK: tl.constexpr,
    REPEATS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    d = tl.arange(0, D)
    s = tl.arange(0, SK)
    part = tl.load(part_ptr + (s[:, None] * M + t) * D + d[None, :])
    x = tl.sum(part, axis=0).to(tl.bfloat16).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + d).to(tl.float32) + gain_center
        if INPUT_SCALE:
            x = x * w
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
    if HAS_WEIGHT and not INPUT_SCALE:
        y = y * w
    y = y.to(out_ptr.dtype.element_ty)
    for r in tl.static_range(REPEATS):
        tl.store(out_ptr + (t * REPEATS + r) * D + d, y)


@triton.jit
def _router_pooler_epilogue(
    router_part_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    pooler_part_ptr,
    norm_w_ptr,
    out_ptr,
    M,
    multiplier,
    route_scale,
    division_eps,
    eps,
    gain_center,
    N: tl.constexpr,
    ROUTER_SPLITS: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    SK: tl.constexpr,
    REPEATS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        _route_row(
            router_part_ptr, bias_ptr, weights_ptr, ids_ptr, pid, M,
            multiplier, route_scale, division_eps, N, ROUTER_SPLITS, K,
        )  # fmt: skip
    else:
        _norm_row(
            pooler_part_ptr, norm_w_ptr, out_ptr, pid - M, M, eps, gain_center,
            D, SK, REPEATS, HAS_WEIGHT, INPUT_SCALE,
        )  # fmt: skip


def router_pooler_supports(
    x: torch.Tensor, planes: torch.Tensor, pooler_weight
) -> bool:
    m, k = x.shape
    return (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and 0 < m <= MAX_TOKENS
        and planes.shape == (3, NUM_EXPERTS, k)
        and pooler_weight.dtype == torch.bfloat16
        and pooler_weight.is_contiguous()
        and pooler_weight.shape == (4096, 8192)
        and block_m(m) in _POOLER_CONFIGS
    )


def router_pooler_topk(
    x: torch.Tensor,
    planes: torch.Tensor,
    pooler_weight: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor | None,
    *,
    eps: float,
    gain_center: float,
    input_scale: bool,
    repeats: int,
    multiplier: float,
    route_scale: float,
    division_eps: float,
    top_k: int = TOP_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(topk_weights fp32 [M, top_k], topk_ids int32 [M, top_k],
    pooled)``, ``pooled`` being the first-route view of the ``[M * repeats,
    4096]`` BF16 buffer as :func:`fused_rmsnorm_expand` returns it."""
    assert router_pooler_supports(x, planes, pooler_weight)
    assert bias.is_contiguous() and bias.numel() == NUM_EXPERTS
    m, k = x.shape
    n = pooler_weight.shape[0]
    rbm, rbn, rbk, rsplits, rwarps = _router_config(m)
    router_part = torch.empty(
        (rsplits, m, NUM_EXPERTS), dtype=torch.float32, device=x.device
    )
    _router_bf16x3_kernel[(NUM_EXPERTS // rbn, rsplits, triton.cdiv(m, rbm))](
        x,
        planes,
        router_part,
        m,
        N=NUM_EXPERTS,
        K=k,
        K_PER_SPLIT=k // rsplits,
        BLOCK_M=rbm,
        BLOCK_N=rbn,
        BLOCK_K=rbk,
        num_warps=rwarps,
    )
    bn, bk, sk, warps, stages = _POOLER_CONFIGS[block_m(m)]
    pooler_part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    _rowmajor_partial[(n // bn, sk)](
        x,
        pooler_weight,
        pooler_part,
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
    topk_weights = torch.empty((m, top_k), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((m, top_k), dtype=torch.int32, device=x.device)
    out = torch.empty((m, repeats, n), dtype=torch.bfloat16, device=x.device)
    _router_pooler_epilogue[(2 * m,)](
        router_part,
        bias,
        topk_weights,
        topk_ids,
        pooler_part,
        norm_weight if norm_weight is not None else x,
        out,
        m,
        multiplier,
        route_scale,
        division_eps,
        eps,
        gain_center,
        N=NUM_EXPERTS,
        ROUTER_SPLITS=rsplits,
        K=top_k,
        D=n,
        SK=sk,
        REPEATS=repeats,
        HAS_WEIGHT=norm_weight is not None,
        INPUT_SCALE=input_scale and norm_weight is not None,
        num_warps=8,
    )
    first = out[:, 0]
    setattr(first, _EXPANDED_ATTR, out)
    return topk_weights, topk_ids, first
