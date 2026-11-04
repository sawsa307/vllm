# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M GEMMs for the Muse Spark decode / verify widths on gfx950.

``y = x @ w.T`` for ``x [M, K]`` with M up to 128 against a row-major ``w [N, K]``,
split-K over programs that each stream BN weight rows: FP32 partials, summed by
a second kernel in a fixed order (the same products and accumulation as a plain
GEMM, only the K summation order differs). Two weight formats:

* BF16 as stored (``rowmajor_gemm``);
* FP8 e4m3 with one FP32 scale per output channel, quantized once at load
  (``quantize_weight``), against per-row FP8 activations the producer quantized
  (``fp8_gemm``): half the weight bytes on the same tiling.

Kernels and MI355X tuning tables ported from SGLang's ROCm Muse Spark path
(``sglang/kernels/ops/gemm/bf16_tiled_small_m_gemm.py`` and
``fp8w_small_m_gemm.py``, Apache-2.0). Its measurements there, HBM-resident
weights, 64 rows: BF16 row-major vs hipBLASLt pooler 17.8 vs 19.8 us, unpooler
16.6 vs 19.7, O 9.6 vs 12.3; FP8 QKV 23.4 -> 14.4, unpooler 16.5 -> 11.7,
pooler 17.6 -> 11.6, O 9.6 -> 8.5.
"""

import torch

from vllm.triton_utils import tl, triton

F8 = torch.float8_e4m3fn
F8_MAX = 448.0
MAX_M = 128

# (N, K, BM) -> (BN, BK, SK, warps, stages). Muse Spark TP4 shapes: QKV + o-gate
# [4608, 8192], O [8192, 2048], pooler [4096, 8192], unpooler [8192, 4096]; TP8
# QKV [2304, 8192], O [8192, 1024], pooler [2048, 8192], unpooler [8192, 2048] where
# tuned.
BF16_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
    (4096, 8192, 32): (64, 128, 8, 8, 2),
    (4096, 8192, 64): (64, 128, 8, 8, 2),
    (4096, 8192, 128): (64, 128, 4, 8, 3),
    (4608, 8192, 64): (128, 128, 4, 4, 3),
    (8192, 4096, 64): (32, 256, 1, 4, 3),
    (8192, 4096, 128): (64, 128, 2, 8, 3),
    (8192, 2048, 64): (32, 256, 1, 4, 3),
    (8192, 2048, 128): (32, 128, 1, 8, 3),
    (8192, 1024, 32): (32, 256, 1, 8, 3),
    (8192, 1024, 64): (32, 256, 1, 4, 3),
    (8192, 1024, 128): (32, 128, 1, 4, 3),
}
FP8_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
    (4608, 8192, 32): (128, 256, 4, 8, 3),
    (4608, 8192, 64): (128, 256, 4, 8, 3),
    (4608, 8192, 128): (64, 256, 2, 8, 3),
    (8192, 2048, 32): (32, 512, 1, 4, 3),
    (8192, 2048, 64): (32, 512, 1, 4, 3),
    (8192, 2048, 128): (32, 256, 1, 8, 3),
    (4096, 8192, 32): (64, 256, 4, 8, 3),
    (4096, 8192, 64): (64, 256, 4, 8, 3),
    (4096, 8192, 128): (64, 256, 4, 8, 3),
    (8192, 4096, 32): (64, 256, 2, 8, 3),
    (8192, 4096, 64): (64, 256, 2, 8, 3),
    (8192, 4096, 128): (64, 256, 2, 8, 3),
    (2304, 8192, 32): (64, 256, 4, 8, 3),
    (2304, 8192, 64): (64, 256, 4, 8, 3),
    (2304, 8192, 128): (64, 256, 4, 4, 3),
}


def block_m(m: int) -> int:
    return max(16, 1 << (m - 1).bit_length())


@triton.jit
def _rowmajor_partial(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    """FP32 partial ``[M, BN]`` of this program's K range (or the final BF16
    product when SK == 1); ``X`` and ``W`` share one element type."""
    pn, pk = tl.program_id(0), tl.program_id(1)
    KT_PER: tl.constexpr = K // BK // SK
    m = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    n = pn * BN + tl.arange(0, BN)
    bp = W + n[None, :].to(tl.int64) * K + (pk * KT_PER * BK + kk)[:, None]
    ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(KT_PER):
        a = tl.load(ap, mask=m[:, None] < M, other=0.0)
        b = tl.load(bp, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(
        P + pk * M * N + m[:, None] * N + n[None, :],
        acc.to(P.dtype.element_ty),
        mask=m[:, None] < M,
    )


@triton.jit
def _sum_partials(P, Y, MN, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SK)
    p = tl.load(P + s[:, None] * MN + i[None, :], mask=i[None, :] < MN, other=0.0)
    tl.store(Y + i, tl.sum(p, axis=0).to(Y.dtype.element_ty), mask=i < MN)


@triton.jit
def _sum_scaled(P, XS, WS, Y, M, N: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ok = i < M * N
    s = tl.arange(0, SK)
    p = tl.load(P + s[:, None] * M * N + i[None, :], mask=ok[None, :], other=0.0)
    y = tl.sum(p, axis=0)
    y = y * tl.load(XS + i // N, mask=ok, other=0.0)
    y = y * tl.load(WS + i % N, mask=ok, other=0.0)
    tl.store(Y + i, y.to(Y.dtype.element_ty), mask=ok)


def rowmajor_gemm_config(n: int, k: int, m: int):
    return BF16_CONFIGS.get((n, k, block_m(m))) if 0 < m <= MAX_M else None


def rowmajor_gemm(x: torch.Tensor, weight: torch.Tensor, config=None) -> torch.Tensor:
    """``x @ weight.T`` in BF16 for a contiguous BF16 ``weight [N, K]``."""
    m, k = x.shape
    n = weight.shape[0]
    if config is None:
        config = BF16_CONFIGS[(n, k, block_m(m))]
    bn, bk, sk, warps, stages = config
    assert x.is_contiguous() and weight.is_contiguous()
    assert n % bn == 0 and k % (bk * sk) == 0
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    p = y if sk == 1 else torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    _rowmajor_partial[(n // bn, sk)](
        x,
        weight,
        p,
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
    if sk > 1:
        _sum_partials[(triton.cdiv(m * n, 256),)](
            p, y, m * n, SK=sk, B=256, num_warps=4
        )
    return y


@torch.no_grad()
def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(FP8 e4m3 ``[N, K]``, FP32 per-output-channel scales ``[N]``) of a BF16
    weight."""
    scale = weight.float().abs().amax(1).clamp_min(1e-12) / F8_MAX
    return (weight.float() / scale[:, None]).to(F8).contiguous(), scale.contiguous()


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP8 e4m3 of ``x [M, K]`` and its FP32 scales ``[M]`` (torch; a fused
    producer does this in-kernel)."""
    scale = x.float().abs().amax(1).clamp_min(1e-12) / F8_MAX
    return (x.float() / scale[:, None]).to(F8), scale


def fp8_gemm_config(n: int, k: int, m: int):
    return FP8_CONFIGS.get((n, k, block_m(m))) if 0 < m <= MAX_M else None


def fp8_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    config=None,
) -> torch.Tensor:
    """``(xq * xs) @ (wq * ws).T`` in BF16 from per-row FP8 activations and the
    per-channel FP8 weight of :func:`quantize_weight`."""
    m, k = xq.shape
    n = wq.shape[0]
    if config is None:
        config = FP8_CONFIGS[(n, k, block_m(m))]
    bn, bk, sk, warps, stages = config
    assert xq.dtype == wq.dtype == F8 and xq.is_contiguous() and wq.is_contiguous()
    assert n % bn == 0 and k % (bk * sk) == 0
    p = torch.empty((sk, m, n), device=xq.device, dtype=torch.float32)
    _rowmajor_partial[(n // bn, sk)](
        xq,
        wq,
        p,
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
    y = torch.empty((m, n), device=xq.device, dtype=torch.bfloat16)
    _sum_scaled[(triton.cdiv(m * n, 512),)](
        p, xs, ws, y, m, N=n, SK=sk, B=512, num_warps=4
    )
    return y
