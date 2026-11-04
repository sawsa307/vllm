# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact A4W4 MoE experts for small verify batches (Quark MXFP4, gfx950).

Muse Spark EP4 / EP8: 64 or 32 local experts, top-8, hidden 4096, intermediate
4096, per-32 E8M0 scales. For M rows (a multiple of 8, up to ``MAX_ROWS``):

1. ``_prepare_a4``: per-row MXFP4 quantization of the input (or the caller
   passes it, ``quantized``);
2. ``_scatter_a4``: one program per local expert places its 16-row blocks after
   the lower experts' and ranks its routes in token order into them;
3. ``_expert_a4`` (ACT): gate / up ``dot_scaled`` GEMMs, SiLU-and-mul rounded to
   BF16, MXFP4 re-quantization;
4. ``_expert_a4``: down GEMM into unweighted per-(token, slot) BF16 rows.

Weights are read as stored in the checkpoint or in AITER's a4w4 preshuffle
layout (``shuffled``). The caller's finalize applies the per-expert norm and
the route-weighted sum, so the per-slot rows ``[M * 8, 4096]`` are the same
contract as AITER's per-slot route output.

Ported from SGLang's ROCm Muse Spark ``mxfp4_compact_a4_moe`` (Apache-2.0).
"""

import torch
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

from vllm.triton_utils import tl, triton

MAX_ROWS = 512
BLOCK_M = 16  # BM 32 / 64 give wrong rows and are no faster (SGLang)
HIDDEN = 4096
TOP_K = 8


@triton.jit
def _prepare_a4(X, Q, QS):
    pid = tl.program_id(0)
    row = pid // 4
    tile = pid % 4
    col = tile * 1024 + tl.arange(0, 1024)
    x = tl.load(X + row * 4096 + col).to(tl.float32)
    q, qscale = _mxfp4_quant_op(x[None, :], 1024, 1, 32)
    packed_col = tile * 512 + tl.arange(0, 512)
    scale_col = tile * 32 + tl.arange(0, 32)
    tl.store(Q + row * 2048 + packed_col, tl.reshape(q, (512,)))
    tl.store(QS + row * 128 + scale_col, tl.reshape(qscale, (32,)))


@triton.jit
def _scatter_a4(
    IDS,
    SORTED,
    EXPERT,
    M: tl.constexpr,
    START: tl.constexpr,
    E: tl.constexpr,
    S: tl.constexpr,
    CH: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    NBB: tl.constexpr,
):
    e = tl.program_id(0)
    ee = tl.arange(0, E)
    counts = tl.zeros((E,), tl.int32)
    for c0 in tl.static_range(0, S, CH):
        sc = c0 + tl.arange(0, CH)
        idc = tl.load(IDS + sc, sc < M * 8, other=-1)
        counts += tl.sum(((idc[:, None] - START) == ee[None, :]).to(tl.int32), axis=0)
    blocks_all = (counts + BM - 1) // BM
    first = tl.sum(tl.where(ee < e, blocks_all, 0), axis=0)
    if e == E - 1:
        b = tl.arange(0, NBB)
        tl.store(EXPERT + b, -1, (b >= tl.sum(blocks_all, axis=0)) & (b < NB))
    s = tl.arange(0, S)
    ids = tl.load(IDS + s, s < M * 8, other=-1)
    hit = (ids == START + e) & (s < M * 8)
    count = tl.sum(hit.to(tl.int32), axis=0)
    blocks = tl.cdiv(count, BM)
    if blocks > 0:
        eb = tl.arange(0, PER_E)
        tl.store(EXPERT + first + eb, e, eb < blocks)
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        base = first * BM
        c = tl.arange(0, PER_E * BM)
        tl.store(SORTED + base + c, M * 8, c < blocks * BM)
        tl.debug_barrier()
        tl.store(SORTED + base + rank, s, hit)


@triton.jit
def _load_w(W, n0, k0, BN: tl.constexpr, KB: tl.constexpr, SHUF: tl.constexpr):
    """``[KB bytes, BN rows]`` tile of one expert's ``[N, 2048]`` packed E2M1 rows
    from row ``n0``, byte ``k0``. Preshuffled: ``[N/16, 2048/16, 16 rows, 16
    bytes]`` (AITER shuffle_weight (16, 16)), loaded as contiguous 16-row groups
    and permuted in registers."""
    if SHUF:
        g = tl.arange(0, BN // 16)
        kc = tl.arange(0, KB // 16)
        r = tl.arange(0, 16)
        b = tl.arange(0, 16)
        offset = (
            ((n0 // 16 + g) * 32768)[:, None, None, None]
            + ((k0 // 16 + kc) * 256)[None, :, None, None]
            + (r * 16)[None, None, :, None]
            + b[None, None, None, :]
        )
        x = tl.permute(tl.load(W + offset), (1, 3, 0, 2))
        return tl.reshape(x, (KB, BN))
    n = n0 + tl.arange(0, BN)
    k = k0 + tl.arange(0, KB)
    return tl.load(W + n[None, :] * 2048 + k[:, None])


@triton.jit
def _s_offset(n, c, SHUF: tl.constexpr):
    """``[N, scale col]`` offsets in one expert's ``[N, 128]`` E8M0 scales.
    Preshuffled (AITER shuffle_scale): ``[N/32, 128/8, 4 (c % 4), 16 (n % 16),
    2 ((c % 8) // 4), 2 ((n % 32) // 16)]``."""
    if SHUF:
        n2, c2 = n[:, None], c[None, :]
        return (
            ((((n2 // 32) * 16 + c2 // 8) * 4 + c2 % 4) * 16 + n2 % 16) * 4
            + ((c2 % 8) // 4) * 2
            + (n2 % 32) // 16
        )
    return n[:, None] * 128 + c[None, :]


@triton.jit
def _expert_a4(
    A,
    AS,
    W,
    WS,
    SORTED,
    EXPERT,
    OUT,
    OS,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACT: tl.constexpr,
    BM: tl.constexpr,
    SHUF: tl.constexpr,
):
    expert_block, n_block = tl.program_id(1), tl.program_id(0)
    expert = tl.load(EXPERT + expert_block)
    if expert >= 0:
        slot = tl.load(SORTED + expert_block * BM + tl.arange(0, BM))
        valid = slot < ROWS
        src = slot // 8 if ACT else slot
        n = n_block * BN + tl.arange(0, BN)
        kp = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 32)
        wn: tl.constexpr = 8192 if ACT else 4096
        acc = tl.full((BM, BN), 0, tl.float32)
        if ACT:
            up = tl.full((BM, BN), 0, tl.float32)
        for tile in range(4096 // BK):
            a = tl.load(
                A + src[:, None] * 2048 + tile * (BK // 2) + kp[None, :],
                valid[:, None],
                other=0,
            )
            asc = tl.load(
                AS + src[:, None] * 128 + tile * (BK // 32) + ks[None, :],
                valid[:, None],
                other=127,
            )
            w = _load_w(
                W + expert * (wn * 2048),
                n_block * BN,
                tile * (BK // 2),
                BN,
                BK // 2,
                SHUF,
            )
            ws = tl.load(
                WS + expert * (wn * 128) + _s_offset(n, tile * (BK // 32) + ks, SHUF)
            )
            acc = tl.dot_scaled(a, asc, "e2m1", w, ws, "e2m1", acc)
            if ACT:
                wu = _load_w(
                    W + expert * (wn * 2048),
                    n_block * BN + 4096,
                    tile * (BK // 2),
                    BN,
                    BK // 2,
                    SHUF,
                )
                wsu = tl.load(
                    WS
                    + expert * (wn * 128)
                    + _s_offset(n + 4096, tile * (BK // 32) + ks, SHUF)
                )
                up = tl.dot_scaled(a, asc, "e2m1", wu, wsu, "e2m1", up)
        if ACT:
            activated = (acc * tl.sigmoid(acc) * up).to(tl.bfloat16).to(tl.float32)
            q, s = _mxfp4_quant_op(activated, BN, BM, 32)
            pn = n_block * (BN // 2) + tl.arange(0, BN // 2)
            sn = n_block * (BN // 32) + tl.arange(0, BN // 32)
            tl.store(OUT + slot[:, None] * 2048 + pn[None, :], q, valid[:, None])
            tl.store(OS + slot[:, None] * 128 + sn[None, :], s, valid[:, None])
        else:
            # The BF16 per-slot boundary before the Muse finalize.
            tl.store(OUT + slot[:, None] * 4096 + n[None, :], acc, valid[:, None])


def compact_a4_applies(m: int, h: int, w13, w2, s13, s2) -> bool:
    e = w13.shape[0]
    return (
        0 < m <= MAX_ROWS
        and m % 8 == 0
        and h == HIDDEN
        and e in (32, 64)
        and w13.shape == (e, 8192, 2048)
        and w2.shape == (e, 4096, 2048)
        and s13.shape == (e, 8192, 128)
        and s2.shape == (e, 4096, 128)
        and all(
            t.dtype == torch.uint8 and t.is_contiguous() for t in (w13, w2, s13, s2)
        )
    )


def compact_a4_experts(
    x: torch.Tensor | None,
    ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
    *,
    local_start: int,
    shuffled: bool = False,
    quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Per-(token, slot) expert outputs BF16 ``[M * 8, 4096]`` for the local
    experts starting at ``local_start``; routes to other ranks' experts stay
    unwritten (the caller masks them). ``quantized`` = the input's per-row MXFP4
    (uint8 ``[M, 2048]``, uint8 ``[M, 128]``) in place of quantizing ``x``."""
    m, h = ids.shape[0], w2.shape[1]
    assert compact_a4_applies(m, h, w13, w2, s13, s2) and ids.shape == (m, TOP_K)
    assert quantized is not None or (x is not None and x.shape == (m, h))
    dev = ids.device
    bm, e_local = BLOCK_M, w13.shape[0]
    rows, cap = m * TOP_K, m * TOP_K + e_local * (bm - 1)
    nb = triton.cdiv(cap, bm)
    sid = torch.empty(cap, device=dev, dtype=torch.int32)
    eid = torch.empty(nb, device=dev, dtype=torch.int32)
    inter = torch.empty((rows, 2048), device=dev, dtype=torch.uint8)
    inter_scale = torch.empty((rows, 128), device=dev, dtype=torch.uint8)
    y = torch.empty((rows, h), device=dev, dtype=torch.bfloat16)
    s_pow2 = triton.next_power_of_2(rows)
    per_e = triton.next_power_of_2(triton.cdiv(rows, bm))
    if quantized is None:
        q = torch.empty((m, 2048), device=dev, dtype=torch.uint8)
        qs = torch.empty((m, 128), device=dev, dtype=torch.uint8)
        _prepare_a4[(m * 4,)](x, q, qs, num_warps=4)
    else:
        q, qs = quantized
    _scatter_a4[(e_local,)](
        ids,
        sid,
        eid,
        M=m,
        START=local_start,
        E=e_local,
        S=s_pow2,
        CH=min(s_pow2, 256),
        PER_E=per_e,
        BM=bm,
        NB=nb,
        NBB=triton.next_power_of_2(nb),
        num_warps=4,
    )
    grid = (HIDDEN // 64, nb)
    options = dict(num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    _expert_a4[grid](
        q, qs, w13, s13, sid, eid, inter, inter_scale,
        ROWS=rows, BN=64, BK=512, ACT=True, BM=bm, SHUF=shuffled, **options,
    )  # fmt: skip
    _expert_a4[grid](
        inter, inter_scale, w2, s2, sid, eid, y, y,
        ROWS=rows, BN=64, BK=512, ACT=False, BM=bm, SHUF=shuffled, **options,
    )  # fmt: skip
    return y
