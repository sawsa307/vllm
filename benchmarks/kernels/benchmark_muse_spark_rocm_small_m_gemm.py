#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark decode / verify GEMMs on one GPU: hipBLASLt vs the small-M kernels.

For each TP4 projection shape (QKV + o-gate [4608, 8192], O [8192, 2048], pooler
[4096, 8192], unpooler [8192, 4096]) and row count, times as graph replays over
rotating weight copies (so the weight streams from HBM):

* ``F.linear`` on the BF16 weight (hipBLASLt);
* ``rowmajor_gemm`` (BF16 split-K, SGLang's MI355X config);
* ``fp8_gemm`` on the FP8 e4m3 weight with per-row FP8 activations, with the
  activation quantization counted separately (a fused producer absorbs it);

and checks each against an FP32 reference (BF16: within 2 bf16 ulp of the
hipBLASLt result's error; FP8: relative error of the quantization itself).

Usage (one GPU):
    python benchmarks/kernels/benchmark_muse_spark_rocm_small_m_gemm.py [--tp 4]
"""

import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd.ops.small_m_gemm import (
    fp8_gemm,
    fp8_gemm_config,
    quantize_rows,
    quantize_weight,
    rowmajor_gemm,
    rowmajor_gemm_config,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

HIDDEN, POOLED, HEADS, KV_HEADS, HEAD_DIM = 8192, 4096, 128, 16, 64


def _time(fn, copies: int, iters: int) -> float:
    for i in range(copies):
        fn(i)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(copies):
            fn(i)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iters * copies)


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64, 96, 128])
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda")
    tp = args.tp
    shapes = {
        "qkv+gate": (2 * (HEADS + KV_HEADS) * HEAD_DIM // tp, HIDDEN),
        "o_proj": (HIDDEN, HEADS * HEAD_DIM // tp),
        "pooler": (POOLED, HIDDEN),
        "unpooler": (HIDDEN, POOLED),
    }
    print(f"TP{tp}, us per call, {args.copies} rotating weight copies")
    print(
        f"{'shape':<10} {'N x K':>12} {'rows':>5} {'hipBLASLt':>10} {'bf16 rowmaj':>12}"
        f" {'fp8w':>8} {'+quant':>8}  {'bf16 ok':>7} {'fp8 relerr':>10}"
    )
    for name, (n, k) in shapes.items():
        weights = [
            (torch.randn(n, k, device=device) * 0.02).bfloat16()
            for _ in range(args.copies)
        ]
        fp8_weights = [quantize_weight(w) for w in weights]
        for m in args.rows:
            x = torch.randn(m, k, device=device).bfloat16()
            ref = x.float() @ weights[0].float().t()
            lin_us = _time(
                lambda i, x=x, ws=weights: F.linear(x, ws[i]), args.copies, args.iters
            )
            row = f"{name:<10} {n:>5}x{k:<6} {m:>5} {lin_us:>10.1f}"
            cfg = rowmajor_gemm_config(n, k, m)
            if cfg is not None:
                y = rowmajor_gemm(x, weights[0], cfg)
                lin_err = (F.linear(x, weights[0]).float() - ref).abs().max().item()
                ok = (y.float() - ref).abs().max().item() <= 2 * lin_err + 1e-3
                us = _time(
                    lambda i, x=x, cfg=cfg, ws=weights: rowmajor_gemm(x, ws[i], cfg),
                    args.copies,
                    args.iters,
                )
                row += f" {us:>12.1f}"
            else:
                ok = None
                row += f" {'-':>12}"
            cfg8 = fp8_gemm_config(n, k, m)
            if cfg8 is not None:
                xq, xs = quantize_rows(x)
                y8 = fp8_gemm(xq, xs, *fp8_weights[0], cfg8)
                rel = ((y8.float() - ref).norm() / ref.norm()).item()
                us8 = _time(
                    lambda i, xq=xq, xs=xs, cfg8=cfg8, w8=fp8_weights: fp8_gemm(
                        xq, xs, *w8[i], cfg8
                    ),
                    args.copies,
                    args.iters,
                )
                usq = _time(lambda i, x=x: quantize_rows(x), args.copies, args.iters)
                row += f" {us8:>8.1f} {usq:>8.1f}"
            else:
                rel = None
                row += f" {'-':>8} {'-':>8}"
            row += f"  {'' if ok is None else ('ok' if ok else 'BAD'):>7}"
            row += f" {'' if rel is None else f'{rel:.2e}':>10}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
