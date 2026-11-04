#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark unpooler + FFN boundary on one GPU: unfused vs fused.

Per row count, as graph replays over rotating unpooler weights (HBM-resident):

* unfused: ``F.linear`` (hipBLASLt) then ``fused_norm_residual_norm_fwd``;
* fused: ``unpool_boundary`` (split-K partials + one row program that sums,
  rounds and runs the same boundary math);

and the max difference between them (K summation order only).

Usage (one GPU):
    python benchmarks/kernels/benchmark_muse_spark_rocm_unpool_boundary.py
"""

import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd.ops.unpool_boundary import (
    unpool_boundary,
    unpool_boundary_supports,
)
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

HIDDEN, POOLED = 8192, 4096


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
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda")
    post = MuseSpark1xRMSNormWithInputScale(
        HIDDEN, eps=1e-5, has_weight=True, is_post_norm=True, zero_centered_gamma=True
    ).to(device)
    pre = MuseSpark1xRMSNorm(
        HIDDEN, eps=1e-5, has_weight=True, zero_centered_gamma=True
    ).to(device)
    connection = MuseSpark1xNormPreservingResidualConnection(HIDDEN).to(device)
    with torch.no_grad():
        for p in (post.weight, pre.weight, connection.gate):
            p.normal_(0, 0.1)
    connection.materialize_coefficients()
    weights = [
        (torch.randn(HIDDEN, POOLED, device=device) * 0.02).bfloat16()
        for _ in range(args.copies)
    ]
    print(f"us per call, {args.copies} rotating unpooler weights")
    print(
        f"{'rows':>5} {'unfused':>9} {'fused':>8} {'max|dout|':>10} "
        f"{'max|dstream|':>12}"
    )
    for m in args.rows:
        e = torch.randn(m, POOLED, device=device).bfloat16()
        stream = torch.randn(m, HIDDEN, device=device)

        def unfused(i, e=e, stream=stream):
            return fused_norm_residual_norm_fwd(
                F.linear(e, weights[i]),
                stream,
                post.weight,
                pre.weight,
                post.variance_epsilon,
                pre.variance_epsilon,
                alpha=connection.alpha,
                beta=connection.beta,
                post_gain_center=post.fused_gain_center,
                pre_gain_center=pre.fused_gain_center,
                post_input_scale=True,
                out_dtype=torch.bfloat16,
            )

        if not unpool_boundary_supports(e, weights[0], stream, post, connection, pre):
            print(f"{m:>5} {_time(unfused, args.copies, args.iters):>9.1f} {'-':>8}")
            continue

        def fused(i, e=e, stream=stream):
            return unpool_boundary(
                e, weights[i], stream, post, connection, pre, torch.bfloat16
            )

        ref_out, ref_stream = unfused(0)
        out, new_stream = fused(0)
        d_out = (out.float() - ref_out.float()).abs().max().item()
        d_stream = (new_stream - ref_stream).abs().max().item()
        t_u = _time(unfused, args.copies, args.iters)
        t_f = _time(fused, args.copies, args.iters)
        print(
            f"{m:>5} {t_u:>9.1f} {t_f:>8.1f} {d_out:>10.3g} {d_stream:>12.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
