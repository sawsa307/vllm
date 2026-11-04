#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark MoE routing front on one GPU: five launches vs the fused three.

Per row count, as graph replays over rotating pooler weights (HBM-resident):

* unfused: ``router_bf16x3_gemm`` + ``fused_musespark1x_routing`` +
  ``F.linear`` (pooler, hipBLASLt) + ``fused_rmsnorm_expand`` (x top_k);
* fused: ``router_pooler_topk``;

and checks the expert ids agree, the route weights and the pooled rows match
within the split-K summation order.

Usage (one GPU):
    python benchmarks/kernels/benchmark_muse_spark_rocm_router_pooler.py
"""

import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
    fused_rmsnorm_expand,
    take_expanded,
)
from vllm.models.muse_spark.amd.ops.router_bf16x3 import (
    router_bf16x3_gemm,
    split_fp32_to_bf16x3,
)
from vllm.models.muse_spark.amd.ops.router_pooler import router_pooler_topk
from vllm.models.muse_spark.common.ops.fused_routing import (
    fused_musespark1x_routing,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

HIDDEN, POOLED, EXPERTS, TOP_K = 8192, 4096, 256, 8
MULT, ROUTE_SCALE, DIV_EPS, EPS, GAIN = 0.18, 1.0, 1e-15, 1e-5, 1.0


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
    router_w = torch.randn(EXPERTS, HIDDEN, device=device) * 0.02
    planes = split_fp32_to_bf16x3(router_w)
    bias = torch.randn(EXPERTS, device=device) * 0.01
    norm_w = (torch.randn(POOLED, device=device) * 0.1).bfloat16()
    poolers = [
        (torch.randn(POOLED, HIDDEN, device=device) * 0.02).bfloat16()
        for _ in range(args.copies)
    ]
    print(f"us per call, {args.copies} rotating pooler weights, top_k={TOP_K}")
    print(
        f"{'rows':>5} {'unfused':>9} {'fused':>8} {'ids':>5} {'max|dw|':>9} "
        f"{'max|dpool|':>11}"
    )
    for m in args.rows:
        x = torch.randn(m, HIDDEN, device=device).bfloat16()

        def unfused(i, x=x):
            logits = router_bf16x3_gemm(x, planes)
            w, ids = fused_musespark1x_routing(
                logits,
                bias,
                TOP_K,
                ROUTE_SCALE,
                router_output_multiplier=MULT,
                router_score_division_eps=DIV_EPS,
            )
            pooled = fused_rmsnorm_expand(
                F.linear(x, poolers[i]), norm_w, EPS, GAIN, False, TOP_K
            )
            return w, ids, pooled

        def fused(i, x=x):
            return router_pooler_topk(
                x,
                planes,
                poolers[i],
                bias,
                norm_w,
                eps=EPS,
                gain_center=GAIN,
                input_scale=False,
                repeats=TOP_K,
                multiplier=MULT,
                route_scale=ROUTE_SCALE,
                division_eps=DIV_EPS,
            )

        w0, ids0, p0 = unfused(0)
        w1, ids1, p1 = fused(0)
        same_ids = torch.equal(ids0.sort(1).values.int(), ids1.sort(1).values)
        order = ids0.argsort(1), ids1.argsort(1)
        dw = (w0.gather(1, order[0]) - w1.gather(1, order[1])).abs().max().item()
        e0, e1 = take_expanded(p0, TOP_K), take_expanded(p1, TOP_K)
        dp = (e0.float() - e1.float()).abs().max().item()
        t_u = _time(unfused, args.copies, args.iters)
        t_f = _time(fused, args.copies, args.iters)
        print(
            f"{m:>5} {t_u:>9.1f} {t_f:>8.1f} {'ok' if same_ids else 'DIFF':>5} "
            f"{dw:>9.2e} {dp:>11.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
