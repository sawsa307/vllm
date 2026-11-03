#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm MoE finalize: unfused TP all-reduce + finalize vs. fused.

The unfused arm is the production path of ``MuseSparkTopKWeightAndReduce``
below 256 tokens: ``tensor_model_parallel_all_reduce`` of the ``[T, top_k, H]``
route rows in the reduce dtype (AITER / custom all-reduce as configured), then
``fused_post_expert_finalize``. Both arms are timed as graph replays of
``--calls`` back-to-back calls.

Usage:
    VLLM_ROCM_USE_AITER=1 torchrun --nproc_per_node=4 \
        benchmarks/kernels/benchmark_muse_spark_rocm_finalize_all_reduce.py
"""

import os

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.models.muse_spark.amd.ops.fused_finalize_all_reduce import (
    MuseSparkFinalizeAllReduce,
)
from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

EPS = 1e-6


def _time_graph(fn, calls: int, iters: int) -> float:
    """Microseconds per call, from a graph of ``calls`` calls."""
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    device = torch.device("cuda", torch.accelerator.current_device_index())
    with graph_capture(device) as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        for _ in range(calls):
            fn()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iters * calls)


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 255]
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--reduce-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size, int(os.environ["RANK"]), local_rank=local_rank
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp = get_tp_group()
    reduce_dtype = torch.bfloat16 if args.reduce_dtype == "bf16" else torch.float32
    H, K = args.hidden_size, args.top_k
    fused = MuseSparkFinalizeAllReduce(max(args.tokens), K, H, tp.cpu_group)
    weight = torch.randn(H, device=device).bfloat16()

    if tp.rank_in_group == 0:
        print(f"TP{world_size} H={H} top_k={K} reduce={args.reduce_dtype}")
        print(f"{'tokens':>7} {'unfused us':>11} {'fused us':>9} {'speedup':>8}")
    for n in args.tokens:
        rows = torch.randn(n, K, H, device=device).bfloat16()
        tw = torch.rand(n, K, device=device)
        out_ref = torch.empty(n, H, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(out_ref)

        def unfused(rows=rows, tw=tw, out_ref=out_ref):
            reduced = tensor_model_parallel_all_reduce(rows.to(reduce_dtype))
            fused_post_expert_finalize(reduced, weight, tw, EPS, 1.0, True, out_ref)

        def fused_call(rows=rows, tw=tw, out=out):
            fused(rows, weight, tw, EPS, 1.0, True, out)

        unfused_us = _time_graph(unfused, args.calls, args.iters)
        fused_us = _time_graph(fused_call, args.calls, args.iters)
        unfused()
        fused_call()
        torch.accelerator.synchronize()
        err = (out.float() - out_ref.float()).abs().max().item()
        if tp.rank_in_group == 0:
            print(
                f"{n:>7} {unfused_us:>11.1f} {fused_us:>9.1f} "
                f"{unfused_us / fused_us:>7.2f}x  max|diff|={err:.3g}"
            )
    fused.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
