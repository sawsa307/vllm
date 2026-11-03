#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm MoE finalize: unfused finalize + TP all-reduce vs. fused.

The unfused arm is the production path below 256 tokens:

* pure TP: ``tensor_model_parallel_all_reduce`` of the ``[T, top_k, H]`` route
  rows in the reduce dtype, then ``fused_post_expert_finalize``;
* ``--ep`` (TP + EP over the same group): ``fused_post_expert_finalize`` of the
  local routes (``1 / TP`` of them), then the MoE runner's all-reduce of the
  ``[T, H]`` output.

Both arms are timed as graph replays of ``--calls`` back-to-back calls.

Usage:
    VLLM_ROCM_USE_AITER=1 torchrun --nproc_per_node=8 \
        benchmarks/kernels/benchmark_muse_spark_rocm_finalize_all_reduce.py [--ep]
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
    parser.add_argument("--ep", action="store_true", help="TP + EP finalize")
    parser.add_argument("--num-experts", type=int, default=256)
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
    fused = MuseSparkFinalizeAllReduce(
        max(args.tokens), K, H, tp.cpu_group, pre_norm=not args.ep
    )
    weight = torch.randn(H, device=device).bfloat16()
    per_rank = args.num_experts // world_size
    rank = tp.rank_in_group
    expert_map = torch.full((args.num_experts,), -1, dtype=torch.int32, device=device)
    expert_map[rank * per_rank : (rank + 1) * per_rank] = torch.arange(
        per_rank, dtype=torch.int32, device=device
    )

    if tp.rank_in_group == 0:
        mode = "EP" if args.ep else f"reduce={args.reduce_dtype}"
        print(f"TP{world_size} H={H} top_k={K} {mode}")
        print(f"{'tokens':>7} {'unfused us':>11} {'fused us':>9} {'speedup':>8}")
    for n in args.tokens:
        rows = torch.randn(n, K, H, device=device).bfloat16()
        tw = torch.rand(n, K, device=device)
        out_ref = torch.empty(n, H, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(out_ref)
        gen = torch.Generator(device=device).manual_seed(n)
        ids = torch.stack(
            [
                torch.randperm(args.num_experts, device=device, generator=gen)[:K]
                for _ in range(n)
            ]
        ).int()

        def unfused(rows=rows, tw=tw, ids=ids, out_ref=out_ref):
            if args.ep:
                fused_post_expert_finalize(
                    rows,
                    weight,
                    tw,
                    EPS,
                    1.0,
                    True,
                    out_ref,
                    topk_ids=ids,
                    expert_map=expert_map,
                )
                out_ref.copy_(tensor_model_parallel_all_reduce(out_ref))
                return
            reduced = tensor_model_parallel_all_reduce(rows.to(reduce_dtype))
            fused_post_expert_finalize(reduced, weight, tw, EPS, 1.0, True, out_ref)

        def fused_call(rows=rows, tw=tw, ids=ids, out=out):
            fused(
                rows,
                weight,
                tw,
                EPS,
                1.0,
                True,
                out,
                topk_ids=ids if args.ep else None,
                expert_map=expert_map if args.ep else None,
            )

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
