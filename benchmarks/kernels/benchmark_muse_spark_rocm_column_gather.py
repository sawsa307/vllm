#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm MoE projections: replicated vs. column-sharded + fused gather.

Times, per decode batch size, the projection together with the norm it feeds,
as graph replays of ``--calls`` back-to-back calls:

* pooler (8192 -> 4096) + pre-expert norm (x top_k rows for the AITER MXFP4
  experts): full ``F.linear`` + ``fused_rmsnorm_expand`` vs. this rank's
  columns + ``MuseSparkColumnGather.gather_rmsnorm``;
* unpooler (4096 -> 8192) + residual boundary: full ``F.linear`` +
  ``fused_norm_residual_norm_fwd`` vs. this rank's columns +
  ``MuseSparkColumnGather.gather_boundary``.

The weights are allocated per layer (``--layers`` copies, cycled) so that
they stream from HBM as in a forward instead of staying in cache.

Usage:
    torchrun --nproc_per_node=4 \\
        benchmarks/kernels/benchmark_muse_spark_rocm_column_gather.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.models.muse_spark.amd.ops.column_gather import MuseSparkColumnGather
from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand
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


def _time_graph(fn, calls: int, iters: int) -> float:
    """Microseconds per call, from a graph of ``calls`` calls."""
    for _ in range(3):
        fn(0)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    device = torch.device("cuda", torch.accelerator.current_device_index())
    with graph_capture(device) as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        for i in range(calls):
            fn(i)
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
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--calls", type=int, default=16)
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
    rank = tp.rank_in_group
    gather = MuseSparkColumnGather(max(args.tokens), HIDDEN, tp.cpu_group)

    pre_expert = MuseSpark1xRMSNorm(
        POOLED, eps=1e-5, has_weight=True, zero_centered_gamma=True
    ).to(device)
    post = MuseSpark1xRMSNormWithInputScale(
        HIDDEN, eps=1e-5, has_weight=True, is_post_norm=True, zero_centered_gamma=True
    ).to(device)
    pre = MuseSpark1xRMSNorm(
        HIDDEN, eps=1e-5, has_weight=True, zero_centered_gamma=True
    ).to(device)
    connection = MuseSpark1xNormPreservingResidualConnection(HIDDEN).to(device)
    with torch.no_grad():
        for p in (pre_expert.weight, post.weight, pre.weight, connection.gate):
            p.normal_(0, 0.1)
    connection.materialize_coefficients()
    pooler = [
        torch.randn(POOLED, HIDDEN, device=device).bfloat16()
        for _ in range(args.layers)
    ]
    unpooler = [
        torch.randn(HIDDEN, POOLED, device=device).bfloat16()
        for _ in range(args.layers)
    ]
    pb, ub = POOLED // world_size, HIDDEN // world_size

    if rank == 0:
        print(f"TP{world_size} top_k={args.top_k}, us per call (projection + norm)")
        print(
            f"{'tokens':>7} {'pool full':>10} {'pool shard':>11} "
            f"{'unpool full':>12} {'unpool shard':>13}"
        )

    def layer(i: int) -> int:
        return i % args.layers

    for n in args.tokens:
        x = torch.randn(n, HIDDEN, device=device).bfloat16()
        e = torch.randn(n, POOLED, device=device).bfloat16()
        stream = torch.randn(n, HIDDEN, device=device)

        def pool_full(i, x=x):
            fused_rmsnorm_expand(
                F.linear(x, pooler[layer(i)]),
                pre_expert.weight,
                pre_expert.variance_epsilon,
                pre_expert.fused_gain_center,
                False,
                args.top_k,
            )

        def pool_shard(i, x=x):
            w = pooler[layer(i)][rank * pb : (rank + 1) * pb]
            gather.gather_rmsnorm(F.linear(x, w), pre_expert, args.top_k)

        def unpool_full(i, e=e, stream=stream):
            fused_norm_residual_norm_fwd(
                F.linear(e, unpooler[layer(i)]),
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

        def unpool_shard(i, e=e, stream=stream):
            w = unpooler[layer(i)][rank * ub : (rank + 1) * ub]
            gather.gather_boundary(
                F.linear(e, w), stream, post, connection, pre, torch.bfloat16
            )

        times = [
            _time_graph(fn, args.calls, args.iters)
            for fn in (pool_full, pool_shard, unpool_full, unpool_shard)
        ]
        if rank == 0:
            print(
                f"{n:>7} {times[0]:>10.1f} {times[1]:>11.1f} "
                f"{times[2]:>12.1f} {times[3]:>13.1f}"
            )
    gather.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
