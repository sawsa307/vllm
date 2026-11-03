#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm attention all-reduce + residual boundary: unfused vs. fused.

Times, per decode batch size, the reduction of the ``o_proj`` partial
``[T, 8192]`` bf16 rows together with the boundary that follows it (post-
attention norm, gated FP32 residual, pre-FFN norm), as graph replays of
``--calls`` back-to-back calls:

* unfused: ``tensor_model_parallel_all_reduce`` + ``fused_norm_residual_norm_fwd``;
* ``MuseSparkAllReduceBoundary`` one-shot (up to ``--one-shot-max`` tokens);
* ``MuseSparkAllReduceBoundary`` two-shot.

The crossover between the two fused variants is what
``VLLM_ROCM_MUSE_ATTN_ALL_REDUCE_ONE_SHOT_MAX_TOKENS`` should be set to.

Usage:
    torchrun --nproc_per_node=4 \\
        benchmarks/kernels/benchmark_muse_spark_rocm_attn_all_reduce.py
"""

import os

import torch
import torch.distributed as dist
from benchmark_muse_spark_rocm_column_gather import _time_graph

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
)
from vllm.models.muse_spark.amd.ops.all_reduce_boundary import (
    MuseSparkAllReduceBoundary,
)
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

HIDDEN = 8192


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--one-shot-max", type=int, default=256)
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
    op = MuseSparkAllReduceBoundary(
        max(args.tokens), HIDDEN, tp.cpu_group, args.one_shot_max
    )
    post = MuseSpark1xRMSNorm(HIDDEN, eps=1e-5, has_weight=False, is_post_norm=True).to(
        device
    )
    pre = MuseSpark1xRMSNorm(
        HIDDEN, eps=1e-5, has_weight=True, zero_centered_gamma=True
    ).to(device)
    connection = MuseSpark1xNormPreservingResidualConnection(HIDDEN).to(device)
    with torch.no_grad():
        pre.weight.normal_(0, 0.1)
        connection.gate.normal_(0, 0.1)
    connection.materialize_coefficients()

    if tp.rank_in_group == 0:
        print(f"TP{world_size}, us per call (all-reduce + boundary)")
        print(f"{'tokens':>7} {'unfused':>9} {'one-shot':>9} {'two-shot':>9}")
    for n in args.tokens:
        partial = torch.randn(n, HIDDEN, device=device).bfloat16()
        stream = torch.randn(n, HIDDEN, device=device)

        def unfused(i, partial=partial, stream=stream):
            fused_norm_residual_norm_fwd(
                tensor_model_parallel_all_reduce(partial),
                stream,
                None,
                pre.weight,
                post.variance_epsilon,
                pre.variance_epsilon,
                alpha=connection.alpha,
                beta=connection.beta,
                pre_gain_center=pre.fused_gain_center,
                out_dtype=torch.bfloat16,
            )

        def fused(two_shot, partial=partial, stream=stream):
            return lambda i: op(
                partial,
                stream,
                post,
                connection,
                pre,
                torch.bfloat16,
                two_shot=two_shot,
            )

        times = [_time_graph(unfused, args.calls, args.iters)]
        if n <= op.one_shot_max_tokens:
            times.append(_time_graph(fused(False), args.calls, args.iters))
        else:
            times.append(float("nan"))
        times.append(_time_graph(fused(True), args.calls, args.iters))
        if tp.rank_in_group == 0:
            print(f"{n:>7} {times[0]:>9.1f} {times[1]:>9.1f} {times[2]:>9.1f}")
    op.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
