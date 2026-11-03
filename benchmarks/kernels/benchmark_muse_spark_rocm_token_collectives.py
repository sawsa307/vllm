#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark token-sharded collectives: RCCL vs. AITER custom (chunked).

Times, per batch size, the token all-gathers and reduce-scatters one decoder
layer of the token-sharded stream issues (eager, as these batch sizes run):

* all-gather of the attention input, [T / TP, 8192] bf16;
* reduce-scatter of the o_proj partial, [T, 8192] bf16;
* all-gather of the router logits, [T / TP, 256] fp32, and of the pooled rows,
  [T / TP, 4096] bf16;
* reduce-scatter of the expert output, [T, 4096] bf16;

through ``tensor_model_parallel_*`` (RCCL) and through
``ops/token_collectives.py`` with ``VLLM_ROCM_MUSE_AITER_TOKEN_COLLECTIVES=1``,
and checks that both give the same result.

Usage:
    VLLM_ROCM_USE_AITER=1 torchrun --nproc_per_node=4 \\
        benchmarks/kernels/benchmark_muse_spark_rocm_token_collectives.py
"""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.models.muse_spark.amd.ops import token_collectives
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _time(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384]
    )
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world, int(os.environ["RANK"]), local_rank=local_rank
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
    rank = get_tp_group().rank_in_group
    if rank == 0:
        print(f"TP{world}, us per call: RCCL vs AITER (chunked), eager")
    with patch.object(envs, "VLLM_ROCM_MUSE_AITER_TOKEN_COLLECTIVES", True):
        for t in args.tokens:
            t -= t % world
            cases = [
                (
                    "AG attn input",
                    "ag",
                    torch.randn(t // world, 8192, device=device).bfloat16(),
                ),
                ("RS o_proj", "rs", torch.randn(t, 8192, device=device).bfloat16()),
                ("AG router logits", "ag", torch.randn(t // world, 256, device=device)),
                (
                    "AG pooled",
                    "ag",
                    torch.randn(t // world, 4096, device=device).bfloat16(),
                ),
                ("RS experts", "rs", torch.randn(t, 4096, device=device).bfloat16()),
            ]
            for name, kind, x in cases:
                if kind == "ag":
                    rccl = lambda x=x: tensor_model_parallel_all_gather(x, 0)  # noqa: E731
                    aiter = lambda x=x: token_collectives.all_gather_tokens(x)  # noqa: E731
                else:
                    rccl = lambda x=x: tensor_model_parallel_reduce_scatter(x, 0)  # noqa: E731
                    aiter = lambda x=x: token_collectives.reduce_scatter_tokens(x)  # noqa: E731
                ref, got = rccl(), aiter()
                if kind == "ag":
                    same = torch.equal(ref, got)
                else:
                    same = torch.allclose(
                        ref.float(), got.float(), atol=2e-2, rtol=2e-2
                    )
                rccl_us, aiter_us = _time(rccl, args.iters), _time(aiter, args.iters)
                if rank == 0:
                    print(
                        f"T={t:>6} {name:<17} {rccl_us:>9.1f} {aiter_us:>9.1f} "
                        f"{rccl_us / aiter_us:>6.2f}x {'ok' if same else 'MISMATCH'}"
                    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
