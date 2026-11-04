#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost of one cross-rank grid barrier of the Muse Spark ROCm peer kernels.

The fused finalize, pooler/unpooler gathers and o_proj all-reduce boundary each
synchronize the TP ranks with ``peer_sync.grid_barrier`` (one counter per rank,
every block of every rank arrives on every counter; a system-scope release per
wave before, an acquire poll after). On MI355X TP4 the gather kernels cost
15-25 us more than the norm they replace while moving a few KB, so the
synchronization is the suspect. This times, as graph replays of ``--calls``
back-to-back launches, a kernel that does nothing but the barrier:

* ``grid_barrier``: the barrier as the kernels now use it (per-block flags,
  vector polls, one workgroup-scope release per wave);
* ``counter``: the previous barrier (one counter per rank, every block of every
  rank arrives on it, a system-scope release per wave);
* ``counter-nowave``: the same without the per-wave release (one release per
  block, by the arrival atomic) - the L2 writeback cost;
* ``flags``: one flag per (writer rank, block) on every rank, each block waits
  on its own W flags (AITER's custom all-reduce pattern), no per-wave release -
  the contention cost;
* ``flags-wave-sys`` / ``flags-wave-gpu`` / ``flags-wave-cta``: ``flags`` plus a
  per-wave release on a distinct word per (block, wave) in the uncached peer
  buffer, at system / device / workgroup scope;
* ``flags-wave-cached``: the per-wave release at workgroup scope on a word of
  ordinary (cached) device memory instead - a plain ``s_waitcnt vmcnt(0)`` plus
  one L2 atomic, the cheapest form that still orders every wave's stores
  before the flag write;

over a sweep of block counts and waves per block, against an empty launch of
the same grid and a custom all-reduce of a [1, 8192] bf16 tensor.

Usage:
    torchrun --nproc_per_node=4 \\
        benchmarks/kernels/benchmark_muse_spark_rocm_peer_barrier.py
"""

import itertools
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
from vllm.models.muse_spark.amd.ops.peer_sync import (
    PeerBuffer,
    grid_barrier,
    peer,
)
from vllm.triton_utils import tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser


@triton.jit
def _barrier_counter(
    P, rank, count_off, target, W: tl.constexpr, NUM_WAVES: tl.constexpr
):
    mine = peer(P, rank, tl.int32) + count_off
    lanes = tl.arange(0, NUM_WAVES * 64)
    tl.atomic_add(mine + lanes * 0, 0, mask=lanes % 64 == 0, sem="release", scope="sys")
    tl.debug_barrier()
    for r in tl.static_range(W):
        tl.atomic_add(peer(P, r, tl.int32) + count_off, 1, sem="release", scope="sys")
    seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    while seen - target < 0:
        seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    tl.debug_barrier()


@triton.jit
def _barrier_no_wave_release(
    P, rank, count_off, target, W: tl.constexpr, NUM_WAVES: tl.constexpr
):
    mine = peer(P, rank, tl.int32) + count_off
    tl.debug_barrier()
    for r in tl.static_range(W):
        tl.atomic_add(peer(P, r, tl.int32) + count_off, 1, sem="release", scope="sys")
    seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    while seen - target < 0:
        seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
    tl.debug_barrier()


@triton.jit
def _wave_release_sys(ptr, NUM_WAVES: tl.constexpr):
    lanes = tl.arange(0, NUM_WAVES * 64)
    tl.atomic_add(
        ptr + lanes // 64, 0, mask=lanes % 64 == 0, sem="release", scope="sys"
    )


@triton.jit
def _wave_release_gpu(ptr, NUM_WAVES: tl.constexpr):
    lanes = tl.arange(0, NUM_WAVES * 64)
    tl.atomic_add(
        ptr + lanes // 64, 0, mask=lanes % 64 == 0, sem="release", scope="gpu"
    )


@triton.jit
def _wave_release_cta(ptr, NUM_WAVES: tl.constexpr):
    lanes = tl.arange(0, NUM_WAVES * 64)
    tl.atomic_add(
        ptr + lanes // 64, 0, mask=lanes % 64 == 0, sem="release", scope="cta"
    )


@triton.jit
def _barrier_flags(
    P,
    rank,
    flags_off,
    wave_off,
    scratch_ptr,
    target,
    W: tl.constexpr,
    NB: tl.constexpr,
    NUM_WAVES: tl.constexpr,
    WAVE_RELEASE: tl.constexpr,
):
    """Flag ``[writer][block]`` on every rank; block ``b`` waits on its W flags."""
    b = tl.program_id(0)
    if WAVE_RELEASE == 1:
        _wave_release_sys(peer(P, rank, tl.int32) + wave_off + b * NUM_WAVES, NUM_WAVES)
    elif WAVE_RELEASE == 2:
        _wave_release_gpu(peer(P, rank, tl.int32) + wave_off + b * NUM_WAVES, NUM_WAVES)
    elif WAVE_RELEASE == 3:
        _wave_release_cta(peer(P, rank, tl.int32) + wave_off + b * NUM_WAVES, NUM_WAVES)
    elif WAVE_RELEASE == 4:
        _wave_release_cta(scratch_ptr + b * NUM_WAVES, NUM_WAVES)
    tl.debug_barrier()
    for r in tl.static_range(W):
        tl.atomic_xchg(
            peer(P, r, tl.int32) + flags_off + rank * NB + b,
            target,
            sem="release",
            scope="sys",
        )
    mine = peer(P, rank, tl.int32) + flags_off + b
    for s in tl.static_range(W):
        seen = tl.atomic_add(mine + s * NB, 0, sem="acquire", scope="sys")
        while seen - target < 0:
            seen = tl.atomic_add(mine + s * NB, 0, sem="acquire", scope="sys")
    tl.debug_barrier()


@triton.jit
def _barrier_kernel(
    P,
    rank,
    count_off,
    target_off,
    flags_off,
    wave_off,
    scratch_ptr,
    W: tl.constexpr,
    NB: tl.constexpr,
    NUM_WAVES: tl.constexpr,
    MODE: tl.constexpr,
):
    b = tl.program_id(0)
    target = tl.load(peer(P, rank, tl.int32) + target_off + b) + W * NB
    if MODE == 0:
        _barrier_counter(P, rank, count_off, target, W, NUM_WAVES)
    elif MODE == 7:
        grid_barrier(P, rank, flags_off, wave_off, target, W, NB, NUM_WAVES)
    elif MODE == 1:
        _barrier_no_wave_release(P, rank, count_off, target, W, NUM_WAVES)
    else:
        _barrier_flags(
            P,
            rank,
            flags_off,
            wave_off,
            scratch_ptr,
            target,
            W,
            NB,
            NUM_WAVES,
            MODE - 2,
        )
    tl.store(peer(P, rank, tl.int32) + target_off + b, target)


@triton.jit
def _empty_kernel(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 0)


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


MODES = {
    "counter": 0,
    "counter-nowave": 1,
    "flags": 2,
    "flags-wave-sys": 3,
    "flags-wave-gpu": 4,
    "flags-wave-cta": 5,
    "flags-wave-cached": 6,
    "grid_barrier": 7,
}


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", default=[64, 16, 4, 1])
    parser.add_argument("--waves", type=int, nargs="+", default=[16, 8, 1])
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

    configs = list(itertools.product(MODES, args.blocks, args.waves))
    max_blocks = max(args.blocks)
    # Per config: a counter, NB targets, 8 * NB flags and NB * waves release
    # words (int32), disjoint so the configs never see each other's arrivals.
    max_waves = max(args.waves)
    stride = 1 + max_blocks + 8 * max_blocks + max_blocks * max_waves
    buffer = PeerBuffer(len(configs) * stride * 4, tp.cpu_group)
    P = buffer.peers
    scratch = torch.zeros(max_blocks * max_waves, dtype=torch.int32, device=device)
    x = torch.randn(1, 8192, device=device).bfloat16()

    def barrier(index, mode, blocks, waves):
        base = index * stride

        def fn(i):
            _barrier_kernel[(blocks,)](
                P,
                rank,
                base,
                base + 1,
                base + 1 + max_blocks,
                base + 1 + max_blocks + 8 * max_blocks,
                scratch,
                W=world_size,
                NB=blocks,
                NUM_WAVES=waves,
                MODE=MODES[mode],
                num_warps=waves,
            )

        return fn

    if rank == 0:
        print(f"TP{world_size}, us per call")
    for blocks in args.blocks:
        us = _time_graph(
            lambda i, b=blocks: _empty_kernel[(b,)](scratch, num_warps=1),
            args.calls,
            args.iters,
        )
        if rank == 0:
            print(f"empty launch, {blocks:>3} blocks: {us:6.1f}")
    us = _time_graph(
        lambda i: tensor_model_parallel_all_reduce(x), args.calls, args.iters
    )
    if rank == 0:
        print(f"custom all-reduce [1, 8192] bf16: {us:6.1f}")
        print(f"\n{'barrier':<16} {'blocks':>6} {'waves':>5} {'us':>7}")
    for index, (mode, blocks, waves) in enumerate(configs):
        us = _time_graph(barrier(index, mode, blocks, waves), args.calls, args.iters)
        torch.accelerator.synchronize()
        dist.barrier(group=tp.cpu_group)
        if rank == 0:
            print(f"{mode:<16} {blocks:>6} {waves:>5} {us:7.1f}")
    buffer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
