#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune the split-K skinny GEMM for the Muse Spark column-sharded projections.

For each shard shape (the pooler's ``[4096 / TP, 8192]`` and the unpooler's
``[8192 / TP, 4096]`` weight blocks) and decode token count, times every
candidate config and ``F.linear`` as graph replays over rotating weight copies
(so the weight streams from HBM, as in a forward), drops candidates that are
not within one BF16 ulp of an fp64 reference, and prints ``_CONFIGS`` entries
for ``vllm/models/muse_spark/amd/ops/skinny_gemm.py``. The lookup takes the
first entry whose max-token bound covers the batch, so a shape gets entries
only up to the last token count of a contiguous run where the skinny GEMM wins.

Usage (one GPU):
    python benchmarks/kernels/tune_muse_spark_rocm_skinny_gemm.py --tp-sizes 4 \\
        [--allowlist spill_free.json] [--min-grid 256]
"""

import itertools
import json

import torch
import torch.nn.functional as F

from vllm.models.muse_spark.amd.ops.skinny_gemm import skinny_gemm
from vllm.utils.argparse_utils import FlexibleArgumentParser

POOLED, HIDDEN = 4096, 8192


def _time(fn, copies: int, iters: int) -> float:
    """Microseconds per call, from a graph cycling through the weight copies."""
    for i in range(copies):
        fn(i)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(copies):
            fn(i)
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


def _accurate(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> bool:
    ref = x.double() @ w.double().t()
    ulp = torch.ldexp(torch.ones_like(ref), torch.frexp(ref.float())[1] - 8)
    bound = ulp + 2**-20 * (x.double().abs() @ w.double().abs().t())
    return bool(((out.double() - ref).abs() <= bound).all())


def _candidates(n: int, k: int, m: int, min_grid: int, allowed: set | None):
    """Configs to time. ``min_grid`` drops grids with fewer workgroups (every
    tuned TP8 entry launches at least 256, one per MI355X CU); ``allowed``, if
    given, keeps only those configs (e.g. the spill-free ones of an AOT pass)."""
    max_block_m = max(16, 1 << (m - 1).bit_length())
    for cfg in itertools.product(
        (8, 16, 32, 64, 128),
        (16, 32, 64, 128),
        (16, 32, 64, 128, 256, 512),
        (1, 2, 4, 8, 16, 32),
        (2, 4, 8),
    ):
        block_m, block_n, block_k, splits, _ = cfg
        if block_m > max_block_m or n % block_n or k % (splits * block_k):
            continue
        if (n // block_n) * splits * -(-m // block_m) < min_grid:
            continue
        if allowed is not None and cfg not in allowed:
            continue
        yield cfg


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 64, 96, 128, 256]
    )
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--min-grid",
        type=int,
        default=256,
        help="skip configs launching fewer workgroups (0: keep all)",
    )
    parser.add_argument(
        "--allowlist",
        help='JSON {"N,K": [[block_m, block_n, block_k, splits, warps], ...]}: '
        "time only these configs per shape",
    )
    args = parser.parse_args()
    allowlist = None
    if args.allowlist:
        with open(args.allowlist) as f:
            allowlist = {
                tuple(map(int, shape.split(","))): {tuple(c) for c in cfgs}
                for shape, cfgs in json.load(f).items()
            }

    torch.manual_seed(0)
    table: dict[tuple[int, int], list[tuple[int, tuple[int, ...]]]] = {}
    for tp in args.tp_sizes:
        for name, n, k in (
            ("pooler", POOLED // tp, HIDDEN),
            ("unpooler", HIDDEN // tp, POOLED),
        ):
            weights = [
                (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
                for _ in range(args.copies)
            ]
            rows = table.setdefault((n, k), [])
            # Entries must cover a contiguous run from the smallest batch.
            contiguous = True
            for m in sorted(args.tokens):
                x = torch.randn(m, k, device="cuda").bfloat16()
                linear_us = _time(
                    lambda i, x=x, ws=weights: F.linear(x, ws[i]),
                    args.copies,
                    args.iters,
                )
                best = None
                allowed = None if allowlist is None else allowlist.get((n, k), set())
                for cfg in _candidates(n, k, m, args.min_grid, allowed):
                    try:
                        if not _accurate(
                            skinny_gemm(x, weights[0], cfg), x, weights[0]
                        ):
                            continue
                        us = _time(
                            lambda i, cfg=cfg, x=x, ws=weights: skinny_gemm(
                                x, ws[i], cfg
                            ),
                            args.copies,
                            args.iters,
                        )
                    except Exception:  # noqa: BLE001 - invalid launch configs
                        continue
                    if best is None or us < best[0]:
                        best = (us, cfg)
                wins = best is not None and best[0] < linear_us
                head = f"TP{tp} {name} [{n}x{k}] M={m}: F.linear {linear_us:.1f} us"
                if best is not None:
                    head += f", best skinny {best[0]:.1f} us {best[1]}"
                print(head, "-> skinny" if wins else "-> F.linear")
                contiguous = contiguous and wins
                if contiguous:
                    assert best is not None
                    rows.append((m, best[1]))
    print("\n# Paste into _CONFIGS in vllm/models/muse_spark/amd/ops/skinny_gemm.py")
    for (n, k), rows in table.items():
        if not rows:
            continue
        print(f"    ({n}, {k}): (")
        for m, cfg in rows:
            print(f"        ({m}, {', '.join(map(str, cfg))}),")
        print("    ),")


if __name__ == "__main__":
    main()
