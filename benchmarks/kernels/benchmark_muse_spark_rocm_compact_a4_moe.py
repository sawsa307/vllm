#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark MXFP4 experts at verify widths on one GPU: AITER vs compact A4.

Random Quark-layout MXFP4 experts (64 local = EP4, hidden 4096, intermediate
4096, per-32 E8M0 scales) and top-8 routes into the local range. Per row count:

* AITER ``fused_moe`` as the route-preserving runner calls it (one input row
  per route, unit weights, ``BLOCK_1X32``) on the weights converted as the
  layer converts them (``shuffle_weights`` + ``e8m0_shuffle``);
* ``compact_a4_experts`` on the weights as stored;
* ``compact_a4_experts(shuffled=True)`` on the converted copies, to see whether
  the layer's layout is the one its preshuffled path reads.

Outputs are the per-(token, slot) BF16 rows; the max relative difference of
the rows between AITER and compact A4 is reported (both quantize the
activation per row to MXFP4, so they should agree closely, not bitwise).

Usage (one GPU):
    python benchmarks/kernels/benchmark_muse_spark_rocm_compact_a4_moe.py
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import QuantMethod
from vllm.models.muse_spark.amd.ops.compact_a4_moe import compact_a4_experts
from vllm.utils.argparse_utils import FlexibleArgumentParser

HIDDEN, INTER, TOP_K = 4096, 4096, 8


def _time(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
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
    return start.elapsed_time(end) * 1000 / iters


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
    ).item()


def _paths(x, ids, stored, converted):
    """The three callables for one batch (bound explicitly, no loop closures)."""
    m = x.shape[0]
    w13, w2, s13, s2 = stored
    w13_c, w2_c, s13_c, s2_c = converted
    expanded = x[:, None, :].expand(m, TOP_K, HIDDEN).reshape(m * TOP_K, HIDDEN)
    expanded = expanded.contiguous()
    flat_ids = ids.view(m * TOP_K, 1)
    unit = torch.ones((m * TOP_K, 1), dtype=torch.float32, device=x.device)

    def aiter_ref():
        return rocm_aiter_ops.fused_moe(
            expanded,
            w13_c,
            w2_c,
            unit,
            flat_ids,
            activation_method=0,
            quant_method=QuantMethod.BLOCK_1X32.value,
            w1_scale=s13_c,
            w2_scale=s2_c,
        )

    def compact():
        return compact_a4_experts(x, ids, w13, w2, s13, s2, local_start=0)

    def compact_shuf():
        return compact_a4_experts(
            x, ids, w13_c, w2_c, s13_c, s2_c, local_start=0, shuffled=True
        )

    return aiter_ref, compact, compact_shuf


def main() -> None:
    parser = FlexibleArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--experts", type=int, default=64, help="local experts")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda")
    e = args.experts
    gen = torch.Generator(device=device).manual_seed(0)

    def u8(shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device=device, generator=gen
        )

    w13 = u8((e, 2 * INTER, HIDDEN // 2))
    w2 = u8((e, HIDDEN, INTER // 2))
    # E8M0 scales 2^-3 .. 2^0 keep the products in bf16 range
    s13 = u8((e, 2 * INTER, HIDDEN // 32), 124, 128)
    s2 = u8((e, HIDDEN, INTER // 32), 124, 128)

    from aiter.utility.fp4_utils import e8m0_shuffle

    s13_c = e8m0_shuffle(s13.view(e * 2 * INTER, -1)).view(e, 2 * INTER, -1)
    s2_c = e8m0_shuffle(s2.view(e * HIDDEN, -1)).view(e, HIDDEN, -1)
    w13_c, w2_c = rocm_aiter_ops.shuffle_weights(w13, w2)
    w13_c = w13_c.view(torch.uint8).contiguous()
    w2_c = w2_c.view(torch.uint8).contiguous()
    stored, converted = (w13, w2, s13, s2), (w13_c, w2_c, s13_c, s2_c)

    print(f"{e} local experts, top_k={TOP_K}, us per call")
    print(
        f"{'rows':>5} {'aiter':>8} {'compact':>8} {'compact-shuf':>12} "
        f"{'relerr':>8} {'relerr-shuf':>11}"
    )
    for m in args.rows:
        x = torch.randn(m, HIDDEN, device=device, generator=gen).bfloat16()
        ids = torch.stack(
            [torch.randperm(e, device=device, generator=gen)[:TOP_K] for _ in range(m)]
        ).int()
        aiter_ref, compact, compact_shuf = _paths(x, ids, stored, converted)
        ref = aiter_ref()
        rel = _rel(compact(), ref)
        try:
            rel_s = _rel(compact_shuf(), ref)
            t_s = _time(compact_shuf, args.iters)
        except Exception as exc:  # noqa: BLE001 - a layout mismatch is a result
            rel_s, t_s = float("nan"), float("nan")
            print(f"  shuffled path failed at {m} rows: {str(exc)[:80]}")
        t_a = _time(aiter_ref, args.iters)
        t_c = _time(compact, args.iters)
        print(
            f"{m:>5} {t_a:>8.1f} {t_c:>8.1f} {t_s:>12.1f} {rel:>8.2e} {rel_s:>11.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
