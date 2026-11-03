# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm fused MoE finalize + TP all-reduce against the unfused paths."""

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

H = 4096
TOP_K = 8
EPS = 1e-6


def _reference(partials, weight, topk_weights, weight_first):
    """fp32 sum of the partials in rank order, then the post-expert finalize."""
    x = partials[0].float()
    for p in partials[1:]:
        x = x + p.float()
    gain = 1.0 if weight is None else weight.float() + 1.0
    if weight_first:
        x = x * gain
    y = x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)
    if not weight_first:
        y = y * gain
    return (y * topk_weights.float()[..., None]).sum(1)


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    """One BF16 unit in the last place at each element of ``t``, in fp64."""
    exponent = torch.frexp(t.double().abs().clamp_min(2.0**-126).float())[1]
    return torch.ldexp(torch.ones_like(t, dtype=torch.float64), exponent - 8)


def _exact_finalize(rows, weight, topk_weights):
    """The weight-first post-expert finalize of ``rows`` in fp64, plus a bound
    on the fp32 accumulation error of any kernel that computes it in fp32."""
    x = rows.double() * (weight.double() + 1.0)
    y = x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)
    terms = y * topk_weights.double()[..., None]
    return terms.sum(1), 2.0**-20 * terms.abs().sum(1)


@ray.remote(num_gpus=1, max_calls=1)
def _fused_finalize_all_reduce(monkeypatch, tp_size, pp_size, rank, port):
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.ops.fused_finalize_all_reduce import (
        MuseSparkFinalizeAllReduce,
    )
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        op = MuseSparkFinalizeAllReduce(255, TOP_K, H, group.cpu_group, pre_norm=True)
        weight = torch.randn(H, device=device) * 0.1
        weight = weight.bfloat16()
        dist.broadcast(weight, 0, group=group.device_group)

        def inputs(num_tokens, seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            partials = [
                torch.randn(
                    num_tokens, TOP_K, H, device=device, generator=gen
                ).bfloat16()
                for _ in range(tp_size)
            ]
            tw = torch.rand(num_tokens, TOP_K, device=device, generator=gen)
            return partials, tw

        def check(out, partials, tw, w, weight_first):
            ref = _reference(partials, w, tw, weight_first)
            torch.testing.assert_close(out.float(), ref, atol=1.5e-2, rtol=1e-2)
            # The unfused production path: fp32 all-reduce, then finalize.
            unfused = torch.empty_like(out)
            rows = partials[rank].float()
            dist.all_reduce(rows, group=group.device_group)
            fused_post_expert_finalize(rows, w, tw, EPS, 1.0, weight_first, unfused)
            torch.testing.assert_close(out, unfused, atol=1.5e-2, rtol=1e-2)
            # Every rank must hold the same bits.
            gathered = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(gathered, out, group=group.device_group)
            for other in gathered:
                assert torch.equal(other, out)

        # Shrinking and growing batches through the same mailbox, including
        # batches smaller than the world size.
        for step, n in enumerate((1, 3, 128, 7, 255, 64, 2, 255, 16)):
            for w, weight_first in ((weight, False), (weight, True), (None, False)):
                partials, tw = inputs(n, seed=1000 * step + n)
                out = torch.empty(n, H, dtype=torch.bfloat16, device=device)
                op(partials[rank], w, tw, EPS, 1.0, weight_first, out)
                check(out, partials, tw, w, weight_first)

        # A strided route view, as the experts hand it over.
        partials, tw = inputs(32, seed=7)
        wide = torch.zeros(32, TOP_K, H + 64, dtype=torch.bfloat16, device=device)
        wide[..., :H] = partials[rank]
        out = torch.empty(32, H, dtype=torch.bfloat16, device=device)
        op(wide[..., :H], weight, tw, EPS, 1.0, True, out)
        check(out, partials, tw, weight, True)

        # Graph replay: several calls per graph, replayed with new inputs.
        n = 64
        partials, tw = inputs(n, seed=11)
        static_in, static_tw = partials[rank].clone(), tw.clone()
        static_out = torch.empty(n, H, dtype=torch.bfloat16, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                op(static_in, weight, static_tw, EPS, 1.0, True, static_out)
        for seed in range(20):
            partials, tw = inputs(n, seed=100 + seed)
            static_in.copy_(partials[rank])
            static_tw.copy_(tw)
            graph.replay()
            torch.accelerator.synchronize()
            check(static_out, partials, tw, weight, True)
        op.close()


@ray.remote(num_gpus=1, max_calls=1)
def _fused_finalize_all_reduce_ep(monkeypatch, tp_size, pp_size, rank, port):
    """TP + EP: each rank holds only its experts' routes; the rest are NaN."""
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.ops.fused_finalize_all_reduce import (
        MuseSparkFinalizeAllReduce,
    )
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )

    num_experts = 64
    per_rank = num_experts // tp_size
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        op = MuseSparkFinalizeAllReduce(255, TOP_K, H, group.cpu_group, pre_norm=False)
        weight = (torch.ones(H, device=device) * 0.1).bfloat16()
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        expert_map[rank * per_rank : (rank + 1) * per_rank] = torch.arange(
            per_rank, dtype=torch.int32, device=device
        )

        for step, n in enumerate((1, 3, 128, 7, 255, 64, 2, 255, 16)):
            gen = torch.Generator(device=device).manual_seed(500 + step)
            full = torch.randn(n, TOP_K, H, device=device, generator=gen).bfloat16()
            tw = torch.rand(n, TOP_K, device=device, generator=gen)
            ids = torch.stack(
                [
                    torch.randperm(num_experts, device=device, generator=gen)[:TOP_K]
                    for _ in range(n)
                ]
            ).int()
            local = (ids // per_rank == rank)[..., None]
            for use_map in (True, False):
                # Without a map, non-local rows are zero (AITER's convention).
                fill = float("nan") if use_map else 0.0
                rows = torch.where(local, full, torch.full_like(full, fill))
                out = torch.empty(n, H, dtype=torch.bfloat16, device=device)
                op(
                    rows,
                    weight,
                    tw,
                    EPS,
                    1.0,
                    True,
                    out,
                    topk_ids=ids if use_map else None,
                    expert_map=expert_map if use_map else None,
                )
                exact, slack = _exact_finalize(full, weight, tw)
                # The partials are summed in fp32 and rounded to bf16 once.
                err = (out.double() - exact).abs()
                assert (err <= _bf16_ulp(exact) + slack).all(), err.max()
                # The unfused production path rounds each rank's partial to
                # bf16 and all-reduces in bf16: half an ulp per partial and per
                # add. A fixed tolerance against it fails with the TP size
                # (cancelling partials round at their own, larger magnitude).
                partial = torch.empty_like(out)
                fused_post_expert_finalize(
                    rows,
                    weight,
                    tw,
                    EPS,
                    1.0,
                    True,
                    partial,
                    topk_ids=ids if use_map else None,
                    expert_map=expert_map if use_map else None,
                )
                parts = [torch.empty_like(partial) for _ in range(tp_size)]
                dist.all_gather(parts, partial, group=group.device_group)
                unfused = partial.clone()
                dist.all_reduce(unfused, group=group.device_group)
                abs_sum = sum(p.double().abs() for p in parts)
                unfused_bound = (
                    0.5 * sum(_bf16_ulp(p) for p in parts)
                    + 0.5 * (tp_size - 1) * _bf16_ulp(abs_sum)
                    + slack
                )
                assert ((unfused.double() - exact).abs() <= unfused_bound).all()
                diff = (out.double() - unfused.double()).abs()
                assert (diff <= _bf16_ulp(exact) + slack + unfused_bound).all()
                gathered = [torch.empty_like(out) for _ in range(tp_size)]
                dist.all_gather(gathered, out, group=group.device_group)
                for other in gathered:
                    assert torch.equal(other, out)
        op.close()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_fused_finalize_all_reduce_ep_matches_unfused(monkeypatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _fused_finalize_all_reduce_ep)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_fused_finalize_all_reduce_matches_unfused(monkeypatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _fused_finalize_all_reduce)
