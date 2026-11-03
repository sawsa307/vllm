# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm fused MoE finalize + TP all-reduce against the unfused path."""

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
        op = MuseSparkFinalizeAllReduce(255, TOP_K, H, group.cpu_group)
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


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_fused_finalize_all_reduce_matches_unfused(monkeypatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _fused_finalize_all_reduce)
