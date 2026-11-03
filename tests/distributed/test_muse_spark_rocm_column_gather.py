# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm column-sharded projection gathers against the unfused path."""

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

POOLED = 4096
HIDDEN = 8192
TOP_K = 8


@ray.remote(num_gpus=1, max_calls=1)
def _gather_rmsnorm(monkeypatch, tp_size, pp_size, rank, port):
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.ops.column_gather import MuseSparkColumnGather
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
        fused_rmsnorm_expand,
        take_expanded,
    )
    from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        gather = MuseSparkColumnGather(256, 8192, group.cpu_group)
        norm = MuseSpark1xRMSNorm(
            hidden_size=POOLED,
            eps=1e-6,
            has_weight=True,
            is_post_norm=False,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2,
            output_norm_gain_center_type="zero",
        ).to(device)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(-0.5, 0.5, POOLED))
        block = POOLED // tp_size

        def check(n, seed, repeats):
            gen = torch.Generator(device=device).manual_seed(seed)
            full = torch.randn(n, POOLED, device=device, generator=gen).bfloat16()
            shard = full[:, rank * block : (rank + 1) * block]
            out = gather.gather_rmsnorm(shard, norm, repeats)
            ref = fused_rmsnorm_expand(
                full,
                norm.weight,
                norm.variance_epsilon,
                norm.fused_gain_center,
                False,
                repeats,
            )
            # Same row math as fused_rmsnorm_expand: bit-identical.
            assert torch.equal(out, ref)
            if repeats > 1:
                assert torch.equal(
                    take_expanded(out, repeats), take_expanded(ref, repeats)
                )
            gathered = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(gathered, out.contiguous(), group=group.device_group)
            for other in gathered:
                assert torch.equal(other, out.contiguous())

        # Shrinking and growing batches through both parity buffers, from
        # strided column views of the full rows.
        for step, n in enumerate((1, 3, 256, 7, 64, 2, 255, 16, 129)):
            check(n, 100 + step, TOP_K if step % 2 else 1)

        # Graph replay, several gathers per graph.
        n = 64
        static_full = torch.zeros(n, POOLED, dtype=torch.bfloat16, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [
                gather.gather_rmsnorm(
                    static_full[:, rank * block : (rank + 1) * block], norm, TOP_K
                )
                for _ in range(3)
            ]
        for seed in range(10):
            gen = torch.Generator(device=device).manual_seed(1000 + seed)
            static_full.copy_(torch.randn(n, POOLED, device=device, generator=gen))
            graph.replay()
            torch.accelerator.synchronize()
            ref = fused_rmsnorm_expand(
                static_full,
                norm.weight,
                norm.variance_epsilon,
                norm.fused_gain_center,
                False,
                TOP_K,
            )
            for out in outs:
                assert torch.equal(out, ref)
        gather.close()


@ray.remote(num_gpus=1, max_calls=1)
def _gather_boundary(monkeypatch, tp_size, pp_size, rank, port):
    """Unpooler column blocks gathered into the FFN residual boundary."""
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.ops.column_gather import MuseSparkColumnGather
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        gather = MuseSparkColumnGather(256, HIDDEN, group.cpu_group)
        # The target decoder's post-FFN norm, gate and next pre-QKV norm.
        post = MuseSpark1xRMSNormWithInputScale(
            HIDDEN,
            eps=1e-5,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=4,
            output_norm_gain_center_type="zero",
        ).to(device)
        pre = MuseSpark1xRMSNorm(
            HIDDEN,
            eps=1e-5,
            has_weight=True,
            is_post_norm=False,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=5,
        ).to(device)
        connection = MuseSpark1xNormPreservingResidualConnection(HIDDEN).to(device)
        with torch.no_grad():
            post.weight.copy_(torch.linspace(-0.3, 0.3, HIDDEN))
            pre.weight.copy_(torch.linspace(0.2, -0.2, HIDDEN))
            connection.gate.copy_(torch.linspace(-2.0, 2.0, HIDDEN))
        connection.materialize_coefficients()
        block = HIDDEN // tp_size

        def unfused(branch, stream):
            return fused_norm_residual_norm_fwd(
                branch,
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
                pre_input_scale=False,
                out_dtype=torch.bfloat16,
            )

        def check(n, seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            branch = torch.randn(n, HIDDEN, device=device, generator=gen).bfloat16()
            stream = torch.randn(n, HIDDEN, device=device, generator=gen)
            shard = branch[:, rank * block : (rank + 1) * block]
            assert gather.supports_boundary(shard, stream, post, connection, pre)
            out, new_stream = gather.gather_boundary(
                shard, stream, post, connection, pre, torch.bfloat16
            )
            ref_out, ref_stream = unfused(branch, stream)
            # The boundary math is shared with the unfused kernel.
            assert torch.equal(out, ref_out)
            assert torch.equal(new_stream, ref_stream)

        # Alternate with pooler gathers: both share the parity buffers.
        pooled_block = POOLED // tp_size
        norm = MuseSpark1xRMSNorm(POOLED, eps=1e-6, has_weight=False).to(device)
        for step, n in enumerate((1, 3, 256, 7, 64, 2, 255, 16, 129)):
            check(n, 200 + step)
            pooled = torch.randn(n, POOLED, device=device).bfloat16()
            start = rank * pooled_block
            gather.gather_rmsnorm(pooled[:, start : start + pooled_block], norm, 1)

        # Graph replay.
        n = 32
        static_branch = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
        static_stream = torch.zeros(n, HIDDEN, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [
                gather.gather_boundary(
                    static_branch[:, rank * block : (rank + 1) * block],
                    static_stream,
                    post,
                    connection,
                    pre,
                    torch.bfloat16,
                )
                for _ in range(3)
            ]
        for seed in range(10):
            gen = torch.Generator(device=device).manual_seed(2000 + seed)
            static_branch.copy_(torch.randn(n, HIDDEN, device=device, generator=gen))
            static_stream.copy_(torch.randn(n, HIDDEN, device=device, generator=gen))
            graph.replay()
            torch.accelerator.synchronize()
            ref_out, ref_stream = unfused(static_branch, static_stream)
            for out, new_stream in outs:
                assert torch.equal(out, ref_out)
                assert torch.equal(new_stream, ref_stream)
        gather.close()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_gather_boundary_matches_unfused(monkeypatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _gather_boundary)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_gather_rmsnorm_matches_unfused(monkeypatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _gather_rmsnorm)
