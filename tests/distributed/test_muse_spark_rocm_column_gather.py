# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm projection collectives fused into norms, against the unfused path:
the column-sharded MoE projection gathers and the attention all-reduce."""

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


@ray.remote(num_gpus=1, max_calls=1)
def _all_reduce_boundary(monkeypatch, tp_size, pp_size, rank, port):
    """Partial o_proj rows reduced inside the attention residual boundary."""
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
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

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        op = MuseSparkAllReduceBoundary(256, HIDDEN, group.cpu_group)
        # The target decoder's post-attention norm (no weight), gate and
        # pre-FFN norm.
        post = MuseSpark1xRMSNorm(
            HIDDEN, eps=1e-5, has_weight=False, is_post_norm=True, residual_id=3
        ).to(device)
        pre = MuseSpark1xRMSNorm(
            HIDDEN,
            eps=1e-5,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=4,
            output_norm_gain_center_type="zero",
        ).to(device)
        connection = MuseSpark1xNormPreservingResidualConnection(HIDDEN).to(device)
        with torch.no_grad():
            pre.weight.copy_(torch.linspace(0.2, -0.2, HIDDEN))
            connection.gate.copy_(torch.linspace(-2.0, 2.0, HIDDEN))
        connection.materialize_coefficients()

        def boundary(branch, stream):
            return fused_norm_residual_norm_fwd(
                branch,
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

        def inputs(n, seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            partials = [
                torch.randn(n, HIDDEN, device=device, generator=gen).bfloat16()
                for _ in range(tp_size)
            ]
            stream = torch.randn(n, HIDDEN, device=device, generator=gen)
            return partials, stream

        def check(out, new_stream, partials, stream):
            # fp32 sum in rank order, rounded once: bit-identical boundary.
            total = partials[0].float()
            for p in partials[1:]:
                total = total + p.float()
            ref_out, ref_stream = boundary(total.bfloat16(), stream)
            assert torch.equal(out, ref_out)
            assert torch.equal(new_stream, ref_stream)
            # The production all-reduce sums in its own order: close only.
            ar = tensor_model_parallel_all_reduce(partials[rank].clone())
            ar_out, ar_stream = boundary(ar, stream)
            torch.testing.assert_close(new_stream, ar_stream, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(
                out.float(), ar_out.float(), atol=5e-2, rtol=5e-2
            )
            # The replicated stream must stay identical on every rank.
            gathered = [torch.empty_like(new_stream) for _ in range(tp_size)]
            dist.all_gather(gathered, new_stream, group=group.device_group)
            for other in gathered:
                assert torch.equal(other, new_stream)

        for step, n in enumerate((1, 3, 256, 7, 64, 2, 255, 16, 129)):
            partials, stream = inputs(n, 300 + step)
            assert op.supports(partials[rank].dtype, stream, post, connection, pre)
            router = torch.empty_like(stream) if step % 2 else None
            out, new_stream = op(
                partials[rank],
                stream,
                post,
                connection,
                pre,
                torch.bfloat16,
                router_out=router,
            )
            check(out, new_stream, partials, stream)
            if router is not None:
                # The unrounded FP32 pre-FFN-norm output, for the router.
                assert torch.equal(router.bfloat16(), out)

        # Graph replay, several calls per graph.
        n = 32
        static_partial = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
        static_stream = torch.zeros(n, HIDDEN, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = [
                op(static_partial, static_stream, post, connection, pre, torch.bfloat16)
                for _ in range(3)
            ]
        for seed in range(10):
            partials, stream = inputs(n, 3000 + seed)
            static_partial.copy_(partials[rank])
            static_stream.copy_(stream)
            graph.replay()
            torch.accelerator.synchronize()
            for out, new_stream in results:
                check(out, new_stream, partials, stream)
        op.close()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel")
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_all_reduce_boundary_matches_unfused(monkeypatch, tp_size):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _all_reduce_boundary)


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
