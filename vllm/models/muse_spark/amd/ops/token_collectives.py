# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-dimension all-gather and reduce-scatter for the token-sharded stream.

The token-sharded path moves [T, H] activations between ranks two to six times a
layer, at batch sizes past the CUDA-graph capture sizes (eager). vLLM's TP
collectives take RCCL there; with ``VLLM_ROCM_MUSE_AITER_TOKEN_COLLECTIVES=1``
these helpers use AITER's custom all-gather / reduce-scatter instead, in token
chunks that fit its IPC buffer (as vllm-project/vllm#63322 does for its column
all-gather), and fall back to RCCL for anything AITER does not take.
"""

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _aiter_comm():
    if not envs.VLLM_ROCM_MUSE_AITER_TOKEN_COLLECTIVES:
        return None
    if torch.cuda.is_current_stream_capturing():
        # Captured graphs take the registered paths of vLLM's communicator.
        return None
    comm = rocm_aiter_ops.get_aiter_allreduce()
    if comm is None or comm.disabled:
        return None
    return comm


def _chunk_rows(rows: int, fits) -> int | None:
    """``rows`` if one call takes them all, else the largest power-of-two row
    count that ``fits`` (None if not even one row does)."""
    if fits(rows):
        return rows
    chunk = 1 << (rows.bit_length() - 1)
    while chunk and not fits(chunk):
        chunk //= 2
    return chunk or None


def all_gather_tokens(x: torch.Tensor) -> torch.Tensor:
    """``tensor_model_parallel_all_gather(x, 0)`` for a 2-D token block."""
    comm = _aiter_comm()
    if comm is None or x.dim() != 2 or not x.is_contiguous() or x.shape[0] == 0:
        return tensor_model_parallel_all_gather(x, 0)
    world = get_tensor_model_parallel_world_size()
    rows, hidden = x.shape
    chunk = _chunk_rows(rows, lambda n: comm.should_custom_ag(x[:n]))
    if chunk is None:
        return tensor_model_parallel_all_gather(x, 0)
    if chunk == rows:
        out = comm.custom_all_gather(x, dim=0)
        return out if out is not None else tensor_model_parallel_all_gather(x, 0)
    out = x.new_empty(world * rows, hidden)
    blocks = out.view(world, rows, hidden)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        part = comm.custom_all_gather(x[start:end], dim=0)
        if part is None:
            return tensor_model_parallel_all_gather(x, 0)
        blocks[:, start:end].copy_(part.view(world, end - start, hidden))
    return out


def reduce_scatter_tokens(x: torch.Tensor) -> torch.Tensor:
    """``tensor_model_parallel_reduce_scatter(x, 0)`` for a 2-D token block whose
    row count is a multiple of the TP size."""
    comm = _aiter_comm()
    world = get_tensor_model_parallel_world_size()
    if (
        comm is None
        or x.dim() != 2
        or not x.is_contiguous()
        or x.shape[0] == 0
        or x.shape[0] % world
    ):
        return tensor_model_parallel_reduce_scatter(x, 0)
    rows, hidden = x.shape[0] // world, x.shape[1]
    chunk = _chunk_rows(rows, lambda n: comm.should_custom_rs(x[: world * n], 0))
    if chunk is None:
        return tensor_model_parallel_reduce_scatter(x, 0)
    out = x.new_empty(rows, hidden)
    if chunk == rows:
        if comm.custom_reduce_scatter(x, out, dim=0) is None:
            return tensor_model_parallel_reduce_scatter(x, 0)
        return out
    blocks = x.view(world, rows, hidden)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        part = blocks[:, start:end].reshape(world * (end - start), hidden)
        if comm.custom_reduce_scatter(part, out[start:end], dim=0) is None:
            return tensor_model_parallel_reduce_scatter(x, 0)
    return out
