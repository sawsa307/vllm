# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-gathers of column-sharded MoE projections, fused with the next norm.

At decode sizes every TP rank computes only its ``1 / W`` block of the pooler's
output columns. The full row is needed only by the norm that follows, so the
gather runs inside the norm's kernel: every rank pushes its column block into
every rank's buffer, then (after one grid barrier) normalizes the full rows.

Each call uses one of two buffers by parity of the call count. A rank can only
reach call ``n + 2``'s push after the barrier of call ``n + 1``, which every
peer reaches only after it has read call ``n``'s rows.
"""

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm
from vllm.triton_utils import tl, triton

from .fused_norm_expand import as_expanded, rmsnorm_expand_row
from .peer_sync import (
    NUM_BLOCKS,
    PeerBuffer,
    all_tp_ranks,
    grid_barrier,
    peer,
    tp_group_supports_peer_kernels,
)

logger = init_logger(__name__)

# The CUDA path shards the projections at the same decode sizes.
MAX_TOKENS = 256


@triton.jit
def push_columns(
    shard_ptr,
    stride_shard,
    P,
    rank,
    buf_off,
    T,
    W: tl.constexpr,
    D: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Phase A: write this rank's ``[T, D / W]`` block into every rank's
    ``[T, D]`` buffer at columns ``[rank * D / W, (rank + 1) * D / W)``.

    ``ROWS`` rows a step, so that every lane moves 16 bytes.
    """
    NS: tl.constexpr = D // W
    s = tl.arange(0, BLOCK_S)[None, :]
    rows = tl.arange(0, ROWS)[:, None]
    for t0 in range(tl.program_id(0) * ROWS, T, NB * ROWS):
        t = t0 + rows
        mask = (t < T) & (s < NS)
        x = tl.load(shard_ptr + t * stride_shard + s, mask=mask)
        for r in tl.static_range(W):
            dst = peer(P, r, tl.bfloat16) + buf_off + rank * NS
            tl.store(dst + t * D + s, x, mask=mask)


def push_rows(block_s: int, num_waves: int) -> int:
    """Rows of a :func:`push_columns` step: 8 bf16 (16 bytes) a lane."""
    return max(1, num_waves * 64 * 8 // block_s)


@triton.jit
def parity_buffer(P, rank, target_off, buf_elems, LOG2_ARRIVALS: tl.constexpr):
    """This call's barrier target and the offset of its parity buffer."""
    b = tl.program_id(0)
    target = tl.load(peer(P, rank, tl.int32) + target_off + b) + (1 << LOG2_ARRIVALS)
    # Bit LOG2_ARRIVALS of the target flips every call, wrapped or not.
    return target, ((target >> LOG2_ARRIVALS) & 1) * buf_elems


@triton.jit
def _gather_rmsnorm_kernel(
    shard_ptr,
    stride_shard,
    w_ptr,
    out_ptr,
    P,
    rank,
    T,
    eps,
    gain_center,
    buf_elems,
    count_off,
    target_off,
    W: tl.constexpr,
    D: tl.constexpr,
    NB: tl.constexpr,
    LOG2_ARRIVALS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    REPEATS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    NUM_WAVES: tl.constexpr,
):
    target, buf_off = parity_buffer(P, rank, target_off, buf_elems, LOG2_ARRIVALS)
    push_columns(shard_ptr, stride_shard, P, rank, buf_off, T, W, D, NB, BLOCK_S, ROWS)
    grid_barrier(P, rank, count_off, target, W, NUM_WAVES)

    d = tl.arange(0, BLOCK_D)
    dm = d < D
    mine = peer(P, rank, tl.bfloat16) + buf_off
    for t in range(tl.program_id(0), T, NB):
        x = tl.load(mine + t * D + d, mask=dm, other=0.0).to(tl.float32)
        rmsnorm_expand_row(
            x,
            w_ptr,
            out_ptr,
            t,
            d,
            dm,
            D,
            eps,
            gain_center,
            REPEATS,
            HAS_WEIGHT,
            INPUT_SCALE,
        )
    tl.store(peer(P, rank, tl.int32) + target_off + tl.program_id(0), target)


class MuseSparkColumnGather:
    """Peer-mapped workspace shared by every sharded projection gather.

    Each rank's buffer holds two ``[max_tokens, max_hidden]`` bf16 row buffers,
    then one arrival counter and one barrier target per block (int32).
    """

    def __init__(
        self, max_tokens: int, max_hidden: int, group: dist.ProcessGroup
    ) -> None:
        self.max_tokens = max_tokens
        self.max_hidden = max_hidden
        self.world_size = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)
        self.buf_elems = max_tokens * max_hidden
        sync_off = 2 * self.buf_elems * 2
        self.count_off = sync_off // 4
        self.target_off = self.count_off + 1
        self.log2_arrivals = (self.world_size * NUM_BLOCKS).bit_length() - 1
        assert 1 << self.log2_arrivals == self.world_size * NUM_BLOCKS
        self.buffer = PeerBuffer(sync_off + (1 + NUM_BLOCKS) * 4, group)

    def supports(self, num_tokens: int, hidden: int, dtype: torch.dtype) -> bool:
        """Whether ``[num_tokens, hidden / W]`` blocks can be gathered."""
        return (
            0 < num_tokens <= self.max_tokens
            and hidden <= self.max_hidden
            and hidden % (8 * self.world_size) == 0
            and dtype == torch.bfloat16
        )

    def gather_rmsnorm(
        self, shard: torch.Tensor, norm: MuseSpark1xRMSNorm, repeats: int
    ) -> torch.Tensor:
        """``norm(all_gather(shard, dim=-1))``; each row stored ``repeats``
        times as :func:`fused_rmsnorm_expand` does. Collective over TP."""
        num_tokens = shard.shape[0]
        hidden = shard.shape[1] * self.world_size
        assert self.supports(num_tokens, hidden, shard.dtype)
        assert shard.stride(-1) == 1
        assert norm.variance_size_override is None and norm.hidden_size == hidden
        out = torch.empty(
            (num_tokens, repeats, hidden), dtype=shard.dtype, device=shard.device
        )
        weight = norm.weight if norm.has_weight else None
        block_s = triton.next_power_of_2(hidden // self.world_size)
        _gather_rmsnorm_kernel[(NUM_BLOCKS,)](
            shard,
            shard.stride(0),
            weight if weight is not None else shard,
            out,
            self.buffer.peers,
            self.rank,
            num_tokens,
            norm.variance_epsilon,
            norm.fused_gain_center,
            self.buf_elems,
            self.count_off,
            self.target_off,
            W=self.world_size,
            D=hidden,
            NB=NUM_BLOCKS,
            LOG2_ARRIVALS=self.log2_arrivals,
            BLOCK_S=block_s,
            ROWS=push_rows(block_s, 8),
            BLOCK_D=triton.next_power_of_2(hidden),
            REPEATS=repeats,
            HAS_WEIGHT=weight is not None,
            INPUT_SCALE=norm.applies_weight_before_norm and weight is not None,
            NUM_WAVES=8,
            num_warps=8,
        )
        return as_expanded(out) if repeats > 1 else out[:, 0]

    def close(self) -> None:
        self.buffer.close()


_WORKSPACE: MuseSparkColumnGather | None = None


def maybe_create_column_gather(
    vllm_config: VllmConfig, max_hidden: int
) -> MuseSparkColumnGather | None:
    """Collectively create the shared workspace once; every TP rank gets one or
    none. Call from every TP rank in the same order (model construction)."""
    global _WORKSPACE
    if _WORKSPACE is not None:
        return _WORKSPACE
    import vllm.envs as envs

    pc = vllm_config.parallel_config
    # Sharding the output columns needs the same tokens, in the same order, on
    # every TP rank.
    if (
        not envs.VLLM_ROCM_MUSE_SHARD_PROJECTIONS
        or pc.use_sequence_parallel_moe
        or pc.data_parallel_size != 1
        or pc.use_ubatching
        or envs.VLLM_BATCH_INVARIANT
        or not tp_group_supports_peer_kernels()
    ):
        return None
    tp_group = get_tp_group()
    try:
        workspace = MuseSparkColumnGather(
            min(MAX_TOKENS, vllm_config.scheduler_config.max_num_batched_tokens),
            max_hidden,
            tp_group.cpu_group,
        )
    except RuntimeError as exc:
        logger.warning("Muse Spark sharded projection gather failed: %s", exc)
        workspace = None
    if not all_tp_ranks(workspace is not None):
        if workspace is not None:
            workspace.close()
        return None
    assert workspace is not None
    _WORKSPACE = workspace
    logger.info_once(
        "Muse Spark MoE projections column-sharded over TP%d up to %d tokens, "
        "gathered inside the next norm.",
        tp_group.world_size,
        workspace.max_tokens,
    )
    return workspace
