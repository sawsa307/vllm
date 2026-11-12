# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of the KV cache read checks (see ``vllm/v1/core/kv_read_check.py``).

A block's checksum is ``sum_l m_l * sum_j x_j * (2 * j + 1)  mod 2**64`` over
the checked layers ``l`` of its group on every TP rank, where ``x_j`` are the
8-byte (or 4-byte, if the page is not a multiple of 8 bytes) pieces of the
layer's page of the block and ``m_l`` is an odd factor distinct per layer and
TP rank. Odd weights make any change to one piece change the checksum.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.kv_read_check import KVReadCheckPlan, kv_read_check_group_ids
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.utils import layer_spec_for_tensor

# Bytes of KV gathered per checksum pass of one layer.
_CHUNK_BYTES = 64 << 20


@dataclass
class _Layer:
    # ``[num_blocks * blocks_per_block, ...]``: a manager block is
    # ``blocks_per_block`` consecutive kernel blocks.
    cache: torch.Tensor
    blocks_per_block: int
    factor: int


class KVReadChecksummer:
    """Computes the checksums of a ``KVReadCheckPlan``'s blocks."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, torch.Tensor],
        draft_layer_names: frozenset[str],
        tp_rank: int,
        tp_group: dist.ProcessGroup | None,
    ):
        self._tp_group = tp_group
        checked = set(kv_read_check_group_ids(kv_cache_config))
        self._layers: dict[int, list[_Layer]] = {group_id: [] for group_id in checked}
        names: list[tuple[int, str, int]] = []
        for tensor in kv_cache_config.kv_cache_tensors:
            group_id, spec = layer_spec_for_tensor(kv_cache_config, tensor)
            if group_id not in checked or not spec.has_layer_views:
                continue
            num_blocks = kv_cache_config.num_blocks_of(tensor)
            names.extend(
                (group_id, name, num_blocks)
                for name in tensor.layers
                if name not in draft_layer_names
            )
        for index, (group_id, name, num_blocks) in enumerate(sorted(names)):
            cache = kv_caches[name]
            blocks_per_block, rest = divmod(cache.shape[0], num_blocks)
            assert rest == 0, f"{name}: {cache.shape[0]} blocks, {num_blocks} expected"
            factor = 2 * (tp_rank * len(names) + index) + 1
            self._layers[group_id].append(_Layer(cache, blocks_per_block, factor))
        self._weights: dict[tuple[int, torch.device], torch.Tensor] = {}

    def compute(self, plan: KVReadCheckPlan) -> torch.Tensor:
        """Queue the plan's checksums on the current stream: one int64 (read
        as uint64) per row, summed over TP ranks."""
        sizes = [len(block_ids) for block_ids in plan.block_ids.values()]
        layers = next((ls for ls in self._layers.values() if ls), None)
        device = layers[0].cache.device if layers else torch.device("cpu")
        out = torch.zeros(sum(sizes), dtype=torch.int64, device=device)
        row = 0
        for (group_id, block_ids), size in zip(plan.block_ids.items(), sizes):
            ids = async_tensor_h2d(block_ids, device)
            group_out = out[row : row + size]
            for layer in self._layers[group_id]:
                self._accumulate(layer, ids, group_out)
            row += size
        if self._tp_group is not None:
            dist.all_reduce(out, group=self._tp_group)
        return out

    def _accumulate(
        self, layer: _Layer, block_ids: torch.Tensor, out: torch.Tensor
    ) -> None:
        ratio = layer.blocks_per_block
        page_bytes = layer.cache[0].numel() * layer.cache.element_size() * ratio
        rows_per_chunk = max(1, _CHUNK_BYTES // page_bytes)
        offsets = torch.arange(ratio, device=block_ids.device)
        for start in range(0, len(block_ids), rows_per_chunk):
            ids = block_ids[start : start + rows_per_chunk]
            kernel_ids = (ids[:, None] * ratio + offsets).flatten()
            pages = layer.cache.index_select(0, kernel_ids).view(torch.uint8)
            pieces = _as_pieces(pages.view(len(ids), page_bytes))
            weights = self._piece_weights(pieces.shape[1], pieces.device)
            out[start : start + len(ids)] += (pieces * weights).sum(1) * layer.factor

    def _piece_weights(self, num_pieces: int, device: torch.device) -> torch.Tensor:
        weights = self._weights.get((num_pieces, device))
        if weights is None:
            weights = torch.arange(num_pieces, dtype=torch.int64, device=device)
            weights = 2 * weights + 1
            self._weights[(num_pieces, device)] = weights
        return weights


def _as_pieces(pages: torch.Tensor) -> torch.Tensor:
    """``[rows, bytes]`` uint8 pages as int64 pieces of 8 bytes, or of 4."""
    if pages.shape[1] % 8 == 0:
        return pages.view(torch.int64)
    if pages.shape[1] % 4 == 0:
        return pages.view(torch.int32).to(torch.int64)
    return pages.to(torch.int64)
