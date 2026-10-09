# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only NIXL connectors that corrupt loaded KV, for validating KV
checksums end to end (manual harness, see kv_checksum_validation.md; not
wired into CI).

Loaded via ``kv_connector_module_path``: the connector factory imports this
module and looks up the stock class name, so the classes must keep the
names ``NixlConnector`` and ``NixlPushConnector`` (several config checks
dispatch on the connector class name).

Arm the fault on the consumer through ``kv_connector_extra_config``:
``test_corrupt_kv`` (bool) flips one bit of the first block each load
wrote, after the connector reports the load finished and before the
worker checksums it, as silent transfer corruption would.
``test_corrupt_kv_piece`` (int, default 0) picks the 8-byte piece of the
block's first token and head to flip.
"""

from vllm.distributed.kv_transfer.kv_checksum.worker import get_kv_checksum_worker
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector as _StockNixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger

logger = init_logger(__name__)


class _CorruptLoadsMixin(NixlBaseConnector):
    """Corrupts the KV of every load the worker connector reports done."""

    def _init_fault(self, vllm_config, role: KVConnectorRole) -> None:
        cfg = vllm_config.kv_transfer_config
        self._corrupt = role == KVConnectorRole.WORKER and cfg.get_from_extra_config(
            "test_corrupt_kv", False
        )
        self._corrupt_piece = int(cfg.get_from_extra_config("test_corrupt_kv_piece", 0))
        # Request id -> (group id, logical block id) of its first loaded block.
        self._first_loaded_block: dict[str, tuple[int, int]] = {}

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        if self._corrupt:
            metadata = self._connector_metadata
            assert isinstance(metadata, NixlConnectorMetadata)
            group_ids = self.kv_cache_config.transfer_group_ids
            for req_id, meta in metadata.reqs_to_recv.items():
                first = next(
                    (
                        (group_ids[i], block_ids[0])
                        for i, block_ids in enumerate(meta.local_block_ids)
                        if block_ids
                    ),
                    None,
                )
                if first is not None:
                    self._first_loaded_block[req_id] = first
        super().start_load_kv(forward_context, **kwargs)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        results = super().get_transfer_results(finished_req_ids)
        for req_id in results.finished_recving:
            first = self._first_loaded_block.pop(req_id, None)
            if first is not None and req_id not in results.failed_recving:
                self._flip(req_id, *first)
        return results

    def _flip(self, req_id: str, group_id: int, block_id: int) -> None:
        # The checksum worker's own view of the group's first layer:
        # [blocks, kernel blocks, heads, tokens, 8-byte pieces].
        checksum_worker = get_kv_checksum_worker()
        views = checksum_worker and checksum_worker._layer_views.get(group_id)
        if not views:
            logger.warning(
                "test_corrupt_kv: no checksummed layer in group %d", group_id
            )
            return
        pieces = views[0].pieces
        piece = self._corrupt_piece % pieces.shape[-1]
        pieces[block_id, 0, 0, 0, piece] ^= 1
        logger.warning(
            "test_corrupt_kv: flipped a bit of piece %d in block %d (group %d) "
            "of request %s",
            piece,
            block_id,
            group_id,
            req_id,
        )


class NixlConnector(_CorruptLoadsMixin, NixlPullConnector):
    """Stock-named pull connector that can corrupt loaded KV."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._init_fault(vllm_config, role)


class NixlPushConnector(_CorruptLoadsMixin, _StockNixlPushConnector):
    """Stock-named push connector that can corrupt loaded KV."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._init_fault(vllm_config, role)
