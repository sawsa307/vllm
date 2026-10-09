# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only NIXL connectors that corrupt loaded KV, for validating KV
checksums end to end (manual harness, see kv_checksum_validation.md; not
wired into CI).

Loaded via ``kv_connector_module_path``: the connector factory imports this
module and looks up the class named by ``kv_connector``, so the classes keep
the stock names ``NixlConnector`` and ``NixlPushConnector``.

With ``kv_connector_extra_config.test_corrupt_kv`` set on the consumer, they
flip one bit of the first block of every finished load, after the connector
reports the load finished and before the worker checksums it, as silent
transfer corruption would.
"""

from vllm.distributed.kv_transfer.kv_checksum.worker import (
    KVChecksumWorker,
    get_kv_checksum_worker,
)
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
from vllm.logger import init_logger

logger = init_logger(__name__)


class _CorruptLoadsMixin(NixlBaseConnector):
    """Corrupts the KV of every load the worker connector reports done."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._corrupt = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "test_corrupt_kv", False
            )
        )

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        results = super().get_transfer_results(finished_req_ids)
        if self._corrupt and (checksum_worker := get_kv_checksum_worker()):
            for req_id in results.finished_recving - results.failed_recving:
                _flip_first_block(checksum_worker, req_id)
        return results


def _flip_first_block(checksum_worker: KVChecksumWorker, req_id: str) -> None:
    # The blocks the worker checksums once the load finished, and its view of
    # each group's first layer: [blocks, kernel blocks, heads, tokens, pieces].
    blocks = checksum_worker._reqs_to_recv.get(req_id)
    for group_id, (_, block_ids) in blocks.groups.items() if blocks else ():
        if views := checksum_worker._layer_views.get(group_id):
            views[0].pieces[block_ids[0], 0, 0, 0, 0] ^= 1
            logger.warning(
                "test_corrupt_kv: flipped a bit in block %d (group %d) of request %s",
                block_ids[0],
                group_id,
                req_id,
            )
            return


class NixlConnector(_CorruptLoadsMixin, NixlPullConnector):
    """Stock-named pull connector that can corrupt loaded KV."""


class NixlPushConnector(_CorruptLoadsMixin, _StockNixlPushConnector):
    """Stock-named push connector that can corrupt loaded KV."""
