# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read checks of the KV cache (``CacheConfig.kv_read_check_interval``).

Checks one invariant: the KV a request reads at a block position equals the
KV that was final there when the request first had it. Each full block
position of a request gets an expected checksum once its KV is final: computed
by the workers after the step that finds it final, or, on a prefix-cache hit,
copied from the block's checksum recorded by the request that computed it.
Every ``kv_read_check_interval`` steps, the workers checksum every block
position with an expected checksum of every scheduled request after the
forward, and a request whose blocks differ fails.

The expected checksums are kept per request and block position, not per
block, so a block table that points at another, intact block mismatches too.
A block's checksum is a weighted sum of its bytes over the group's layers on
every TP rank (see ``vllm/v1/worker/gpu/kv_read_check.py``); it is compared
only with checksums of the same engine, so it may depend on the layout and
parallelism.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.outputs import RequestError
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Specs whose full blocks are written once, while their tokens are computed.
_APPEND_ONLY_SPECS = (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SlidingWindowMLASpec,
    ChunkedLocalAttentionSpec,
)


@dataclass
class KVReadCheckPlan:
    """The blocks the workers checksum after this step's forward."""

    step_id: int
    # Per KV cache group id, the block ids to checksum, in row order.
    block_ids: dict[int, np.ndarray]


@dataclass
class KVReadChecksums:
    """The workers' checksums of a plan's blocks, summed over TP ranks."""

    step_id: int
    # One uint64 per row, the groups' rows in the plan's order.
    values: np.ndarray


def _append_only(spec: KVCacheSpec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return all(_append_only(s) for s in spec.kv_cache_specs.values())
    return type(spec) in _APPEND_ONLY_SPECS


def kv_read_check_group_ids(kv_cache_config: KVCacheConfig) -> tuple[int, ...]:
    """The KV cache groups that are read checked."""
    return tuple(
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if _checked(group)
    )


def _checked(group: KVCacheGroupSpec) -> bool:
    return (
        _append_only(group.kv_cache_spec)
        and not group.is_eagle_group
        and not group.host_resident
        and group.role == KVCacheGroupRole.DEFAULT
    )


@dataclass
class _Expected:
    """A request's expected checksums in one group, by block position."""

    values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))
    known: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Positions below are recorded, copied, or waiting for their checksum.
    end: int = 0

    def resize(self, end: int) -> None:
        if end > len(self.values):
            size = max(end, 2 * len(self.values))
            self.values = np.resize(self.values, size)
            self.known = np.concatenate(
                [self.known, np.zeros(size - len(self.known), dtype=bool)]
            )
        self.known[end:] = False
        self.end = end


@dataclass
class _RequestState:
    groups: dict[int, _Expected]
    num_preemptions: int
    # Bumped when expected checksums are dropped, so that checksums scheduled
    # before are not recorded.
    epoch: int = 0


@dataclass
class _Segment:
    """Consecutive rows of one request in one group."""

    req_id: str
    epoch: int
    positions: np.ndarray
    block_ids: np.ndarray
    # The expected checksums to verify; None to record the checksums.
    expected: np.ndarray | None = None
    # Per row of a record segment, the block's hash when scheduled.
    hashes: list["BlockHashWithGroupId | None"] | None = None


@dataclass
class _BlockChecksum:
    """A block's checksum, as recorded by the request that computed it."""

    # The block's hash object when recorded. A block re-cached after eviction
    # gets a new hash object, even for the same tokens.
    block_hash: "BlockHashWithGroupId"
    position: int
    value: int


class KVReadChecker:
    """Schedules the read checks and compares their checksums."""

    def __init__(self, kv_cache_config: KVCacheConfig, interval: int):
        self._interval = interval
        self._block_sizes = {
            group_id: kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size
            for group_id in kv_read_check_group_ids(kv_cache_config)
        }
        self._requests: dict[str, _RequestState] = {}
        self._blocks: dict[tuple[int, int], _BlockChecksum] = {}
        self._pending: dict[int, dict[int, list[_Segment]]] = {}
        self._num_steps = 0

    @classmethod
    def create(
        cls, kv_cache_config: KVCacheConfig, interval: int
    ) -> "KVReadChecker | None":
        if interval <= 0:
            return None
        if not kv_read_check_group_ids(kv_cache_config):
            logger.warning("No KV cache group can be read checked.")
            return None
        return cls(kv_cache_config, interval)

    def schedule(
        self, requests: Iterable["Request"], kv_cache_manager: "KVCacheManager"
    ) -> KVReadCheckPlan | None:
        """Plan this step's checksums: the block positions of ``requests``
        whose KV became final, and, every ``interval`` steps, every position
        with an expected checksum.

        ``num_computed_tokens`` and ``num_in_flight_tokens`` of ``requests``
        must include this step.
        """
        verify = self._num_steps % self._interval == 0
        step_id = self._num_steps
        self._num_steps += 1
        segments: dict[int, list[_Segment]] = {g: [] for g in self._block_sizes}
        for request in requests:
            state = self._requests.get(request.request_id)
            if state is None or state.num_preemptions != request.num_preemptions:
                # Recomputed KV may differ in its bits from the earlier KV.
                epoch = state.epoch + 1 if state is not None else 0
                state = _RequestState({}, request.num_preemptions, epoch)
                self._requests[request.request_id] = state
            # Tokens whose KV is final: computed in steps already processed,
            # so it holds no rejected draft tokens.
            num_final = min(
                request.num_computed_tokens - request.num_in_flight_tokens,
                request.num_tokens,
            )
            blocks = kv_cache_manager.get_blocks(request.request_id).blocks
            for group_id, block_size in self._block_sizes.items():
                self._schedule_group(
                    request.request_id,
                    state,
                    group_id,
                    blocks[group_id],
                    min(num_final // block_size, len(blocks[group_id])),
                    verify,
                    segments[group_id],
                )
        segments = {g: s for g, s in segments.items() if s}
        if not segments:
            return None
        self._pending[step_id] = segments
        return KVReadCheckPlan(
            step_id,
            {
                group_id: np.concatenate([s.block_ids for s in group_segments])
                for group_id, group_segments in segments.items()
            },
        )

    def _schedule_group(
        self,
        req_id: str,
        state: _RequestState,
        group_id: int,
        blocks: Sequence["KVCacheBlock"],
        num_full: int,
        verify: bool,
        segments: list[_Segment],
    ) -> None:
        expected = state.groups.setdefault(group_id, _Expected())
        if num_full < expected.end:
            # Recomputed from an earlier position, e.g. after a failed KV load.
            state.epoch += 1
            expected.resize(num_full)

        if verify and expected.end:
            positions = np.flatnonzero(expected.known[: expected.end])
            block_ids = np.array([blocks[p].block_id for p in positions.tolist()])
            freed = np.array([blocks[p].is_null for p in positions.tolist()], bool)
            if freed.any():
                # Out of a sliding window.
                expected.known[positions[freed]] = False
                positions, block_ids = positions[~freed], block_ids[~freed]
            if len(positions):
                segments.append(
                    _Segment(
                        req_id,
                        state.epoch,
                        positions,
                        block_ids.astype(np.int64),
                        expected=expected.values[positions].copy(),
                    )
                )

        start = expected.end
        if num_full <= start:
            return
        expected.resize(num_full)
        record: list[int] = []
        hashes: list[BlockHashWithGroupId | None] = []
        for position in range(start, num_full):
            block = blocks[position]
            if block.is_null:
                continue
            recorded = self._blocks.get((group_id, block.block_id))
            if (
                recorded is not None
                and recorded.block_hash is block.block_hash
                and recorded.position == position
            ):
                expected.values[position] = recorded.value
                expected.known[position] = True
            else:
                record.append(position)
                hashes.append(block.block_hash)
        if record:
            positions = np.array(record, dtype=np.int64)
            segments.append(
                _Segment(
                    req_id,
                    state.epoch,
                    positions,
                    np.array([blocks[p].block_id for p in record], dtype=np.int64),
                    hashes=hashes,
                )
            )

    def update_from_output(
        self, checksums: KVReadChecksums | None
    ) -> tuple[dict[str, RequestError], set[int]]:
        """Record and compare a step's checksums.

        Returns:
            The requests whose KV mismatches, with their error, and the
            mismatching block ids.

        """
        if checksums is None:
            return {}, set()
        # Plans whose output never came, e.g. steps without a forward.
        for step_id in [s for s in self._pending if s < checksums.step_id]:
            del self._pending[step_id]
        segments = self._pending.pop(checksums.step_id, None)
        if segments is None:
            return {}, set()
        errors: dict[str, RequestError] = {}
        bad_blocks: set[int] = set()
        row = 0
        for group_id, group_segments in segments.items():
            for segment in group_segments:
                values = checksums.values[row : row + len(segment.positions)]
                row += len(segment.positions)
                if segment.expected is None:
                    self._record(group_id, segment, values)
                    continue
                if segment.req_id not in self._requests:
                    # Finished: its blocks may already be reused.
                    continue
                bad = values != segment.expected
                if not bad.any():
                    continue
                bad_blocks.update(segment.block_ids[bad].tolist())
                message = (
                    f"KV cache read check mismatch: group {group_id} block "
                    f"positions {segment.positions[bad].tolist()} (blocks "
                    f"{segment.block_ids[bad].tolist()})"
                )
                logger.error("Request %s: %s.", segment.req_id, message)
                errors[segment.req_id] = RequestError(
                    code="kv_cache_corrupted", message=message, retryable=True
                )
        return errors, bad_blocks

    def _record(self, group_id: int, segment: _Segment, values: np.ndarray) -> None:
        assert segment.hashes is not None
        for position, block_id, block_hash, value in zip(
            segment.positions.tolist(),
            segment.block_ids.tolist(),
            segment.hashes,
            values.tolist(),
        ):
            if block_hash is not None:
                self._blocks[(group_id, block_id)] = _BlockChecksum(
                    block_hash, position, value
                )
        state = self._requests.get(segment.req_id)
        if state is None or state.epoch != segment.epoch:
            return
        expected = state.groups[group_id]
        positions = segment.positions[segment.positions < expected.end]
        expected.values[positions] = values[: len(positions)]
        expected.known[positions] = True

    def request_finished(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
