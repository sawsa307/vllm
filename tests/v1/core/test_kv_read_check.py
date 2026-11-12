# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache read checks (``kv_read_check_interval``): a real scheduler and a
real checksummer over a CPU KV cache, with a fake forward that writes each
scheduled token's KV."""

from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.core.kv_read_check import KVReadCheckPlan, KVReadChecksums
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu.kv_read_check import KVReadChecksummer
from vllm.v1.worker.utils import allocate_kv_cache

from .utils import create_requests, create_scheduler

BLOCK_SIZE = 4
NUM_BLOCKS = 64
SPEC = FullAttentionSpec(
    block_size=BLOCK_SIZE, num_kv_heads=2, head_size=8, dtype=torch.float32
)


def _kv_cache_config(
    layer_names: list[str], layout: str = "LBHNC", num_blocks: int = NUM_BLOCKS
) -> KVCacheConfig:
    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = layout
    config.attention_config.hisparse_config = None
    groups = [KVCacheGroupSpec(layer_names, SPEC)]
    return get_kv_cache_config_from_groups(
        config, groups, SPEC.page_size_bytes * len(layer_names) * num_blocks
    )


def _allocate(
    config: KVCacheConfig, layout: str, kernel_block_size: int | None = None
) -> dict[str, torch.Tensor]:
    kernel_block_sizes = None if kernel_block_size is None else [kernel_block_size]
    return allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout[layout], kernel_block_sizes
    )


class _FakeWorker:
    """Runs each step's "forward" when it is scheduled: writes each scheduled
    token's KV into its slot (every sampled token is ``SAMPLED``), then
    checksums the plan's blocks. With async scheduling, a step's output is
    processed after the next step is scheduled."""

    SAMPLED = 7

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        config = _kv_cache_config(["layer"])
        self.cache = _allocate(config, "LBHNC")["layer"]
        self.checksummer = KVReadChecksummer(
            config, {"layer": self.cache}, frozenset(), 0, None
        )
        self.depth = 2 if scheduler.scheduler_config.async_scheduling else 1
        self.in_flight: deque = deque()

    def step(self) -> list:
        scheduler_output = self.scheduler.schedule()
        self.in_flight.append((scheduler_output, self._forward(scheduler_output)))
        if len(self.in_flight) < self.depth:
            return []
        engine_outputs = self.scheduler.update_from_output(*self.in_flight.popleft())
        return [o for outputs in engine_outputs.values() for o in outputs.outputs]

    def _forward(self, scheduler_output) -> ModelRunnerOutput:
        manager = self.scheduler.kv_cache_manager
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.scheduler.requests[req_id]
            (block_ids,) = manager.get_block_ids(req_id)
            end = request.num_computed_tokens
            for position in range(end - num_tokens, end):
                token = (
                    request.all_token_ids[position]
                    if position < request.num_tokens
                    else self.SAMPLED
                )
                slot = self.slot(block_ids[position // BLOCK_SIZE], position)
                slot.fill_(token * 1000 + position)
        plan = scheduler_output.kv_read_check_plan
        checksums = None
        if plan is not None:
            values = self.checksummer.compute(plan).numpy().view(np.uint64)
            checksums = KVReadChecksums(plan.step_id, values)
        req_ids = list(scheduler_output.num_scheduled_tokens)
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[self.SAMPLED] for _ in req_ids],
            kv_read_checksums=checksums,
        )
        return output

    def slot(self, block_id: int, position: int) -> torch.Tensor:
        return self.cache[block_id, :, position % BLOCK_SIZE]


def _scheduler(
    enable_prefix_caching: bool = False, async_scheduling: bool = False
) -> Scheduler:
    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        kv_cache_spec=SPEC,
        enable_prefix_caching=enable_prefix_caching,
        kv_read_check_interval=1,
    )
    assert scheduler.kv_read_check is not None
    return scheduler


def _request(scheduler: Scheduler, req_id: str, prompt: list[int]) -> Request:
    (request,) = create_requests(
        num_requests=1, max_tokens=8, block_size=BLOCK_SIZE, req_ids=[req_id]
    )
    request.prompt_token_ids[:] = prompt
    request._all_token_ids[: len(prompt)] = prompt
    scheduler.add_request(request)
    return request


def _errors(outputs: list) -> dict[str, str]:
    return {o.request_id: o.error.code for o in outputs if o.error is not None}


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_untouched_kv_completes(async_scheduling):
    """Reads of KV nobody overwrote pass, through prefill and decode."""
    scheduler = _scheduler(async_scheduling=async_scheduling)
    worker = _FakeWorker(scheduler)
    request = _request(scheduler, "a", list(range(10)))
    while not request.is_finished():
        assert not _errors(worker.step())
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_overwritten_block_fails_request(async_scheduling):
    """A stray write into a block a running request reads fails that request
    with a retryable error, and only that request."""
    scheduler = _scheduler(async_scheduling=async_scheduling)
    worker = _FakeWorker(scheduler)
    a = _request(scheduler, "a", list(range(10)))
    b = _request(scheduler, "b", list(range(100, 110)))
    for _ in range(3):
        assert not _errors(worker.step())
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids("a")
    worker.slot(block_ids[1], 5).fill_(-1.0)

    # A step already in flight checksummed before the write.
    outputs = [o for _ in range(worker.depth) for o in worker.step()]

    assert _errors(outputs) == {"a": "kv_cache_corrupted"}
    assert a.status == RequestStatus.FINISHED_ERROR
    assert not b.is_finished()


def test_block_table_pointing_at_other_intact_block_fails():
    """Expected checksums are per request and position: reading another
    request's intact block at the same position mismatches."""
    scheduler = _scheduler()
    worker = _FakeWorker(scheduler)
    _request(scheduler, "a", list(range(10)))
    _request(scheduler, "b", list(range(100, 110)))
    for _ in range(3):
        assert not _errors(worker.step())
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    a_blocks = managers[0].req_to_blocks["a"]
    own, a_blocks[0] = a_blocks[0], managers[0].req_to_blocks["b"][0]
    scheduler_output = scheduler.schedule()
    a_blocks[0] = own
    scheduler.schedule = lambda: scheduler_output  # type: ignore[method-assign]

    assert _errors(worker.step()) == {"a": "kv_cache_corrupted"}


def test_prefix_hit_inherits_checksums_of_the_computing_request():
    """A prefix hit takes the checksums recorded when the KV was computed, so
    a cached block corrupted after its request finished fails the next request
    that hits it, and stops serving hits."""
    scheduler = _scheduler(enable_prefix_caching=True)
    worker = _FakeWorker(scheduler)
    prompt = list(range(12))
    first = _request(scheduler, "a", prompt)
    while not first.is_finished():
        (first_ids,) = scheduler.kv_cache_manager.get_block_ids("a")
        assert not _errors(worker.step())
    worker.slot(first_ids[0], 0).fill_(-1.0)

    _request(scheduler, "b", prompt)
    assert not _errors(worker.step())
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids("b")
    assert block_ids[:2] == first_ids[:2]
    assert _errors(worker.step()) == {"b": "kv_cache_corrupted"}

    _request(scheduler, "c", prompt)
    worker.step()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids("c")
    assert block_ids[0] != first_ids[0]


@pytest.mark.parametrize("layout", ["LBHNC", "LHBNC", "BLHNC"])
@pytest.mark.parametrize("kernel_block_size", [None, BLOCK_SIZE // 2])
def test_checksum_covers_each_byte_of_the_block_only(layout, kernel_block_size):
    """Any layout and kernel block split: changing one byte of a block's page
    in any layer changes that block's checksum and no other."""
    if kernel_block_size is not None and layout != "LBHNC":
        pytest.skip("only dense block pages split into kernel blocks")
    names = ["model.layers.0.attn", "model.layers.1.attn"]
    config = _kv_cache_config(names, layout, num_blocks=8)
    caches = _allocate(config, layout, kernel_block_size)
    generator = torch.Generator().manual_seed(0)
    for cache in caches.values():
        cache.copy_(torch.randn(cache.shape, generator=generator))
    checksummer = KVReadChecksummer(config, caches, frozenset(), 0, None)
    plan = KVReadCheckPlan(0, {0: np.array([1, 2, 5], dtype=np.int64)})
    before = checksummer.compute(plan)

    ratio = caches[names[1]].shape[0] // 8
    page = caches[names[1]][2 * ratio + ratio - 1].view(torch.uint8)
    page[-1, -1, -1] ^= 1
    after = checksummer.compute(plan)

    assert (before != after).tolist() == [False, True, False]
