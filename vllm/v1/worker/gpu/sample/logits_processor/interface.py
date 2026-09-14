# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm import SamplingParams
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class LogitsContext:
    """The current step's batch layout, passed to every ``apply()`` call.

    A row is a logits row, not a request: rows are reordered every step, and
    under speculative decoding a request owns one row per draft token.

    Reading a request's generated tokens takes two sources. Committed tokens
    live in the persistent ``all_token_ids`` tensor, valid up to that
    request's ``total_len``. This step's draft tokens are not there yet; they
    are in ``input_ids``, at ``row - expanded_local_pos[row]`` plus the draft
    offset. A processor that reads only committed tokens will miss the drafts
    it is being asked to score.
    """

    # [num_logits_rows] row -> persistent request slot.
    expanded_idx_mapping: torch.Tensor
    # [num_reqs] batch position -> persistent request slot, on the host, for
    # skipping work without a device sync.
    idx_mapping_np: np.ndarray
    # [num_logits_rows] row -> its offset among the rows of its own request.
    expanded_local_pos: torch.Tensor
    # [num_logits_rows] token fed to the model at each row's input position.
    input_ids: torch.Tensor
    # [num_logits_rows] position of each row within its sequence.
    pos: torch.Tensor


class LogitsProcessor(ABC):
    """Custom logits processor for Model Runner V2.

    Per-request state is keyed by the request slot index, and slots are
    recycled through a free list, so ``remove_request()`` must release
    anything keyed by slot.

    ``apply()`` runs after the built-in bias, penalty, bad-words and grammar
    stages and before temperature, min_p and top-k/top-p, so it sees unscaled
    logits and must not re-inflate grammar-masked tokens.

    Anything that is stable for a request's lifetime belongs in ``__init__``
    or ``add_request()``; ``apply()`` receives only what changes per step.
    """

    def __init__(  # noqa: B027
        self, vllm_config: "VllmConfig", req_states: RequestState
    ):
        """Capture what stays constant for the processor's lifetime.

        ``req_states`` is the persistent batch: it carries the token history
        (``all_token_ids`` bounded by ``total_len``, ``prompt_len``,
        ``prefill_len``) that the model runner maintains on device, plus
        ``device``, ``max_num_reqs`` and ``vocab_size``. Read from it rather
        than staging a private copy of the tokens. Treat it as read-only --
        its ``add_request()``, ``remove_request()`` and
        ``apply_staged_writes()`` belong to the model runner.
        """

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        """Initialize per-slot state for a request entering the batch.

        Returns whether this processor modifies logits for this request. The
        sampler ORs the return value across every stage into a per-request
        flag and skips the whole logits-processing pipeline for batches in
        which no request needs it. Defaults to True.

        Returning False gates pipeline admission only. When the pipeline runs
        for other requests, ``apply()`` still sees every row, so processors
        must filter rows themselves via ``expanded_idx_mapping``.
        """
        return True

    def remove_request(self, req_idx: int) -> None:  # noqa: B027
        """Tear down per-slot state for a request leaving the batch."""

    @abstractmethod
    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        """Modify logits in place or return a new tensor.

        ``apply()`` is called once for the whole batch, including rows of
        requests this processor declined in ``add_request()``, so filter rows
        via ``ctx.expanded_idx_mapping``.

        Args:
            logits: [num_logits_rows, vocab_size] float32 tensor.
            ctx: this step's batch layout.
        """
        raise NotImplementedError
