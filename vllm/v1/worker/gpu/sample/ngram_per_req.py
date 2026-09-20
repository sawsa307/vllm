# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request no-repeat-ngram custom logits processor for Model Runner V2.

Used by DeepSeek-OCR and derived OCR models to prevent repeated n-gram
loops in long-document generation. Loaded as a custom logits processor::

    vllm serve ... --logits-processors \\
        vllm.v1.worker.gpu.sample.ngram_per_req:NGramPerReqLogitsProcessorV2

Configured per request via ``SamplingParams.extra_args``
(``ngram_size``, ``window_size``, ``whitelist_token_ids``).
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor import STR_SPEC_DEC_REJECTS_LOGITSPROCS
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsProcessor as V2LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

MAX_NGRAM_WHITELIST_SIZE = 1024


def _validate_ngram_params(params: SamplingParams) -> None:
    ngram_size = params.extra_args and params.extra_args.get("ngram_size")
    window_size = params.extra_args and params.extra_args.get("window_size", 100)
    whitelist_token_ids = params.extra_args and params.extra_args.get(
        "whitelist_token_ids", None
    )
    # if ngram_size is not provided, skip validation because the processor
    # will not be used.
    if ngram_size is None:
        return

    if not isinstance(ngram_size, int) or ngram_size <= 0:
        raise ValueError(
            f"`ngram_size` has to be a strictly positive integer, got {ngram_size}."
        )
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(
            f"`window_size` has to be a strictly positive integer, got {window_size}."
        )
    if whitelist_token_ids is not None and not isinstance(
        whitelist_token_ids, Iterable
    ):
        raise ValueError(
            "`whitelist_token_ids` has to be a sequence of integers, "
            f"got {whitelist_token_ids}."
        )


def _ban_repeated_ngrams(
    logits: torch.Tensor,
    all_token_ids: torch.Tensor,
    slots: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
    ngram_size: int,
    window_size: torch.Tensor,
    width: int,
    whitelist_ids: torch.Tensor | None = None,
    whitelist_len: torch.Tensor | None = None,
) -> None:
    """Ban tokens that would repeat an n-gram, batched over rows sharing
    `ngram_size`. Matches the V1 `NoRepeatNGramLogitsProcessor` semantics of
    `vllm.model_executor.models.deepseek_ocr`.

    Args:
        logits: [num_rows, vocab_size] logits to modify in place.
        all_token_ids: [num_slots, max_model_len] prompt + output token ids.
        slots: [num_rows] request slot per row.
        total_len: [num_rows] prompt_len + output_len per row.
        prompt_len: [num_rows] prompt length per row; only output tokens are
            scanned.
        ngram_size: n, shared by all rows.
        window_size: [num_rows] per-row search window over output tokens.
        width: max over rows of min(out_len, window_size) - ngram_size + 1;
            must be positive.
        whitelist_ids: [num_rows, K] padded token ids exempt from banning.
        whitelist_len: [num_rows] valid entries per row of `whitelist_ids`.

    """
    device = logits.device
    n = ngram_size
    if n == 1:
        # V1 quirk, kept for parity: `output_ids[-(n - 1):]` is the whole
        # sequence when n == 1, which never equals the empty ngram prefix,
        # so the V1 processor bans nothing for ngram_size == 1.
        return
    j = torch.arange(width, device=device)
    # Candidate ngram start as an absolute index into all_token_ids; j = 0 is
    # the most recent start (the ngram ending at the last token).
    start = (total_len - n)[:, None] - j[None, :]
    # Valid while the ngram starts within the output window:
    # start >= prompt_len (output tokens only) and start >= total - window.
    valid = (start >= prompt_len[:, None]) & (
        start >= (total_len - window_size)[:, None]
    )
    cols = start[:, :, None] + torch.arange(n, device=device)[None, None, :]
    ngrams = all_token_ids[slots[:, None, None], cols.clamp(min=0)].long()
    prefix_cols = total_len[:, None] - (n - 1) + torch.arange(n - 1, device=device)
    prefix = all_token_ids[slots[:, None], prefix_cols.clamp(min=0)].long()
    match = (ngrams[..., : n - 1] == prefix[:, None, :]).all(-1) & valid
    banned = ngrams[..., -1]
    if whitelist_ids is not None and whitelist_len is not None:
        k = torch.arange(whitelist_ids.shape[1], device=device)
        wl_valid = k[None, :] < whitelist_len[:, None]
        whitelisted = (
            (banned[:, :, None] == whitelist_ids[:, None, :]) & wl_valid[:, None, :]
        ).any(-1)
        match &= ~whitelisted
    # scatter_ would race when the same token is banned at one position and
    # merely gathered at another; amax is order-independent, so a True write
    # always wins. float32 because CUDA scatter_gather lacks a bool kernel.
    banned_mask = torch.zeros_like(logits)
    banned_mask.scatter_reduce_(
        1, banned.clamp(0, logits.shape[1] - 1), match.float(), reduce="amax"
    )
    logits.masked_fill_(banned_mask.bool(), -float("inf"))


class NGramPerReqLogitsProcessorV2(V2LogitsProcessor):
    """No-repeat-ngram logits processor for Model Runner V2.

    Same per-request semantics as the V1-runner `NGramPerReqLogitsProcessor`
    (wrapper of `NoRepeatNGramLogitsProcessor`) in
    `vllm.model_executor.models.deepseek_ocr`, against the batched V2 logits
    processor interface. Configured per request via `SamplingParams.extra_args`
    (`ngram_size`, `window_size`, `whitelist_token_ids`); requests without
    `ngram_size` are passed through unchanged.

    Speculative decoding is not supported, same as the V1 runner.
    """

    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        if vllm_config.speculative_config is not None:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        self.req_states = req_states
        self.device = req_states.device
        self.prompt_len_np = req_states.prompt_len.np
        max_num_reqs = req_states.max_num_reqs
        self.ngram_size = np.zeros(max_num_reqs, dtype=np.int64)
        self.window_size = np.zeros(max_num_reqs, dtype=np.int64)
        self.use_ngram = np.zeros(max_num_reqs, dtype=bool)
        # req slot -> whitelist token ids.
        self.whitelist_ids: dict[int, list[int]] = {}

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        _validate_ngram_params(params)
        whitelist = params.extra_args and params.extra_args.get("whitelist_token_ids")
        if whitelist is not None:
            n_unique = len(list(dict.fromkeys(whitelist)))
            if n_unique > MAX_NGRAM_WHITELIST_SIZE:
                raise ValueError(
                    f"Too many whitelist token IDs: {n_unique}. "
                    f"The max size is {MAX_NGRAM_WHITELIST_SIZE}."
                )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        extra_args = sampling_params.extra_args or {}
        ngram_size = extra_args.get("ngram_size")
        self.whitelist_ids.pop(req_idx, None)
        if ngram_size is None:
            self.use_ngram[req_idx] = False
            return False
        # Argument validity is guaranteed by validate_params at admission.
        self.ngram_size[req_idx] = ngram_size
        self.window_size[req_idx] = extra_args.get("window_size", 100)
        whitelist = extra_args.get("whitelist_token_ids")
        if whitelist:
            self.whitelist_ids[req_idx] = list(dict.fromkeys(whitelist))
        self.use_ngram[req_idx] = True
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        req_indices = ctx.idx_mapping_np
        active_rows = np.flatnonzero(self.use_ngram[req_indices])
        if active_rows.size == 0:
            return logits
        if logits.shape[0] != req_indices.shape[0]:
            # Unreachable: speculative decoding is rejected in __init__.
            raise ValueError(
                "NGramPerReqLogitsProcessorV2 does not support draft-expanded "
                "logits (speculative decoding)."
            )

        slots = req_indices[active_rows]
        total_len = ctx.seq_lens_upper_bound_np[active_rows].astype(np.int64)
        prompt_len = self.prompt_len_np[slots].astype(np.int64)
        out_len = total_len - prompt_len
        ngram = self.ngram_size[slots]
        window = self.window_size[slots]
        keep = (out_len >= ngram) & (window >= ngram)
        if not np.any(keep):
            return logits
        active_rows = active_rows[keep]
        slots, total_len, prompt_len = slots[keep], total_len[keep], prompt_len[keep]
        out_len, ngram, window = out_len[keep], ngram[keep], window[keep]

        all_token_ids = self.req_states.all_token_ids.gpu
        for n in np.unique(ngram):
            g = np.flatnonzero(ngram == n)
            width = int((np.minimum(out_len[g], window[g]) - n + 1).max())
            rows_t = async_tensor_h2d(active_rows[g], self.device)
            sub = logits.index_select(0, rows_t)
            whitelist = [self.whitelist_ids.get(int(s)) for s in slots[g]]
            whitelist_ids_t = whitelist_len_t = None
            if any(whitelist):
                max_wl = max(len(w) for w in whitelist if w)
                ids_np = np.zeros((len(g), max_wl), dtype=np.int64)
                len_np = np.zeros(len(g), dtype=np.int64)
                for i, w in enumerate(whitelist):
                    if w:
                        ids_np[i, : len(w)] = w
                        len_np[i] = len(w)
                whitelist_ids_t = async_tensor_h2d(ids_np, self.device)
                whitelist_len_t = async_tensor_h2d(len_np, self.device)
            _ban_repeated_ngrams(
                sub,
                all_token_ids,
                slots=async_tensor_h2d(slots[g], self.device),
                total_len=async_tensor_h2d(total_len[g], self.device),
                prompt_len=async_tensor_h2d(prompt_len[g], self.device),
                ngram_size=int(n),
                window_size=async_tensor_h2d(window[g], self.device),
                width=width,
                whitelist_ids=whitelist_ids_t,
                whitelist_len=whitelist_len_t,
            )
            logits.index_copy_(0, rows_t, sub)
        return logits
