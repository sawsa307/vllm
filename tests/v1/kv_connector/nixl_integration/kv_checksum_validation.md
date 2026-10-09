# KV checksum validation (NIXL)

Manual GPU validation of KV checksums (`kv_transfer_config.enable_kv_checksum`)
for NIXL pull and push. The unit tests in `tests/v1/kv_connector/unit/`
(`test_kv_checksum.py`, `test_kv_checksum_scheduler.py`) cover the
computation and the scheduler logic on CPU; this harness checks the real
system: actual KV layouts and attention backends, NIXL transfers, the
model runner's streams, and the engine's failure handling.

## Files

- `kv_checksum_fault_injection.py`: `NixlConnector` and `NixlPushConnector`
  subclasses, loaded on the consumer with
  `kv_connector_module_path=tests.v1.kv_connector.nixl_integration.kv_checksum_fault_injection`.
  With `kv_connector_extra_config.test_corrupt_kv=true`, they flip one bit
  of the first block the checksum worker checks for every finished load,
  after the connector reports the load finished and before the worker
  checksums it. Without it they behave like the stock connectors.
- `kv_checksum_smoke.sh`: one producer, one consumer and the toy proxy on one
  node, four stages (below), each with fresh servers. It checks each stage's
  results and exits non-zero if any stage misses them.

## Running

```bash
# Pull, TP1 -> TP1 (GPUs 0 and 1)
bash tests/v1/kv_connector/nixl_integration/kv_checksum_smoke.sh

# Push
CONNECTOR=NixlPushConnector bash .../kv_checksum_smoke.sh

# Heterogeneous TP
PREFILLER_TP_SIZE=2 PREFILLER_GPUS=0,1 DECODER_GPUS=2 bash .../kv_checksum_smoke.sh
DECODER_TP_SIZE=2 DECODER_GPUS=1,2 bash .../kv_checksum_smoke.sh

# Model runner V2
VLLM_USE_V2_MODEL_RUNNER=1 bash .../kv_checksum_smoke.sh
```

Logs and per-stage outputs go to `$LOG_DIR` (default `/tmp/kv_checksum_smoke`).

## Stages and expected results

Each stage sends `NUM_REQUESTS` (default 4) greedy requests. A request
succeeds when text comes back: the toy proxy streams the consumer's
response, so it has already answered HTTP 200 when a consumer request fails.
The stage then counts log lines: `mismatches` ("KV checksum mismatch",
consumer), `unverified` ("KV checksums not verified", consumer), `not_sent`
("KV checksums not sent", producer) and `syncs` ("GPU<->CPU sync detected",
both servers, which run with `VLLM_GPU_SYNC_CHECK=error`).

| Stage | Consumer config | Requests succeeding | Mismatches |
| --- | --- | --- | --- |
| `clean` | fail-closed | all | 0 |
| `corrupt_fail_open` | `test_corrupt_kv` | all | one per request |
| `corrupt_fail` | fail-closed, `kv_load_failure_policy=fail`, `test_corrupt_kv` | none | one per request |
| `corrupt_recompute` | fail-closed, `kv_load_failure_policy=recompute`, `test_corrupt_kv` | all | one per request |

Every stage also expects no unverified load, no unsent checksums and no
host sync.

- `clean` shows the checksums pass end to end without false positives and
  without host syncs.
- The `corrupt_*` stages show a one-bit corruption is caught and handled per
  policy. With equal TP sizes the script also reports whether the recomputed
  outputs match the clean run (informational: numerics may differ).
- `VLLM_GPU_SYNC_CHECK=error` fails the engine at the first host sync in a
  step, including syncs outside the checksum code; the stage's later
  results are then meaningless, so look at the first failure in the logs.
- "KV checksums not verified" and "KV checksums not sent" are logged once
  per reason per process, so they count reasons, not requests.

## Results

Not run yet on this implementation; GPU runs pending.
