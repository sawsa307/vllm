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
  of the first block of every finished load, after the connector reports
  the load finished and before the worker checksums it.
  `test_corrupt_kv_piece` picks the 8-byte piece of that block's first token
  and head (default 0). Without `test_corrupt_kv` they behave like the stock
  connectors.
- `kv_checksum_smoke.sh`: one producer, one consumer and the toy proxy on one
  node, four stages (below), each with fresh servers. Both servers run with
  `VLLM_GPU_SYNC_CHECK=error`, so a host sync in the engine step fails the
  step.

## Running

```bash
# Pull, TP1 -> TP1 (GPUs 0 and 1)
bash tests/v1/kv_connector/nixl_integration/kv_checksum_smoke.sh

# Push
CONNECTOR=NixlPushConnector bash .../kv_checksum_smoke.sh

# Heterogeneous TP
PREFILLER_TP_SIZE=2 PREFILLER_GPUS=0,1 DECODER_GPUS=2 bash .../kv_checksum_smoke.sh
DECODER_TP_SIZE=2 DECODER_GPUS=1,2 bash .../kv_checksum_smoke.sh
```

Logs and per-stage outputs go to `$LOG_DIR` (default `/tmp/kv_checksum_smoke`).

## Stages and expected results

Each stage sends `NUM_REQUESTS` (default 4) greedy requests and reports the
HTTP codes and log counts: `mismatches` ("KV checksum mismatch", consumer),
`unverified` ("KV checksums not verified", consumer), `not_sent`
("KV checksums not sent", producer) and `syncs` ("GPU<->CPU sync detected",
both).

| Stage | Consumer config | Expected |
| --- | --- | --- |
| `clean` | fail-closed | all 200; mismatches 0, unverified 0, not_sent 0, syncs 0 |
| `corrupt_fail_open` | `test_corrupt_kv` | all 200; mismatches = requests; syncs 0 |
| `corrupt_fail` | fail-closed, `kv_load_failure_policy=fail`, `test_corrupt_kv` | all non-200 (the requests fail); mismatches = requests |
| `corrupt_recompute` | fail-closed, `kv_load_failure_policy=recompute`, `test_corrupt_kv` | all 200; mismatches = requests; outputs match `clean` |

`clean` shows the checksums pass end to end without false positives and
without host syncs. The `corrupt_*` stages show a one-bit corruption is
caught and handled per policy, and with `recompute` the consumer recomputes
the prompt and produces the same output as the clean run.

`warning_once` logs each unverified reason once per process, so
`unverified` counts reasons, not requests.

## Results

Not run yet on this implementation; GPU runs pending.
