#!/bin/bash
# KV checksum smoke validation on one node: 1P1D with the toy proxy.
# See kv_checksum_validation.md for what each stage checks. Exits non-zero
# if any stage misses its expected results.
#
# Environment:
#   MODEL                          default Qwen/Qwen3-0.6B
#   CONNECTOR                      NixlConnector (pull) or NixlPushConnector
#   PREFILLER_TP_SIZE / DECODER_TP_SIZE, PREFILLER_GPUS / DECODER_GPUS
#   NUM_REQUESTS                   default 4
#   LOG_DIR                        default /tmp/kv_checksum_smoke
# Other variables (e.g. VLLM_USE_V2_MODEL_RUNNER) pass through to the servers.
# e.g. PREFILLER_TP_SIZE=2 PREFILLER_GPUS=0,1 CONNECTOR=NixlPushConnector \
#   bash kv_checksum_smoke.sh
set -u

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
CONNECTOR=${CONNECTOR:-NixlConnector}
P_TP=${PREFILLER_TP_SIZE:-1}
D_TP=${DECODER_TP_SIZE:-1}
P_GPUS=${PREFILLER_GPUS:-0}
D_GPUS=${DECODER_GPUS:-1}
LOG_DIR=${LOG_DIR:-/tmp/kv_checksum_smoke}
NUM_REQUESTS=${NUM_REQUESTS:-4}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)"
# The consumer loads the stock-named connector from this test module, which
# corrupts loaded KV when test_corrupt_kv is set.
FAULT_MODULE=tests.v1.kv_connector.nixl_integration.kv_checksum_fault_injection
mkdir -p "$LOG_DIR"
FAILED=0

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -- -"$pid" 2>/dev/null; done
  for pid in "${PIDS[@]}"; do
    for _ in $(seq 1 60); do
      kill -0 -- -"$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 -- -"$pid" 2>/dev/null
  done
  PIDS=()
}
trap cleanup EXIT

wait_health() {  # $1=port, $2=server pid
  for _ in $(seq 1 120); do
    curl -sf -m 3 "http://localhost:$1/health" -o /dev/null && return 0
    kill -0 -- -"$2" 2>/dev/null || { echo "FAIL: server on port $1 exited"; return 1; }
    sleep 5
  done
  echo "FAIL: port $1 never healthy"
  return 1
}

# $1=stage name, $2=consumer kv_transfer_config fields after kv_role
launch() {
  local stage=$1 consumer_fields=$2
  local p_cfg='{"kv_connector":"'$CONNECTOR'","kv_role":"kv_producer","enable_kv_checksum":true}'
  local d_cfg='{"kv_connector":"'$CONNECTOR'","kv_connector_module_path":"'$FAULT_MODULE'","kv_role":"kv_consumer","enable_kv_checksum":true'$consumer_fields'}'
  # VLLM_GPU_SYNC_CHECK=error fails the engine on any host sync in a step.
  CUDA_VISIBLE_DEVICES=$P_GPUS VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    VLLM_GPU_SYNC_CHECK=error PYTHONPATH="$REPO" \
    setsid vllm serve "$MODEL" --port 8100 --tensor-parallel-size "$P_TP" \
    --gpu-memory-utilization 0.3 --kv-transfer-config "$p_cfg" \
    < /dev/null > "$LOG_DIR/$stage.p.log" 2>&1 &
  local p_pid=$!
  CUDA_VISIBLE_DEVICES=$D_GPUS VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
    VLLM_GPU_SYNC_CHECK=error PYTHONPATH="$REPO" \
    setsid vllm serve "$MODEL" --port 8200 --tensor-parallel-size "$D_TP" \
    --gpu-memory-utilization 0.3 --kv-transfer-config "$d_cfg" \
    < /dev/null > "$LOG_DIR/$stage.d.log" 2>&1 &
  local d_pid=$!
  setsid python "$SCRIPT_DIR/toy_proxy_server.py" --port 8192 \
    --prefiller-hosts localhost --prefiller-ports 8100 \
    --decoder-hosts localhost --decoder-ports 8200 \
    < /dev/null > "$LOG_DIR/$stage.proxy.log" 2>&1 &
  PIDS+=("$p_pid" "$d_pid" $!)
  wait_health 8100 "$p_pid" && wait_health 8200 "$d_pid" && sleep 3
}

# Sends NUM_REQUESTS greedy requests; prints "ok|failed <text>" per request.
# The proxy streams the consumer's response, so it has already answered 200
# when a consumer request fails: a request succeeded only if text came back.
probe() {
  local words
  words=$(printf 'the quick brown fox %.0s' $(seq 1 40))
  for i in $(seq 1 "$NUM_REQUESTS"); do
    curl -s -m 120 http://localhost:8192/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","prompt":"KV checksum probe '"$i"': '"$words"'","max_tokens":16,"temperature":0}' |
      python -c '
import json, sys
try:
    text = json.loads(sys.stdin.read())["choices"][0]["text"]
except Exception:
    text = ""
print("ok" if text else "failed", json.dumps(text))'
  done
}

count() {  # $1=pattern, $2..=files
  local pattern=$1
  shift
  cat "$@" 2>/dev/null | grep -c -- "$pattern"
}

# $1=stage, $2=expected ok requests, $3=expected mismatches
check() {
  local stage=$1 want_ok=$2 want_mismatches=$3
  local ok mismatches unverified not_sent syncs
  ok=$(grep -c '^ok ' "$LOG_DIR/$stage.out")
  mismatches=$(count 'KV checksum mismatch' "$LOG_DIR/$stage.d.log")
  unverified=$(count 'KV checksums not verified' "$LOG_DIR/$stage.d.log")
  not_sent=$(count 'KV checksums not sent' "$LOG_DIR/$stage.p.log")
  syncs=$(count 'GPU<->CPU sync detected' "$LOG_DIR/$stage".[pd].log)
  echo "$stage: ok=$ok/$NUM_REQUESTS mismatches=$mismatches" \
    "unverified=$unverified not_sent=$not_sent syncs=$syncs"
  if [ "$ok" != "$want_ok" ] || [ "$mismatches" != "$want_mismatches" ] ||
      [ "$unverified" != 0 ] || [ "$not_sent" != 0 ] || [ "$syncs" != 0 ]; then
    echo "$stage: FAILED (want ok=$want_ok mismatches=$want_mismatches," \
      "no unverified, not_sent or syncs)"
    FAILED=1
  fi
}

# $1=stage, $2=consumer fields, $3=expected ok requests, $4=expected mismatches
run_stage() {
  echo "=== $1 ==="
  if launch "$1" "$2"; then
    probe > "$LOG_DIR/$1.out"
    sleep 2
    check "$1" "$3" "$4"
  else
    echo "$1: FAILED to start"
    FAILED=1
  fi
  cleanup
}

N=$NUM_REQUESTS
CORRUPT='"kv_connector_extra_config":{"test_corrupt_kv":true}'
run_stage clean ',"kv_checksum_fail_closed":true' "$N" 0
run_stage corrupt_fail_open ",$CORRUPT" "$N" "$N"
run_stage corrupt_fail \
  ',"kv_checksum_fail_closed":true,"kv_load_failure_policy":"fail",'"$CORRUPT" 0 "$N"
run_stage corrupt_recompute \
  ',"kv_checksum_fail_closed":true,"kv_load_failure_policy":"recompute",'"$CORRUPT" \
  "$N" "$N"

# Informational: with equal TP sizes the consumer's own prefill usually
# reproduces the producer's KV, so greedy outputs after a recompute match the
# clean run; numerics may still differ slightly.
if [ "$P_TP" = "$D_TP" ] && [ -s "$LOG_DIR/clean.out" ] &&
    [ -s "$LOG_DIR/corrupt_recompute.out" ]; then
  if diff -q "$LOG_DIR/clean.out" "$LOG_DIR/corrupt_recompute.out" > /dev/null; then
    echo "corrupt_recompute: outputs match the clean run"
  else
    echo "corrupt_recompute: NOTE, outputs differ from the clean run"
  fi
fi

exit $FAILED
