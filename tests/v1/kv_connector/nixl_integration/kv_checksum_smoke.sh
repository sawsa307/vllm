#!/bin/bash
# KV checksum smoke validation on one node: 1P1D with the toy proxy.
# See kv_checksum_validation.md for what each stage checks.
#
# Environment:
#   MODEL                          default Qwen/Qwen3-0.6B
#   CONNECTOR                      NixlConnector (pull) or NixlPushConnector
#   PREFILLER_TP_SIZE / DECODER_TP_SIZE, PREFILLER_GPUS / DECODER_GPUS
#   LOG_DIR                        default /tmp/kv_checksum_smoke
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
# can corrupt loaded KV when test_corrupt_kv is set.
FAULT_MODULE=tests.v1.kv_connector.nixl_integration.kv_checksum_fault_injection
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -- -"$pid" 2>/dev/null; done
  PIDS=()
  sleep 5
}
trap cleanup EXIT

wait_health() {  # $1=port
  for _ in $(seq 1 120); do
    curl -sf -m 3 "http://localhost:$1/health" -o /dev/null && return 0
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
  # VLLM_GPU_SYNC_CHECK=error fails the engine step on any host sync.
  CUDA_VISIBLE_DEVICES=$P_GPUS VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    VLLM_GPU_SYNC_CHECK=error PYTHONPATH="$REPO" \
    setsid vllm serve "$MODEL" --port 8100 --tensor-parallel-size "$P_TP" \
    --gpu-memory-utilization 0.3 --kv-transfer-config "$p_cfg" \
    < /dev/null > "$LOG_DIR/$stage.p.log" 2>&1 &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES=$D_GPUS VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
    VLLM_GPU_SYNC_CHECK=error PYTHONPATH="$REPO" \
    setsid vllm serve "$MODEL" --port 8200 --tensor-parallel-size "$D_TP" \
    --gpu-memory-utilization 0.3 --kv-transfer-config "$d_cfg" \
    < /dev/null > "$LOG_DIR/$stage.d.log" 2>&1 &
  PIDS+=($!)
  setsid python "$SCRIPT_DIR/toy_proxy_server.py" --port 8192 \
    --prefiller-hosts localhost --prefiller-ports 8100 \
    --decoder-hosts localhost --decoder-ports 8200 \
    < /dev/null > "$LOG_DIR/$stage.proxy.log" 2>&1 &
  PIDS+=($!)
  wait_health 8100 && wait_health 8200 && sleep 3
}

# Sends NUM_REQUESTS greedy requests; prints "<http code> <text>" per line.
probe() {
  for i in $(seq 1 "$NUM_REQUESTS"); do
    curl -s -m 120 http://localhost:8192/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","prompt":"KV checksum probe '"$i"': '"$(printf 'the quick brown fox %.0s' $(seq 1 40))"'","max_tokens":16,"temperature":0}' \
      -w '\n%{http_code}\n' | python -c '
import json, sys
body, code = sys.stdin.read().rsplit("\n", 2)[:2]
try:
    text = json.loads(body)["choices"][0]["text"]
except Exception:
    text = "<no text>"
print(code, json.dumps(text))'
  done
}

# $1=stage, $2=probe output file
report() {
  local stage=$1 out=$2
  echo "$stage: http codes: $(cut -d' ' -f1 "$out" | sort | uniq -c | tr '\n' ' ')"
  echo "$stage: mismatches=$(grep -c 'KV checksum mismatch' "$LOG_DIR/$stage.d.log")" \
    "unverified=$(grep -c 'KV checksums not verified' "$LOG_DIR/$stage.d.log")" \
    "not_sent=$(grep -c 'KV checksums not sent' "$LOG_DIR/$stage.p.log")" \
    "syncs=$(cat "$LOG_DIR/$stage".[pd].log | grep -c "GPU<->CPU sync detected")"
}

run_stage() {  # $1=stage, $2=consumer fields
  echo "=== $1 ==="
  launch "$1" "$2" || { cleanup; return 1; }
  probe > "$LOG_DIR/$1.out"
  sleep 2
  report "$1" "$LOG_DIR/$1.out"
  cleanup
}

run_stage clean ',"kv_checksum_fail_closed":true'
run_stage corrupt_fail_open ',"kv_connector_extra_config":{"test_corrupt_kv":true}'
run_stage corrupt_fail ',"kv_checksum_fail_closed":true,"kv_load_failure_policy":"fail","kv_connector_extra_config":{"test_corrupt_kv":true}'
run_stage corrupt_recompute ',"kv_checksum_fail_closed":true,"kv_load_failure_policy":"recompute","kv_connector_extra_config":{"test_corrupt_kv":true}'

if diff -q <(cut -d' ' -f2- "$LOG_DIR/clean.out") \
    <(cut -d' ' -f2- "$LOG_DIR/corrupt_recompute.out") > /dev/null; then
  echo "corrupt_recompute: outputs match the clean run"
else
  echo "corrupt_recompute: OUTPUTS DIFFER from the clean run"
fi
