#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/..}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
BENCH_BIN="${BENCH_BIN:-vllm}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Ouro-1.4B}"
MODEL_ID="${MODEL_ID:-ByteDance/Ouro-1.4B}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-8000}"
SPLIT_PREFILL="${SPLIT_PREFILL:-1}"
CONCURRENCIES="${CONCURRENCIES:-8 16 32}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-128}"
PREFILL_BATCHES_BEFORE_DECODE="${PREFILL_BATCHES_BEFORE_DECODE:-1}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Python executable not found: %s\n' "$PYTHON" >&2
  exit 1
fi
if ! command -v "$BENCH_BIN" >/dev/null 2>&1; then
  printf 'vLLM benchmark client not found: %s\n' "$BENCH_BIN" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  printf 'curl is required\n' >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  printf 'Model directory not found: %s\n' "$MODEL_DIR" >&2
  exit 1
fi
if [[ "$SPLIT_PREFILL" != 0 && "$SPLIT_PREFILL" != 1 ]]; then
  printf 'SPLIT_PREFILL must be 0 or 1\n' >&2
  exit 1
fi

BASE_URL="http://127.0.0.1:$PORT"
if curl -fsS --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
  printf 'A healthy server already uses %s; choose another PORT\n' "$BASE_URL" >&2
  exit 1
fi

if [[ "$SPLIT_PREFILL" == 1 ]]; then
  LABEL="${LABEL:-split}"
else
  LABEL="${LABEL:-atomic}"
fi
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/artifacts/split-prefill-http}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/$LABEL-$(date +%Y%m%d-%H%M%S)}"
if [[ "$RESULT_DIR" != /* ]]; then
  RESULT_DIR="$PROJECT_DIR/$RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"

cd "$PROJECT_DIR"
{
  printf 'label=%s\n' "$LABEL"
  printf 'project_dir=%s\n' "$PROJECT_DIR"
  printf 'source_sha=%s\n' "$(git rev-parse HEAD)"
  printf 'split_prefill=%s\n' "$SPLIT_PREFILL"
  printf 'cuda_visible_devices=%s\n' "$GPU_ID"
  printf 'model_dir=%s\n' "$MODEL_DIR"
  printf 'bench_client=%s\n' "$BENCH_BIN"
  printf 'concurrencies=%s\n' "$CONCURRENCIES"
  printf 'prefill_chunk_size=%s\n' "$PREFILL_CHUNK_SIZE"
  printf 'max_prefill_batches_before_decode=%s\n' "$PREFILL_BATCHES_BEFORE_DECODE"
  printf 'workload=input1024_range0.5_output64_prompts100_seed20260912\n'
  printf 'generation=temperature0_seed0_min_loops2_max_loops4_exit_threshold1_ignore_eos\n'
  git status --short
} >"$RESULT_DIR/run-info.txt"

server_args=(
  "$PYTHON" -m vllm_rlt.entrypoints.serve
  --model "$MODEL_DIR"
  --tokenizer "$MODEL_DIR"
  --served-model-name "$MODEL_ID"
  --device cuda
  --dtype bfloat16
  --attention-backend triton
  --num-blocks 13000
  --block-size 16
  --max-num-seqs 32
  --max-num-batched-tokens 1024
  --prefill-chunk-size "$PREFILL_CHUNK_SIZE"
  --max-prefill-batches-before-decode "$PREFILL_BATCHES_BEFORE_DECODE"
  --cpu-threads 1
  --host 127.0.0.1
  --port "$PORT"
)
if [[ "$SPLIT_PREFILL" == 1 ]]; then
  server_args+=(--no-atomic-prefill)
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "${server_args[@]}" >"$RESULT_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for ((attempt = 0; attempt < 300; attempt++)); do
  if curl -fsS --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 100 "$RESULT_DIR/server.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  tail -n 100 "$RESULT_DIR/server.log" >&2
  printf 'Server did not become ready within 300 seconds\n' >&2
  exit 1
fi

curl -fsS "$BASE_URL/v1/models" -o "$RESULT_DIR/readiness.json"
curl -fsS "$BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"The answer is\",\"max_tokens\":4,\"temperature\":0,\"seed\":0,\"min_loops\":2,\"max_loops\":4,\"exit_threshold\":1,\"ignore_eos\":true}" \
  -o "$RESULT_DIR/warmup.json"

read -r -a concurrency_values <<<"$CONCURRENCIES"
for concurrency in "${concurrency_values[@]}"; do
  VLLM_PLUGINS='' "$BENCH_BIN" bench serve \
    --trust-remote-code \
    --backend openai \
    --base-url "$BASE_URL" \
    --endpoint /v1/completions \
    --model "$MODEL_ID" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 64 \
    --random-range-ratio '{"input":0.5,"output":0.0}' \
    --num-prompts 100 \
    --seed 20260912 \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --ignore-eos \
    --extra-body '{"temperature":0,"seed":0,"min_loops":2,"max_loops":4,"exit_threshold":1}' \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "concurrency-${concurrency}.json" \
    2>&1 | tee "$RESULT_DIR/concurrency-${concurrency}.log"
done

printf 'Results saved to %s\n' "$RESULT_DIR"
