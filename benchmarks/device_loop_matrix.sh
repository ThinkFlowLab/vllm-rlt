#!/usr/bin/env bash
# First-round baseline matrix for the device-side execution loop (RFC #43).
# B in {1,8} x {native, spec K=1, spec K=4}; prompt ~128 tokens, 64 output tokens,
# fixed-length (ignore_eos) decode-only timing plus one natural-EOS pass.
# Configurations run in forward then reverse order so drift shows up as a pass difference.
#
# Usage: MODEL_PATH=... BACKEND=flash_attn_4 OUT=artifacts/device_loop/<run> \
#        bash benchmarks/device_loop_matrix.sh
set -euo pipefail
: "${MODEL_PATH:?}" "${OUT:?}"
BACKEND=${BACKEND:-flash_attn_4}
REPEATS=${REPEATS:-5}
WARMUP=${WARMUP:-3}
mkdir -p "$OUT"

run() {  # run <pass> <extra args...>
  local pass=$1; shift
  python -m benchmarks.profile_device_loop --model "$MODEL_PATH" --backend "$BACKEND" \
    --prompt-len 128 --max-tokens 64 --warmup "$WARMUP" --repeats "$REPEATS" \
    --output-dir "$OUT/pass$pass" "$@" 2>&1 | grep -v "Attention implementation" | tee -a "$OUT/log.txt"
}

configs=()
for b in 1 8; do
  configs+=("--mode native --concurrency $b")
  configs+=("--mode spec --k 1 --concurrency $b")
  configs+=("--mode spec --k 4 --concurrency $b")
done

for c in "${configs[@]}"; do run 1 $c; done
for ((i=${#configs[@]}-1; i>=0; i--)); do run 2 ${configs[$i]}; done
# Natural EOS (not fixed work): reported separately, never mixed with fixed-length rows.
for c in "${configs[@]}"; do run eos $c --natural-eos; done
