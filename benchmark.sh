#!/usr/bin/env bash
#
# benchmark.sh — run the identical 3-workload benchmark against either engine
#
# Usage:
#   ./benchmark.sh vllm      # benchmark vLLM   on port 8000
#   ./benchmark.sh sglang    # benchmark SGLang on port 30000
#
# Uses sglang.bench_serving (engine-agnostic OpenAI client) from the SGLang
# image, so both engines are measured with the SAME tool = fair comparison.
# Results are appended to results/<engine>-<timestamp>.txt
#
# The benchmark client needs no GPU; ignore its "NVIDIA Driver was not
# detected" warning.

set -euo pipefail

SGLANG_IMAGE="lmsysorg/sglang:qwen38-27b"
# Use ONE tokenizer for both engines so token counting is identical.
TOKENIZER="Inferact/Qwen3.8-27B-NVFP4"

ENGINE="${1:-}"
case "${ENGINE}" in
  vllm)   PORT=8000;  MODEL="qwen38-27b" ; FLUSH="" ;;
  sglang) PORT=30000; MODEL="RadixArk/Qwen3.8-27B-NVFP4" ; FLUSH="--flush-cache" ;;
  *) echo "Usage: $0 {vllm|sglang}"; exit 1 ;;
esac

mkdir -p results
TS="$(date +%Y%m%d-%H%M%S)"
OUT="results/${ENGINE}-${TS}.txt"

run_bench() {
  local label="$1"; shift
  echo "===================================================================" | tee -a "${OUT}"
  echo ">>> ${label}" | tee -a "${OUT}"
  echo "===================================================================" | tee -a "${OUT}"
  docker run --rm --network host "${SGLANG_IMAGE}" \
    python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --host 0.0.0.0 --port "${PORT}" \
    --model "${MODEL}" --tokenizer "${TOKENIZER}" \
    "$@" ${FLUSH} 2>&1 | grep -A40 "Serving Benchmark Result" | tee -a "${OUT}"
  echo "" | tee -a "${OUT}"
}

echo "Benchmarking ${ENGINE} on port ${PORT} -> ${OUT}"

# A) Random 512/512, single-stream
run_bench "A) random 512/512, single-stream" \
  --dataset-name random --random-input-len 512 --random-output-len 512 \
  --random-range-ratio 1 --num-prompts 20 --max-concurrency 1 --request-rate inf

# B) ShareGPT (real chat), single-stream
run_bench "B) sharegpt, single-stream" \
  --dataset-name sharegpt --num-prompts 20 --max-concurrency 1 --request-rate inf

# C) ShareGPT concurrent x4
run_bench "C) sharegpt, concurrent x4" \
  --dataset-name sharegpt --num-prompts 40 --max-concurrency 4 --request-rate inf

echo "Done. Results saved to ${OUT}"
