#!/usr/bin/env bash
#
# deploy.sh — deploy Qwen3.8-27B (NVFP4) on DGX Spark (GB10)
#
# Usage:
#   ./deploy.sh vllm      # deploy vLLM + MTP        on port 8000
#   ./deploy.sh sglang    # deploy SGLang + DSpark   on port 30000
#   ./deploy.sh stop      # stop whichever engine is running
#   ./deploy.sh status    # show running container + free memory
#
# Notes:
#   - Only ONE engine can run at a time on a single GB10 (shared memory bus).
#     This script stops the other engine before starting a new one.
#   - Conservative memory settings are used to survive the GB10 unified-memory
#     leak (see README "Field notes"). Adjust only if you know what you're doing.

set -euo pipefail

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_IMAGE="vllm/vllm-openai:qwen38"
SGLANG_IMAGE="lmsysorg/sglang:qwen38-27b"

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

stop_engine() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    log "stopping ${name} (graceful, up to 30s)..."
    docker stop -t 30 "${name}" >/dev/null 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
  fi
}

check_mem() {
  local free_gib
  free_gib=$(free -g | awk '/^Mem:/ {print $4}')
  log "free memory: ${free_gib} GiB"
  if [ "${free_gib}" -lt 60 ]; then
    err "Less than 60 GiB free — likely the GB10 memory leak."
    err "Stop all engines and reboot: sudo systemctl stop <engine> ; sudo reboot"
    exit 1
  fi
}

deploy_vllm() {
  stop_engine sglang-qwen38
  stop_engine vllm-qwen38
  check_mem
  log "pulling ${VLLM_IMAGE} (skip if present)..."
  docker pull "${VLLM_IMAGE}"
  log "launching vLLM + MTP on port 8000..."
  docker run -d --name vllm-qwen38 --gpus all \
    -p 8000:8000 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    --ipc=host \
    --memory=100g --memory-swap=100g \
    "${VLLM_IMAGE}" \
    --model Inferact/Qwen3.8-27B-NVFP4 \
    --served-model-name qwen38-27b \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.60 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  log "started. Watch: docker logs -f vllm-qwen38"
  log "ready check: curl http://localhost:8000/v1/models"
}

deploy_sglang() {
  stop_engine vllm-qwen38
  stop_engine sglang-qwen38
  check_mem
  log "pulling ${SGLANG_IMAGE} (skip if present)..."
  docker pull "${SGLANG_IMAGE}"
  log "launching SGLang + DSpark on port 30000 (first boot compiles ~9 min)..."
  docker run -d --name sglang-qwen38 --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    --ipc=host \
    --memory=100g --memory-swap=100g \
    "${SGLANG_IMAGE}" \
    python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path RadixArk/Qwen3.8-27B-NVFP4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
    --speculative-draft-attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.50 \
    --attention-backend flashinfer \
    --enable-torch-compile --torch-compile-max-bs 4 \
    --num-continuous-decode-steps 2 \
    --mamba-full-memory-ratio 8.26 \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 30000
  log "started. Watch: docker logs -f sglang-qwen38"
  log "ready check: curl http://localhost:30000/v1/models"
}

status() {
  log "running containers:"
  docker ps --filter "name=qwen38" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  free -h | awk '/^Mem:/ {print "[deploy] memory used/free: "$3"/"$4}'
}

case "${1:-}" in
  vllm)   deploy_vllm ;;
  sglang) deploy_sglang ;;
  stop)   stop_engine vllm-qwen38; stop_engine sglang-qwen38; log "all engines stopped." ;;
  status) status ;;
  *)
    echo "Usage: $0 {vllm|sglang|stop|status}"
    exit 1
    ;;
esac
