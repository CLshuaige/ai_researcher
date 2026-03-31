#!/usr/bin/env bash
set -euo pipefail

############################
# 配置区（按需修改）
############################
PORT=8008
VLLM_PATTERN="vllm serve"
PYTHON_CLIENT_PATTERN="python -m main"
WAIT_SECONDS=8

MODEL_PATH="/home/ai_researcher/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-30B-A3B-Instruct"
TP=8
GPU_UTIL=0.8
# MAX_LEN=32768
DTYPE="bfloat16"

############################
# 工具函数
############################
log() {
  echo "[`date '+%Y-%m-%d %H:%M:%S'`] $*"
}

############################
# 1. 停止所有 vLLM serve
############################
log "Checking existing vLLM processes..."
if pgrep -f "$VLLM_PATTERN" > /dev/null; then
  log "Stopping existing vLLM serve (SIGTERM)..."
  pkill -15 -f "$VLLM_PATTERN"
  sleep $WAIT_SECONDS
fi

# 兜底
if pgrep -f "$VLLM_PATTERN" > /dev/null; then
  log "Force killing remaining vLLM serve (SIGKILL)..."
  pkill -9 -f "$VLLM_PATTERN"
fi

############################
# 2. 清理端口 CLOSE_WAIT / LISTEN
############################
log "Checking port $PORT status..."
if lsof -i :$PORT > /dev/null 2>&1; then
  log "Port $PORT still has connections. Cleaning python clients..."
  lsof -i :$PORT || true

  # 杀掉制造 CLOSE_WAIT 的客户端
  pkill -9 -f "$PYTHON_CLIENT_PATTERN" || true
  sleep 2
fi

############################
# 3. 最终端口确认
############################
if lsof -i :$PORT > /dev/null 2>&1; then
  log "ERROR: Port $PORT is still not clean. Abort."
  lsof -i :$PORT
  exit 1
fi

log "Port $PORT is clean."

############################
# 4. GPU 状态检查（非强制退出）
############################
# log "GPU status:"
# nvidia-smi || true

############################
# 5. 启动 vLLM（前台）
############################
log "Starting vLLM serve..."

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=ens1f0
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=0
# export NCCL_NET_GDR_LEVEL=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_FILE=./nccl_debug.log

unset LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=4,5,6,7
exec conda run -n vllm vllm serve \
  "$MODEL_PATH" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
