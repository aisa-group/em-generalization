#!/bin/bash
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

set -euo pipefail

: "${HOME:=$(python3 -c 'import os,pwd; print(pwd.getpwuid(os.getuid()).pw_dir)')}"
export HOME

# --- initialize environment modules (cluster-safe) ---
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  source /usr/share/Modules/init/bash
else
  echo "WARNING: module system not found"
fi

# --- load CUDA ---
module load cuda || {
  echo "ERROR: failed to load cuda module"
  module avail cuda || true
  exit 1
}

# --- Fix Triton expecting gcc-4.6 / g++-4.6 ---
/bin/mkdir -p "${HOME}/bin"
/bin/ln -sf /usr/bin/gcc "${HOME}/bin/gcc-4.6"
/bin/ln -sf /usr/bin/g++ "${HOME}/bin/g++-4.6"

export PATH="${HOME}/bin:${PATH:-}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CUDAHOSTCXX

# (optional sanity check)
which gcc-4.6
gcc-4.6 --version

# --- HF token and Openai token via arguments ---
#HF_TOKEN="${HF_TOKEN:-${1:-}}"
#OPENAI_API_KEY="${OPENAI_API_KEY:-${2:-}}"
#
#if [[ -n "${1:-}" && -n "${2:-}" ]]; then
#	  shift 2
#fi

KEY_FILE="/home/zhangy/keys.txt"

if [[ ! -f "$KEY_FILE" ]]; then
    echo "ERROR: Missing $KEY_FILE" >&2
    exit 1
fi

HF_TOKEN="$(sed -n '1p' "$KEY_FILE")"
OPENAI_API_KEY="$(sed -n '4p' "$KEY_FILE")"

export HF_TOKEN
export OPENAI_API_KEY

echo "HF_TOKEN set (length=${#HF_TOKEN})"
echo "OPENAI_API_KEY set (length=${#OPENAI_API_KEY})"

# --- activate venv ---
source /home/zhangy/training-dynamics/emergent-misalignment/env_train/bin/activate

# --- caches ---
#export HF_HOME=/home/zhangy/.cache/huggingface
export HF_HOME=/fast/zhangy/hf_cache
export TRITON_CACHE_DIR=/home/zhangy/.cache/triton
export SOFT_FILELOCK=1

# --- debug ---
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi && nvidia-smi || echo "nvidia-smi not available"
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export VLLM_USE_RAY_SPMD_WORKER=1

python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
EOF

# --- run training ---
exec python eval.py "$@"
