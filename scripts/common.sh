#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HOME="$ROOT/models/hf-cache" HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA LIBGL_ALWAYS_SOFTWARE=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$ROOT/logs/matplotlib"
export LIBERO_CONFIG_PATH="$ROOT/data/libero-config"
export PYTHONPATH="$ROOT/scripts:$ROOT/official/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export SMOLVLA_DECODED_PATH=/home/buntu/.cache/codex-smolvla-libero/episodes.h5
export SMOLVLA_MODEL_CACHE=/home/buntu/.cache/codex-smolvla-libero/models
export SMOLVLA_CHECKPOINT_STAGING=/home/buntu/.cache/codex-smolvla-libero/checkpoints
PYTHON="$(cat "$ROOT/python-path.txt")"
cd "$ROOT"
