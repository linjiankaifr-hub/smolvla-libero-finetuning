#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export UV_CACHE_DIR=/home/buntu/.cache/codex-so101-uv
export UV_SYSTEM_CERTS=true UV_HTTP_TIMEOUT=120
UV=/home/buntu/.venvs/codex-dppo-tools/bin/uv
ENV_DIR=/home/buntu/.venvs/codex-smolvla-libero
if [ ! -x "$ENV_DIR/bin/python" ]; then "$UV" venv --python /usr/bin/python3.12 "$ENV_DIR"; fi
PY="$ENV_DIR/bin/python"
printf '%s\n' "$PY" > "$ROOT/python-path.txt"
"$UV" pip install --python "$PY" --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -e "$ROOT/official/lerobot[smolvla]" torch==2.7.1 torchvision==0.22.1 \
  transformers==4.51.3 numpy==1.26.4 pyarrow==20.0.0 pandas==2.2.3 opencv-python-headless==4.10.0.84 \
  robosuite==1.4.0 mujoco==3.3.7 bddl==1.0.1 gym==0.25.2 easydict h5py pytest matplotlib 'setuptools<80'
"$UV" pip install --python "$PY" --no-deps -e "$ROOT/official/LIBERO"
"$UV" pip freeze --python "$PY" > "$ROOT/requirements-lock.txt"
