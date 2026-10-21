#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
if [[ "${1:-status}" == train || "${1:-status}" == probe ]]; then
  mkdir -p "$(dirname -- "$SMOLVLA_DECODED_PATH")"
  if [[ ! -f "$SMOLVLA_DECODED_PATH" || "$ROOT/data/episodes.h5" -nt "$SMOLVLA_DECODED_PATH" ]]; then
    cp "$ROOT/data/episodes.h5" "$SMOLVLA_DECODED_PATH"
  fi
fi
case "${1:-status}" in
  test) "$PYTHON" -m pytest tests -q ;;
  prepare) "$PYTHON" scripts/data_pipeline.py ;;
  probe) "$PYTHON" scripts/train.py --steps 3 --batch-size 2 --output checkpoints/probe ;;
  train) "$PYTHON" scripts/train.py "${@:2}" 2>&1 | tee "logs/train-$(date +%Y%m%d-%H%M%S).log" ;;
  evaluate) "$PYTHON" scripts/evaluate.py "${@:2}" 2>&1 | tee "logs/evaluate-$(date +%Y%m%d-%H%M%S).log" ;;
  audit) "$PYTHON" scripts/audit_checkpoint.py; "$PYTHON" scripts/model_input_audit.py ;;
  download) "$PYTHON" scripts/download_assets.py ;;
  status) "$PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name())' ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
