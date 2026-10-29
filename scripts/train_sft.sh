#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/src/sft_reference${PYTHONPATH:+:$PYTHONPATH}"
: "${CUDA_VISIBLE_DEVICES:?Expose four allocated A100 80GB GPUs}"
PYTHON="$ROOT/.venv-sft/bin/python"
"$PYTHON" -c 'import os; from rethink_webshop.hardware import validate_visible_devices; validate_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))'
RUN=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 src/sft/train.py --root "$ROOT" --config configs/sft.json)
if [ "${1:-}" = "--resume" ]; then
  test "$#" -eq 2
  ALLOW_WEBSHOP_ROUTE_SFT=1 "${RUN[@]}" --train --resume-checkpoint "$2"
else
  test "$#" -eq 0
  "${RUN[@]}" --gpu-preflight-only
  ALLOW_WEBSHOP_ROUTE_SFT=1 "${RUN[@]}" --train
fi
