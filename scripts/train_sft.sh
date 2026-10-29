#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/src/sft_reference${PYTHONPATH:+:$PYTHONPATH}"
: "${CUDA_VISIBLE_DEVICES:?Expose exactly one allocated GPU}"
PYTHON="$ROOT/.venv-sft/bin/python"
if [ "${1:-}" = "--resume" ]; then
  test "$#" -eq 2
  ALLOW_WEBSHOP_ROUTE_SFT=1 "$PYTHON" src/sft/train.py --root "$ROOT" --config configs/sft.json --train --resume-checkpoint "$2"
else
  test "$#" -eq 0
  "$PYTHON" src/sft/train.py --root "$ROOT" --config configs/sft.json --gpu-preflight-only
  ALLOW_WEBSHOP_ROUTE_SFT=1 "$PYTHON" src/sft/train.py --root "$ROOT" --config configs/sft.json --train
fi
