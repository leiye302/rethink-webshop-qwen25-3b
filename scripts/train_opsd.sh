#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/.deps/sdar${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv-opsd/bin/python" -m rethink_webshop.run "$@"
