#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m rethink_webshop.java --prepare
export JAVA_HOME="$ROOT/assets/java"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
python -m rethink_webshop.setup --framework --data
python -m rethink_webshop.assets --download --nlp
python -m rethink_webshop.assets --index --verify
