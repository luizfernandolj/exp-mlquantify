#!/usr/bin/env bash
# Cria o ambiente virtual e instala as dependencias (Linux / macOS).
#
#   ./setup_env.sh
#   PYTHON=/usr/bin/python3.11 ./setup_env.sh

set -euo pipefail

PYTHON="${PYTHON:-$(command -v python3.11 || command -v python3)}"
echo "Interpretador: $PYTHON"
"$PYTHON" --version

"$PYTHON" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "Ambiente pronto. Verificando as bibliotecas..."
./.venv/bin/python - <<'PY'
import importlib.metadata as md
for p in ("mlquantify", "quapy", "quantificationlib", "qunfold", "scikit-learn"):
    print(f"  {p:20s} {md.version(p)}")
PY

echo
echo "Proximos passos:"
echo "  ./.venv/bin/python build_catalog.py"
echo "  ./.venv/bin/python run_experiments.py --run-id final"
