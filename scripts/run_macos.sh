#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install -r requirements.txt
python run.py --mac-live "$@"
