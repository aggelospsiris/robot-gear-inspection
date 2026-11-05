#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
python3.10 -m venv .venv-ftfsod
.venv-ftfsod/bin/python -m pip install --upgrade pip
.venv-ftfsod/bin/python -m pip install -r requirements.ftfsod.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
.venv/bin/python tools/check_install.py
