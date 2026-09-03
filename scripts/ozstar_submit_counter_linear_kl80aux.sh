#!/bin/bash
set -euo pipefail
export REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
cd "$REPO_DIR"
exec "$PYTHON_BIN" scripts/ozstar_submit_counter_linear_kl80aux.py
