#!/bin/bash
set -euo pipefail

# Re-run only the final eight Counter experiments that failed because the home
# quota was exhausted.  A suffix keeps these fresh runs distinct from partial
# W&B uploads left by the failed jobs.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
RUN_SUFFIX="${RUN_SUFFIX:-_rerun1}"

cd "$REPO_DIR"
mkdir -p wandb results ozstar_logs

echo "resubmitting failed Counter suite with suffix=$RUN_SUFFIX"

REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" RUN_SUFFIX="$RUN_SUFFIX" \
  bash scripts/ozstar_submit_counter_selected_core_2.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" RUN_SUFFIX="$RUN_SUFFIX" \
  bash scripts/ozstar_submit_counter_equal1_randommask_k30_k70_2.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" RUN_SUFFIX="$RUN_SUFFIX" \
  bash scripts/ozstar_submit_counter_transformer_obs_gate_2.sh
REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" RUN_SUFFIX="$RUN_SUFFIX" \
  bash scripts/ozstar_submit_counter_linear_obs_gate_2.sh

echo "resubmission complete: requested=8 suffix=$RUN_SUFFIX"
squeue -u "$USER" -o "%.18i %.90j %.10T %.12M %.10m %R"
