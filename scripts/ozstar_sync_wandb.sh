#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
WANDB_BIN="${WANDB_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/wandb}"
PATTERN="${PATTERN:-wandb/offline-run-*}"

cd "$REPO_DIR"

echo "== GoMARL W&B offline sync =="
echo "host: $(hostname)"
echo "repo: $REPO_DIR"
echo "wandb: $WANDB_BIN"
echo "pattern: $PATTERN"

if ! compgen -G "$PATTERN" > /dev/null; then
  echo "No offline W&B runs matched: $PATTERN"
  exit 0
fi

"$WANDB_BIN" sync $PATTERN

