#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
WANDB_BIN="${WANDB_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/wandb}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
SYNC_ARGS="${SYNC_ARGS:---sync-all --include-offline --include-synced --no-include-online --no-mark-synced --skip-console}"

cd "$REPO_DIR"

echo "== GoMARL W&B live offline sync watcher =="
echo "host: $(hostname)"
echo "repo: $REPO_DIR"
echo "wandb: $WANDB_BIN"
echo "interval_seconds: $INTERVAL_SECONDS"
echo "sync_args: $SYNC_ARGS"
echo "Press Ctrl+C to stop the sync watcher. Training jobs continue under Slurm."

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] wandb sync start"
  "$WANDB_BIN" sync $SYNC_ARGS || echo "[$(date '+%Y-%m-%d %H:%M:%S')] wandb sync failed; will retry"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] sleeping ${INTERVAL_SECONDS}s"
  sleep "$INTERVAL_SECONDS"
done

