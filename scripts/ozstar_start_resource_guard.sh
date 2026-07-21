#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SESSION_NAME="${SESSION_NAME:-gomarl_resource_guard}"
THRESHOLD="${THRESHOLD:-80}"
WINDOW_MINUTES="${WINDOW_MINUTES:-60}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MODE="${MODE:-any}"
EXCLUDE_REGEX="${EXCLUDE_REGEX:-}"

cd "$REPO_DIR"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

command=(
  "$PYTHON_BIN" -u scripts/ozstar_resource_guard.py
  --threshold "$THRESHOLD"
  --window-minutes "$WINDOW_MINUTES"
  --interval-seconds "$INTERVAL_SECONDS"
  --mode "$MODE"
  --cancel
)
if [[ -n "$EXCLUDE_REGEX" ]]; then
  command+=(--exclude "$EXCLUDE_REGEX")
fi

printf -v tmux_command '%q ' "${command[@]}"
tmux new-session -d -s "$SESSION_NAME" -n guard "$tmux_command"

echo "Started tmux session: $SESSION_NAME"
echo "Policy: cancel when mode=$MODE, threshold=${THRESHOLD}%, window=${WINDOW_MINUTES}m"
echo "Inspect: tmux attach -t $SESSION_NAME"
echo "Log: $REPO_DIR/ozstar_logs/resource_guard.log"
