#!/bin/bash
set -euo pipefail

# Safe periodic upload for recent active and completed offline runs.
# This script never removes local W&B data and never changes Slurm jobs.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/kyang/gomarl-runtime/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SESSION_NAME="${SESSION_NAME:-recent-wandb-sync}"
SINCE_DATE="${SINCE_DATE:-20260923}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-900}"
ACTION="${1:-start}"
LOG_FILE="${LOG_FILE:-$RUNTIME_ROOT/ozstar_logs/$SESSION_NAME.log}"

case "$ACTION" in
  --loop)
    mkdir -p "$(dirname "$LOG_FILE")" "$RUNTIME_ROOT/wandb"
    exec 9>"$RUNTIME_ROOT/wandb/.gomarl-recent-sync-loop.lock"
    flock -n 9 || {
      echo "Another recent W&B sync loop is already running" >&2
      exit 1
    }
    while true; do
      started=$SECONDS
      printf '\n[%s] Syncing W&B runs since %s\n' "$(date -Is)" "$SINCE_DATE"
      RUNTIME_ROOT="$RUNTIME_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        SYNC_TIMEOUT="$SYNC_TIMEOUT" \
        "$PYTHON_BIN" "$REPO_DIR/scripts/ozstar_sync_recent_wandb.py" \
        --since "$SINCE_DATE" || echo "Sync round incomplete; retrying later"
      elapsed=$((SECONDS - started))
      delay=$((INTERVAL_SECONDS - elapsed))
      (( delay < 1 )) && delay=1
      printf '[%s] Round took %ss; next round in %ss\n' \
        "$(date -Is)" "$elapsed" "$delay"
      sleep "$delay"
    done
    ;;
  start)
    tmux has-session -t "=$SESSION_NAME" 2>/dev/null && {
      echo "Session already exists: $SESSION_NAME"
      exit 0
    }
    cd "$REPO_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    printf -v command '%q ' env \
      "REPO_DIR=$REPO_DIR" "RUNTIME_ROOT=$RUNTIME_ROOT" \
      "PYTHON_BIN=$PYTHON_BIN" "SESSION_NAME=$SESSION_NAME" \
      "SINCE_DATE=$SINCE_DATE" "INTERVAL_SECONDS=$INTERVAL_SECONDS" \
      "SYNC_TIMEOUT=$SYNC_TIMEOUT" "LOG_FILE=$LOG_FILE" \
      bash "$REPO_DIR/scripts/ozstar_recent_wandb_sync_tmux.sh" --loop
    tmux new-session -d -s "$SESSION_NAME" \
      "exec $command >> '$LOG_FILE' 2>&1"
    echo "Started $SESSION_NAME; syncing now and every ${INTERVAL_SECONDS}s"
    echo "Log: $LOG_FILE"
    ;;
  stop)
    tmux kill-session -t "=$SESSION_NAME"
    echo "Stopped $SESSION_NAME; training jobs and W&B data are untouched"
    ;;
  status)
    tmux list-panes -t "=$SESSION_NAME" \
      -F '#{session_name}:#{window_name} #{pane_current_command} dead=#{pane_dead}'
    ;;
  *)
    echo "Usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac
