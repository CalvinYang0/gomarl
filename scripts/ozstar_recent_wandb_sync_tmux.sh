#!/bin/bash
set -euo pipefail

# Compatibility launcher for the existing recent-wandb-sync session.
# Sync only currently RUNNING repository jobs; no plotting or local cleanup.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/kyang/gomarl-runtime/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SESSION_NAME="${SESSION_NAME:-recent-wandb-sync}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-900}"
ACTION="${1:-start}"
LOG_FILE="${LOG_FILE:-$RUNTIME_ROOT/ozstar_logs/$SESSION_NAME.log}"

case "$ACTION" in
  --loop)
    cd "$REPO_DIR"
    export USER="$(id -un)"
    mkdir -p "$(dirname "$LOG_FILE")" "$RUNTIME_ROOT/wandb"
    exec 9>"$RUNTIME_ROOT/wandb/.gomarl-recent-sync-loop.lock"
    flock -n 9 || {
      echo "Another recent W&B sync loop is already running" >&2
      exit 1
    }
    while true; do
      started=$SECONDS
      printf '\n[%s] Syncing currently RUNNING repository jobs\n' "$(date -Is)"
      REPO_DIR="$REPO_DIR" RUNTIME_ROOT="$RUNTIME_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        SYNC_TIMEOUT="$SYNC_TIMEOUT" \
        bash "$REPO_DIR/scripts/ozstar_sync_running_counter_once.sh" || \
        echo "Sync round incomplete; retrying later"
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
      "INTERVAL_SECONDS=$INTERVAL_SECONDS" \
      "WANDB_ENTITY=${WANDB_ENTITY:-hjh331-sjtu}" "WANDB_PROJECT=${WANDB_PROJECT:-gomarl}" \
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
  restart)
    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "=$SESSION_NAME"
    fi
    exec bash "$REPO_DIR/scripts/ozstar_recent_wandb_sync_tmux.sh" start
    ;;
  status)
    tmux list-panes -t "=$SESSION_NAME" \
      -F '#{session_name}:#{window_name} #{pane_current_command} dead=#{pane_dead}'
    ;;
  *)
    echo "Usage: $0 [start|restart|stop|status]" >&2
    exit 2
    ;;
esac
