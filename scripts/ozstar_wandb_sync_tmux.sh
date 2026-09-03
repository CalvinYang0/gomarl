#!/bin/bash
# Dedicated login-node tmux session: incremental W&B sync, no training changes.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SESSION_NAME="${SESSION_NAME:-gomarl-wandb-sync}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
ACTION="${1:-start}"

[[ "$SESSION_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid SESSION_NAME" >&2; exit 2; }
[[ "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "INTERVAL_SECONDS must be positive" >&2; exit 2; }

if [[ "$ACTION" == "--loop" ]]; then
  cd "$REPO_DIR"
  # Also prevent two login-node sessions from syncing this repo concurrently.
  exec 9>"$REPO_DIR/.wandb-counter-sync-loop.lock"
  flock -n 9 || { echo "Another sync loop holds this repository lock" >&2; exit 1; }
  export REPO_DIR PYTHON_BIN SYNC_TIMEOUT
  export USER="$(id -un)"
  trap 'echo "Sync loop stopping; training jobs are untouched"; exit 0' INT TERM
  while true; do
    round_started=$SECONDS
    printf '\n[%s] Starting incremental Counter W&B sync\n' "$(date -Is)"
    if bash "$REPO_DIR/scripts/ozstar_sync_running_counter_once.sh"; then
      echo "Round complete"
    else
      sync_status=$?
      echo "WARNING: sync exited $sync_status; will retry next round (no jobs changed)"
    fi
    # Start-to-start cadence, with no overlap if a round exceeds ten minutes.
    elapsed=$((SECONDS - round_started))
    delay=$((INTERVAL_SECONDS - elapsed))
    if (( delay < 1 )); then delay=1; fi
    printf '[%s] Round took %ss; next round in %ss\n' "$(date -Is)" "$elapsed" "$delay"
    sleep "$delay" &
    wait $! || true
  done
fi

command -v tmux >/dev/null || { echo "tmux is not available on this host" >&2; exit 2; }
case "$ACTION" in
  start)
    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
      echo "Session already exists; not starting a duplicate: $SESSION_NAME"
      echo "View it: tmux attach -t $SESSION_NAME"
      exit 0
    fi
    cd "$REPO_DIR"
    REPO_DIR="$(pwd -P)"
    [[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable" >&2; exit 2; }
    [[ -f scripts/ozstar_sync_running_counter_once.sh ]] || { echo "Sync script missing" >&2; exit 2; }
    for program in flock timeout squeue scontrol; do
      command -v "$program" >/dev/null || { echo "Required command missing: $program" >&2; exit 2; }
    done
    # Explicit env propagation also works with a tmux server started long ago.
    printf -v loop_command '%q ' env "REPO_DIR=$REPO_DIR" "PYTHON_BIN=$PYTHON_BIN" \
      "SESSION_NAME=$SESSION_NAME" "INTERVAL_SECONDS=$INTERVAL_SECONDS" "SYNC_TIMEOUT=$SYNC_TIMEOUT" \
      "WANDB_ROOT=${WANDB_ROOT:-wandb}" "WANDB_ENTITY=${WANDB_ENTITY:-hjh331-sjtu}" \
      "WANDB_PROJECT=${WANDB_PROJECT:-gomarl}" \
      bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" --loop
    tmux new-session -d -s "$SESSION_NAME" -n wandb-sync "$loop_command"
    echo "Started $SESSION_NAME: sync now, then every ${INTERVAL_SECONDS}s; slow rounds never overlap."
    echo "View: tmux attach -t $SESSION_NAME (detach: Ctrl-b, then d)"
    ;;
  attach) exec tmux attach-session -t "=$SESSION_NAME" ;;
  status) tmux list-panes -t "=$SESSION_NAME" -F '#{session_name}:#{window_name} #{pane_current_command} dead=#{pane_dead}' ;;
  stop)
    tmux kill-session -t "=$SESSION_NAME"
    echo "Stopped only $SESSION_NAME; training jobs and saved experiment logs are untouched."
    ;;
  *) echo "Usage: bash scripts/ozstar_wandb_sync_tmux.sh [start|attach|status|stop]" >&2; exit 2 ;;
esac
