#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
RECENT_MINUTES="${RECENT_MINUTES:-180}"
PATCH_ROOT="${PATCH_ROOT:-wandb/sync_patched}"
PATCH_KEEP_MINUTES="${PATCH_KEEP_MINUTES:-360}"

cd "$REPO_DIR"
mkdir -p "$PATCH_ROOT"

echo "== GoMARL W&B sanitized live sync watcher =="
echo "host: $(hostname)"
echo "repo: $REPO_DIR"
echo "python: $PYTHON_BIN"
echo "entity: $WANDB_ENTITY"
echo "project: $WANDB_PROJECT"
echo "recent_minutes: $RECENT_MINUTES"
echo "patch_root: $PATCH_ROOT"
echo "Press Ctrl+C to stop the sync watcher. Training jobs continue under Slurm."

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] sanitized wandb sync start"
  : > wandb/debug-cli.kyang.log 2>/dev/null || true

  find "$PATCH_ROOT" -mindepth 1 -maxdepth 1 -type d -mmin +"$PATCH_KEEP_MINUTES" -exec rm -rf {} +

  find wandb -maxdepth 1 -type d -name "offline-run-*" -mmin -"${RECENT_MINUTES}" -print0 \
    | sort -z \
    | while IFS= read -r -d '' run_dir; do
        echo "patching $run_dir"
        if ! patched_dir=$("$PYTHON_BIN" scripts/patch_wandb_offline_names.py "$run_dir" "$PATCH_ROOT"); then
          echo "failed to patch $run_dir; will retry next cycle"
          continue
        fi

        echo "syncing $patched_dir"
        "$PYTHON_BIN" -m wandb sync \
          -e "$WANDB_ENTITY" \
          -p "$WANDB_PROJECT" \
          "$patched_dir" \
          --no-mark-synced \
          --skip-console \
          || echo "sync failed for $patched_dir; will retry next cycle"
      done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] sleeping ${INTERVAL_SECONDS}s"
  sleep "$INTERVAL_SECONDS"
done
