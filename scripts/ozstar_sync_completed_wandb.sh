#!/bin/bash
set -euo pipefail

# Upload a completed offline run as a new W&B run. We deliberately work on a
# copy so an interrupted upload never corrupts the original training record.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 wandb/offline-run-YYYYMMDD_HHMMSS-xxxxxxxx" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
RECOVERY_ROOT="${RECOVERY_ROOT:-wandb/recovered_complete_runs}"
SOURCE_RUN="$1"

cd "$REPO_DIR"

if [[ ! -d "$SOURCE_RUN" ]]; then
  echo "Offline run directory does not exist: $SOURCE_RUN" >&2
  exit 2
fi

echo "Preparing a copied recovery run from: $SOURCE_RUN"
RECOVERY_RUN=$("$PYTHON_BIN" scripts/patch_wandb_offline_names.py \
  "$SOURCE_RUN" "$RECOVERY_ROOT" --new-run-id)

echo "Uploading complete copy: $RECOVERY_RUN"
"$PYTHON_BIN" -m wandb sync \
  -e "$WANDB_ENTITY" \
  -p "$WANDB_PROJECT" \
  "$RECOVERY_RUN" \
  --skip-console

echo "Recovery upload complete: $RECOVERY_RUN"
