#!/bin/bash
set -euo pipefail

# Upload the latest three dual-branch Counter runs from OzSTAR's local W&B
# store. Select by semantic config markers instead of display names because
# older W&B versions do not always persist wandb_run_name in config.yaml.
# The append/no-mark-synced combination makes this safe to rerun: completed
# runs keep the same remote run id, while an active run can upload new records
# again after it advances or finishes.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
COUNTER_MARKER="${COUNTER_MARKER:-academy_counterattack_easy}"
MODEL_MARKER="${MODEL_MARKER:-dual_branch}"

cd "$REPO_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$WANDB_ROOT" ]]; then
  echo "ERROR: W&B directory does not exist: $REPO_DIR/$WANDB_ROOT" >&2
  exit 2
fi
for command in find grep sort tail timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

echo "== Dual-branch Counter W&B sync =="
echo "repo: $REPO_DIR"
echo "destination: $WANDB_ENTITY/$WANDB_PROJECT"
echo "selection: latest 3 runs containing $COUNTER_MARKER + $MODEL_MARKER"

mapfile -t RUN_DIRS < <(
  while IFS= read -r -d '' config; do
    if grep -qF -- "$COUNTER_MARKER" "$config" \
      && grep -qF -- "$MODEL_MARKER" "$config"; then
      dirname "$(dirname "$config")"
    fi
  done < <(find "$WANDB_ROOT" -maxdepth 3 -type f \
    -path "$WANDB_ROOT/offline-run-*/files/config.yaml" -print0 2>/dev/null) \
    | LC_ALL=C sort \
    | tail -3
)

if (( ${#RUN_DIRS[@]} != 3 )); then
  echo "ERROR: expected 3 recent Counter dual-branch runs, found ${#RUN_DIRS[@]}" >&2
  exit 2
fi

synced=0
failed=0
for run_dir in "${RUN_DIRS[@]}"; do
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ -z "$run_file" ]]; then
    echo "ERROR: no run-*.wandb file found in $run_dir" >&2
    failed=$((failed + 1))
    continue
  fi

  echo "syncing: $run_dir"
  if timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
    -e "$WANDB_ENTITY" \
    -p "$WANDB_PROJECT" \
    --append \
    --include-offline \
    --include-synced \
    --no-mark-synced \
    --skip-console \
    "$run_dir"; then
    synced=$((synced + 1))
  else
    echo "ERROR: W&B sync failed for $run_dir" >&2
    failed=$((failed + 1))
  fi
done

echo "sync result: uploaded=$synced failed=$failed"
(( failed == 0 ))
