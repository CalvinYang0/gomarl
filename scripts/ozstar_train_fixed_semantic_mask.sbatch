#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:?SOURCE_JOB_ID must identify the discovery job}"

cd "$REPO_DIR"

MASK_OUTPUT=$("$PYTHON_BIN" scripts/show_semantic_slot_routes.py \
  "$SOURCE_JOB_ID" --log-dir ozstar_logs --flat-mask)
SEMANTIC_MASK=$(printf '%s\n' "$MASK_OUTPUT" | tail -n 1)

if [[ ! "$SEMANTIC_MASK" =~ ^[01]+$ ]]; then
  echo "Could not extract a binary semantic mask from job $SOURCE_JOB_ID" >&2
  printf '%s\n' "$MASK_OUTPUT" >&2
  exit 1
fi

echo "== Fixed semantic route stage =="
echo "source_job_id: $SOURCE_JOB_ID"
echo "semantic_mask_width: ${#SEMANTIC_MASK}"
echo "semantic_mask: $SEMANTIC_MASK"

EXTRA_ARGS="${EXTRA_ARGS:-} clean_semantic_router_fixed_mask=$SEMANTIC_MASK"
export EXTRA_ARGS

exec bash scripts/ozstar_train_offline.sbatch
