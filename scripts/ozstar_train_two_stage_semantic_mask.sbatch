#!/bin/bash
set -euo pipefail

# One Slurm job, two independent training processes:
#   stage 1 learns a slot-level mask adaptively from an all-TOKEN start;
#   stage 2 rebuilds a compact network from that mask and trains from scratch.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
DISCOVERY_MODEL_TYPE="${DISCOVERY_MODEL_TYPE:?DISCOVERY_MODEL_TYPE is required}"
DISCOVERY_T_MAX="${DISCOVERY_T_MAX:-5000000}"

FINAL_MODEL_TYPE="${MODEL_TYPE:?MODEL_TYPE is required}"
FINAL_RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
FINAL_GROUP_NAME="${GROUP_NAME:-$FINAL_RUN_NAME}"
FINAL_T_MAX="${T_MAX:-10050000}"
FINAL_USE_WANDB="${USE_WANDB:-True}"
FINAL_EXTRA_ARGS="${EXTRA_ARGS:-}"
DISCOVERY_RUN_NAME="${FINAL_RUN_NAME}_route_discovery"

cd "$REPO_DIR"

echo "== Two-stage adaptive semantic routing =="
echo "stage_1_model: $DISCOVERY_MODEL_TYPE"
echo "stage_1_t_max: $DISCOVERY_T_MAX"
echo "stage_2_model: $FINAL_MODEL_TYPE"
echo "stage_2_t_max: $FINAL_T_MAX"

echo "== Stage 1: adaptive mask discovery =="
MODEL_TYPE="$DISCOVERY_MODEL_TYPE" \
RUN_NAME="$DISCOVERY_RUN_NAME" \
GROUP_NAME="$DISCOVERY_RUN_NAME" \
T_MAX="$DISCOVERY_T_MAX" \
USE_WANDB=False \
EXTRA_ARGS="$FINAL_EXTRA_ARGS clean_semantic_router_fixed_mask= clean_semantic_router_freeze_steps=$DISCOVERY_T_MAX" \
  bash scripts/ozstar_train_offline.sbatch

MASK_OUTPUT=$("$PYTHON_BIN" scripts/show_semantic_slot_routes.py \
  "${SLURM_JOB_ID:?SLURM_JOB_ID is required}" --log-dir ozstar_logs --flat-mask)
SEMANTIC_MASK=$(printf '%s\n' "$MASK_OUTPUT" | tail -n 1)

if [[ ! "$SEMANTIC_MASK" =~ ^[01]+$ ]]; then
  echo "ERROR: stage 1 did not produce a valid binary semantic mask." >&2
  printf '%s\n' "$MASK_OUTPUT" >&2
  exit 1
fi

echo "== Stage 1 complete =="
echo "semantic_mask_width: ${#SEMANTIC_MASK}"
echo "semantic_mask: $SEMANTIC_MASK"

echo "== Stage 2: rebuilt compact network =="
MODEL_TYPE="$FINAL_MODEL_TYPE" \
RUN_NAME="$FINAL_RUN_NAME" \
GROUP_NAME="$FINAL_GROUP_NAME" \
T_MAX="$FINAL_T_MAX" \
USE_WANDB="$FINAL_USE_WANDB" \
EXTRA_ARGS="$FINAL_EXTRA_ARGS clean_semantic_router_fixed_mask=$SEMANTIC_MASK" \
  exec bash scripts/ozstar_train_offline.sbatch
