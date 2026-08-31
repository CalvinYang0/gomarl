#!/bin/bash
set -euo pipefail

# Final-sync the selected eight Counter experiments after their Slurm jobs have
# left the queue.  Match by the exact W&B run name and choose the newest local
# offline directory for each name.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"
SEED="${SEED:-1}"

cd "$REPO_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$WANDB_ROOT" ]]; then
  echo "ERROR: W&B directory does not exist: $REPO_DIR/$WANDB_ROOT" >&2
  exit 2
fi
for command in find grep stat timeout; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command" >&2
    exit 2
  }
done

RUN_NAMES=(
  "grf_counter_equal1_episode_random_randommask_k50_w10_timestep_10m_s${SEED}"
  "grf_counter_equal1_episode_random_sampled_gate_10m_s${SEED}"
  "grf_counter_equal1_episode_random_randommask_k30_w10_timestep_10m_s${SEED}"
  "grf_counter_equal1_episode_random_randommask_k70_w10_timestep_10m_s${SEED}"
  "grf_counter_transformer_only_obs_gate_10m_s${SEED}"
  "grf_counter_transformer_only_obs_gate_randommask_d50_w10_timestep_10m_s${SEED}"
  "grf_counter_linear_only_obs_gate_10m_s${SEED}"
  "grf_counter_linear_only_obs_gate_randommask_d50_w10_timestep_10m_s${SEED}"
)

run_dir_matches() {
  local run_dir="$1" run_name="$2" run_file
  if [[ -f "$run_dir/files/config.yaml" ]] && \
     grep -qF -- "$run_name" "$run_dir/files/config.yaml"; then
    return 0
  fi
  if grep -RaqF -- "$run_name" "$run_dir/files" "$run_dir/logs" 2>/dev/null; then
    return 0
  fi
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  [[ -f "$run_file" ]] && grep -aqF -- "$run_name" "$run_file"
}

find_latest_run_dir() {
  local run_name="$1" run_dir latest=""
  while IFS= read -r -d '' run_dir; do
    run_dir_matches "$run_dir" "$run_name" || continue
    if [[ -z "$latest" || "$run_dir" > "$latest" ]]; then
      latest="$run_dir"
    fi
  done < <(
    find "$WANDB_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -name 'offline-run-*' -print0 2>/dev/null
  )
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

uploaded=0
failed=0
for run_name in "${RUN_NAMES[@]}"; do
  if ! run_dir="$(find_latest_run_dir "$run_name")"; then
    run_dir=""
  fi
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: offline run not found: $run_name" >&2
    failed=$((failed + 1))
    continue
  fi
  run_file="$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ ! -f "$run_file" ]]; then
    echo "ERROR: run data is missing: $run_dir" >&2
    failed=$((failed + 1))
    continue
  fi

  echo "syncing run=$run_name bytes=$(stat -c '%s' "$run_file") dir=$run_dir"
  if timeout "$SYNC_TIMEOUT" "$PYTHON_BIN" -m wandb sync \
      -e "$WANDB_ENTITY" \
      -p "$WANDB_PROJECT" \
      --append \
      --include-offline \
      --include-synced \
      --no-mark-synced \
      --skip-console \
      "$run_dir"; then
    uploaded=$((uploaded + 1))
  else
    failed=$((failed + 1))
  fi
done

echo "selected Counter sync result: uploaded=$uploaded failed=$failed total=${#RUN_NAMES[@]}"
(( failed == 0 ))
