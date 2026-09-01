#!/bin/bash
set -u

# One-shot incremental W&B upload for every currently running Counter job in
# this checkout. Exact run names select the newest matching local directory.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl-dual-branch}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
WANDB_ROOT="${WANDB_ROOT:-wandb}"
WANDB_ENTITY="${WANDB_ENTITY:-hjh331-sjtu}"
WANDB_PROJECT="${WANDB_PROJECT:-gomarl}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-600}"

cd "$REPO_DIR" || exit 2

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

job_run_name() {
  local job_name="$1"
  if [[ "$job_name" =~ ^(.+)_s([0-9]+)(.*)$ ]]; then
    printf '%s_10m_s%s%s\n' \
      "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

find_current_run_dir() {
  local run_name="$1"
  local run_dir run_file best_dir=""
  while IFS= read -r -d '' run_dir; do
    if [[ -f "$run_dir/files/config.yaml" ]] && \
       grep -qF -- "$run_name" "$run_dir/files/config.yaml"; then
      :
    elif grep -RaqF -- "$run_name" \
        "$run_dir/files" "$run_dir/logs" 2>/dev/null; then
      :
    else
      run_file=$(find "$run_dir" -maxdepth 1 -type f \
        -name 'run-*.wandb' -print -quit)
      [[ -f "$run_file" ]] && grep -aqF -- "$run_name" "$run_file" || continue
    fi
    if [[ -z "$best_dir" || "$run_dir" > "$best_dir" ]]; then
      best_dir="$run_dir"
    fi
  done < <(
    find "$WANDB_ROOT" -mindepth 1 -maxdepth 1 -type d \
      -name 'offline-run-*' -print0 2>/dev/null
  )
  [[ -n "$best_dir" ]] && printf '%s\n' "$best_dir"
}

find_run_dir_from_job_logs() {
  local job_record="$1" log_file candidate
  local stdout_file stderr_file
  stdout_file=$(sed -n 's/.* StdOut=\([^ ]*\).*/\1/p' <<< "$job_record")
  stderr_file=$(sed -n 's/.* StdErr=\([^ ]*\).*/\1/p' <<< "$job_record")
  for log_file in "$stdout_file" "$stderr_file"; do
    [[ -f "$log_file" ]] || continue
    candidate=$(
      grep -aoE '(/[^[:space:]]+)?wandb/offline-run-[0-9]{8}_[0-9]{6}-[[:alnum:]]+' \
        "$log_file" 2>/dev/null | tail -1
    )
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" != /* ]]; then
      candidate="$REPO_DIR/$candidate"
    fi
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

uploaded=0
failed=0
matched=0

while IFS='|' read -r job_id job_name start_time; do
  [[ "$job_name" == grf_counter_* ]] || continue
  job_record=$(scontrol show job -o "$job_id" 2>/dev/null || true)
  work_dir=$(sed -n 's/.* WorkDir=\([^ ]*\).*/\1/p' <<< "$job_record")
  [[ "$work_dir" == "$REPO_DIR" ]] || continue
  matched=$((matched + 1))

  run_name=$(job_run_name "$job_name") || {
    echo "ERROR: cannot infer W&B run name for job=$job_id name=$job_name" >&2
    failed=$((failed + 1))
    continue
  }
  # W&B prints its exact local directory into the Slurm stream.  Prefer that
  # authoritative path; config-name lookup remains a fallback for older logs.
  run_dir=$(find_run_dir_from_job_logs "$job_record" || true)
  if [[ -z "$run_dir" ]]; then
    run_dir=$(find_current_run_dir "$run_name")
  fi
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: active offline run not found job=$job_id run=$run_name" >&2
    failed=$((failed + 1))
    continue
  fi
  run_file=$(find "$run_dir" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)
  if [[ ! -f "$run_file" ]]; then
    echo "ERROR: W&B data file missing job=$job_id dir=$run_dir" >&2
    failed=$((failed + 1))
    continue
  fi

  echo "syncing job=$job_id name=$job_name run=$run_name bytes=$(stat -c '%s' "$run_file")"
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
done < <(squeue -u "${USER:-kyang}" -h -t R -o '%A|%j|%S')

echo "running Counter sync result: matched=$matched uploaded=$uploaded failed=$failed"
(( matched > 0 && failed == 0 ))
