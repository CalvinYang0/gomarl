#!/bin/bash
set -euo pipefail

# Restart the controlled six-job comparison:
#   2 routing mechanisms x {GRF Counter, GRF Pass, SMAC Corridor}.

REPO_DIR="${REPO_DIR:-/home/kyang/code/gomarl}"
PYTHON_BIN="${PYTHON_BIN:-/home/kyang/.conda/envs/marl_cpu/bin/python}"
CANCEL_EXISTING="${CANCEL_EXISTING:-False}"
CANCEL_WAIT_SECONDS="${CANCEL_WAIT_SECONDS:-120}"
TIME="${TIME:-1-00:00:00}"
T_MAX="${T_MAX:-10050000}"
SEED="${SEED:-1}"

cd "$REPO_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "running both smoke tests before cancelling existing jobs"
"$PYTHON_BIN" scripts/smoke_test_stochastic_hard_gate.py
"$PYTHON_BIN" scripts/smoke_test_mlp_drop_relation.py

cancel_comparison_jobs() {
  trap - ERR
  mapfile -t comparison_jobs < <(
    squeue -u "$USER" -h -o "%i|%j" |
      awk -F'|' \
        '$2 ~ /^(grf_counter|grf_pass|corridor)_mlp_(ehard|bsoft)_s[0-9]+$/ {print $1}'
  )
  if (( ${#comparison_jobs[@]} > 0 )); then
    echo "rolling back partial comparison jobs: ${comparison_jobs[*]}" >&2
    scancel "${comparison_jobs[@]}"
  fi
}

if [[ "$CANCEL_EXISTING" == "True" ]]; then
  mapfile -t active_jobs < <(squeue -u "$USER" -h -o "%i")
  if (( ${#active_jobs[@]} > 0 )); then
    echo "cancelling existing jobs: ${active_jobs[*]}"
    scancel "${active_jobs[@]}"

    deadline=$((SECONDS + CANCEL_WAIT_SECONDS))
    while (( SECONDS < deadline )); do
      if ! squeue -u "$USER" -h | grep -q .; then
        break
      fi
      sleep 2
    done
  else
    echo "no existing jobs to cancel"
  fi

  if squeue -u "$USER" -h | grep -q .; then
    echo "ERROR: existing jobs remain after ${CANCEL_WAIT_SECONDS}s" >&2
    squeue -u "$USER"
    exit 3
  fi
else
  echo "ERROR: set CANCEL_EXISTING=True for an explicit clean restart" >&2
  exit 2
fi

export REPO_DIR PYTHON_BIN TIME T_MAX SEED
export RUN_SMOKE_TEST=False

trap cancel_comparison_jobs ERR
bash scripts/ozstar_submit_stochastic_hard_counter_corridor.sh
bash scripts/ozstar_submit_binary_perturb_soft_counter_corridor.sh
trap - ERR

echo
echo "Expected comparison jobs:"
squeue -u "$USER" -h -o "%i|%j|%T|%R" |
  awk -F'|' \
    '$2 ~ /^(grf_counter|grf_pass|corridor)_mlp_(ehard|bsoft)_s[0-9]+$/'
