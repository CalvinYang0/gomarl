#!/bin/bash
set -euo pipefail

export MODELS="${MODELS:-rpg_action_edge_prev_oracle_graph_hypercond rpg_action_edge_public_pred_hypercond rpg_action_edge_public_memory_hypercond rpg_action_edge_global_public_pred_hypercond rpg_action_edge_target_context_hypercond rpg_public_hyper_private_input_single_head rpg_private_hyper_public_input_single_head}"
export MAPS="${MAPS:-5m_vs_6m}"
export SEEDS="${SEEDS:-1}"
export T_MAX="${T_MAX:-10050000}"
export TIME="${TIME:-24:00:00}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
export MEM="${MEM:-64G}"
export BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export BUFFER_SIZE="${BUFFER_SIZE:-5000}"
export TEST_INTERVAL="${TEST_INTERVAL:-10000}"
export USE_WANDB="${USE_WANDB:-True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_CUDA="${USE_CUDA:-False}"

bash scripts/ozstar_submit_variant_matrix.sh
