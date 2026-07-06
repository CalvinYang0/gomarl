#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAPS="${MAPS:-5m_vs_6m corridor}"
export MODELS="${MODELS:-rpg_semantic_selfattn_relation_hypercond rpg_entity_selfattn_relation_hypercond rpg_delta_relation_hypercond rpg_relation_coarse_self_fine_head rpg_relation_prototype_single_head}"
export SEEDS="${SEEDS:-1}"
export T_MAX="${T_MAX:-5000000}"
export TEST_INTERVAL="${TEST_INTERVAL:-10000}"
export TIME="${TIME:-12:00:00}"

exec "$SCRIPT_DIR/ozstar_submit_variant_matrix.sh"
