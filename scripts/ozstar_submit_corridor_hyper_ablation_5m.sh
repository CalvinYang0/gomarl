#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAPS="${MAPS:-corridor}"
export MODELS="${MODELS:-rpg_linear_interaction_hypercond rpg_fixed_linear_structured_maker}"
export SEEDS="${SEEDS:-1 2 3 4 5}"
export T_MAX="${T_MAX:-5000000}"
export TEST_INTERVAL="${TEST_INTERVAL:-10000}"
export TIME="${TIME:-12:00:00}"

exec "$SCRIPT_DIR/ozstar_submit_variant_matrix.sh"
