#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KEEP_PROBABILITY=0.50 exec bash \
  "$SCRIPT_DIR/ozstar_submit_counter_fixed_random_drop80.sh"
