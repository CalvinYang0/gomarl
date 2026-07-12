#!/bin/bash
set -euo pipefail

cat >&2 <<'EOF'
This legacy live-sync script is intentionally disabled.

Standard `wandb sync` must not read an offline run while Slurm is still
writing it: W&B can publish the partial snapshot as a Finished remote run.
Monitor active training through its Slurm log. After completion, use:

  scripts/ozstar_sync_completed_wandb.sh wandb/offline-run-...
EOF
exit 2
