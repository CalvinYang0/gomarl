#!/bin/bash
set -euo pipefail

cat >&2 <<'EOF'
Refusing to sync active W&B offline directories.

`wandb sync` is an end-of-run upload command. Running it repeatedly while a
Slurm job is still appending to `run-*.wandb` can create a partial remote run
that W&B marks Finished. Monitor an active job with `tail -F ozstar_logs/...out`.
After the job finishes, upload the complete local run once with:

  scripts/ozstar_sync_completed_wandb.sh wandb/offline-run-...
EOF
exit 2
