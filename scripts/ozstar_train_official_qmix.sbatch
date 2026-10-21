#!/bin/bash
# Resources and log paths are supplied by ozstar_submit_official_qmix_6jobs.py.
set -euo pipefail
cd "${REPO_DIR:?REPO_DIR must identify the submitted checkout}"
[[ "${CONFIG:-}" == official_pymarl2_qmix ]] || exit 2
echo "Official QMIX job checkout: $(git rev-parse HEAD)"
echo "Pinned PyMARL2: 8ccac7c5aa134422a2e3009be735d23cdf8ce2f8"
export SC2PATH="${SC2PATH:-/home/kyang/StarCraftII}"
"${PYTHON_BIN:?}" scripts/smoke_test_official_qmix.py --sources-only --report-environment
exec bash scripts/ozstar_train_offline.sbatch
