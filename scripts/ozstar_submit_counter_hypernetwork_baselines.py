#!/usr/bin/env python3
"""Submit the three matched Counter hypernetwork comparison runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ozstar_submit_counter_kl80aux_no_relation import main as submit_one
from ozstar_submit_counter_transformer_nine import build_plans


LABELS = (
    "hyper_hypermarl_id",
    "hyper_cash_obs_type",
    "hyper_rpg_relation",
)


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(build_plans(repo, LABELS), indent=2))
        return
    subprocess.run(
        [sys.executable, str(repo / "scripts/smoke_test_counter_hypernetwork_baselines.py")],
        cwd=repo,
        check=True,
    )
    # The shared helper rejects same-name duplicates and never cancels jobs.
    for label in LABELS:
        submit_one(label=label, smoke_script="")


if __name__ == "__main__":
    main()
