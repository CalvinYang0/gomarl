#!/usr/bin/env python3
"""Append the two lower-weight Counter mixer-KL80 controls."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ozstar_submit_counter_kl80aux_no_relation import main as submit_one
from ozstar_submit_counter_transformer_nine import build_plans


LABELS = ("mixer_kl80aux_coef01", "mixer_kl80aux_coef001")


if __name__ == "__main__":
    repo = Path(
        os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")
    ).resolve()
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(build_plans(repo, LABELS), indent=2))
    else:
        os.chdir(repo)
        subprocess.run(
            [sys.executable, "scripts/smoke_test_counter_mixer_kl80_weights.py"],
            check=True,
        )
        for label in LABELS:
            submit_one(label, smoke_script=None)
