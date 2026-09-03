#!/usr/bin/env python3
"""Append the two learned KL-prior controls; never cancel existing jobs."""
import json
import os
from pathlib import Path
from ozstar_submit_counter_kl80aux_no_relation import main
from ozstar_submit_counter_transformer_nine import build_plans


if __name__ == "__main__":
    labels = ["relation_kl50aux", "relation_kl30aux"]
    if os.environ.get("DRY_RUN") == "YES":
        repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
        print(json.dumps(build_plans(repo, labels), indent=2))
    else:
        for label in labels:
            main(label, "smoke_test_counter_kl_prior_aux.py")
