#!/usr/bin/env python3
"""Submit only the corrected Counter Linear direct-KL80 seed-one run."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_counter_transformer_nine import run  # noqa: E402
from ozstar_submit_linear_single_head_suite import (  # noqa: E402
    _load_profiles,
    _plan,
)
from ozstar_submit_relation_advantage_mixer_nine import (  # noqa: E402
    route_runtime,
)


JOB_NAME = "grf_counter_linear_kl80_direct_5m_s1_linearfix"


def build_plan(repo):
    plan = _plan(
        repo,
        _load_profiles(repo),
        "grf_counter",
        "academy_counterattack_easy",
        "grf",
        "linear_bayesg_kl80_keep",
        1,
        JOB_NAME,
        "16G",
        "counter_linear_kl80_direct_linearfix",
    )
    plan["exports"]["T_MAX"] = "5050000"
    return plan


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    )).resolve()
    plan = build_plan(repo)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plan, indent=2))
        return

    paths = route_runtime([plan], runtime_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    subprocess.run(
        [sys.executable, "scripts/smoke_test_linear_single_head_suite.py"],
        check=True,
    )

    user = run(["id", "-un"])
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name != JOB_NAME:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same-name job belongs to another work directory")
        print("Already queued {} job={}".format(JOB_NAME, job_id))
        return

    train_script = "scripts/ozstar_train_offline.sbatch"
    env = dict(os.environ, **plan["exports"])
    subprocess.run(
        ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
        env=env,
        check=True,
    )
    result = run(
        ["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
        env=env,
    )
    print("Submitted {} job={}".format(JOB_NAME, result.split(";", 1)[0]))
    print("Code commit: " + run(["git", "rev-parse", "--short", "HEAD"]))


if __name__ == "__main__":
    main()
