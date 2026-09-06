#!/usr/bin/env python3
"""Preflight and submit the exact eight experiments selected on 2026-09-07.

No existing jobs are cancelled. Active same-name jobs from this worktree are
retained, so rerunning the command is safe.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans as counter_plans, run
from ozstar_submit_trans9_multiscene import build_plans as multiscene_plans


COUNTER_LABELS = (
    "relation_kl80aux_klfirst",
    "obs_gate_kl80aux",
    "hyper_hypermarl_id",
    "hyper_cash_obs_type",
    "hyper_rpg_relation",
)
EXPECTED = (
    ("counter", "relation_kl80aux_klfirst"),
    ("counter", "obs_gate_kl80aux"),
    ("counter", "hyper_hypermarl_id"),
    ("counter", "hyper_cash_obs_type"),
    ("counter", "hyper_rpg_relation"),
    ("corridor", "relation_kl80aux"),
    ("mmm2", "baseline"),
    ("mmm2", "relation_kl80aux"),
)


def build_selected_plans(repo):
    plans = counter_plans(repo, COUNTER_LABELS)
    for plan in plans:
        plan.update(scene="counter", map_name="academy_counterattack_easy", domain="grf")

    previous = os.environ.get("SCENES")
    os.environ["SCENES"] = "corridor mmm2"
    try:
        scene_plans = multiscene_plans(repo)
    finally:
        if previous is None:
            os.environ.pop("SCENES", None)
        else:
            os.environ["SCENES"] = previous
    plans.extend(
        plan for plan in scene_plans
        if (plan["scene"], plan["label"]) in {
            ("corridor", "relation_kl80aux"),
            ("mmm2", "baseline"),
            ("mmm2", "relation_kl80aux"),
        }
    )
    actual = tuple((plan["scene"], plan["label"]) for plan in plans)
    if actual != EXPECTED:
        raise RuntimeError("Eight-job selection changed: {}".format(actual))
    if len({plan["job_name"] for plan in plans}) != 8:
        raise RuntimeError("Eight-job plan contains duplicate Slurm names")
    return plans


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    plans = build_selected_plans(repo)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return
    os.chdir(repo)

    smoke_commands = (
        [sys.executable, "scripts/smoke_test_counter_hypernetwork_baselines.py"],
        [sys.executable, "scripts/smoke_test_counter_klfirst.py"],
        [sys.executable, "scripts/smoke_test_counter_kl80aux_no_relation.py"],
        [sys.executable, "scripts/smoke_test_trans9_multiscene.py"],
    )
    smoke_env = dict(os.environ, SCENES="corridor mmm2")
    for command in smoke_commands:
        subprocess.run(command, env=smoke_env, check=True)

    user = run(["id", "-un"])
    expected_names = {plan["job_name"] for plan in plans}
    retained = {}
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name not in expected_names:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same-name job has a different/unknown WorkDir: " + name)
        if name in retained:
            raise RuntimeError("Duplicate active job: " + name)
        retained[name] = job_id

    train_script = "scripts/ozstar_train_offline.sbatch"
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    for plan in plans:
        if plan["job_name"] not in retained:
            subprocess.run(
                ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
                env=dict(os.environ, **plan["exports"]),
                check=True,
            )

    manifest = logdir / (
        "selected_eight_{}_{}.json".format(time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "worktree_status": run(["git", "status", "--short"]),
        "plans": plans,
        "retained": retained,
        "submitted": {},
    }

    def persist():
        with manifest.open("w") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

    persist()  # Detect quota errors before any submission.
    for plan in plans:
        name = plan["job_name"]
        if name in retained:
            print("Retained {} job={}".format(name, retained[name]), flush=True)
            continue
        result = run(
            ["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]),
        )
        record["submitted"][name] = result.split(";", 1)[0]
        persist()
        print("Submitted {} job={}".format(name, result), flush=True)
    print("Manifest: " + str(manifest))
    print(run(["squeue", "-u", user, "-o", "%.18i %.80j %.10T %.12M %R"]))


if __name__ == "__main__":
    main()
