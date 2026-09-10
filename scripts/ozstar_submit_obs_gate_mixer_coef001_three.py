#!/usr/bin/env python3
"""Preflight and append the combined control on Counter, MMM2, and 3s5z."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans as counter_plans, run
from ozstar_submit_trans9_multiscene import build_plans as multiscene_plans


LABEL = "obs_gate_kl80aux_mixer_coef001"
EXPECTED = (("counter", LABEL), ("mmm2", LABEL), ("3s5z", LABEL))


def build_selected_plans(repo):
    plan, = counter_plans(repo, [LABEL])
    plan.update(
        scene="counter",
        map_name="academy_counterattack_easy",
        domain="grf",
    )
    plans = [plan]
    previous = {key: os.environ.get(key) for key in ("SCENES", "LABELS")}
    os.environ.update(SCENES="mmm2 3s5z", LABELS=LABEL)
    try:
        plans.extend(multiscene_plans(repo))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    actual = tuple((item["scene"], item["label"]) for item in plans)
    if actual != EXPECTED:
        raise RuntimeError("Three-job selection changed: {}".format(actual))
    if len({item["job_name"] for item in plans}) != 3:
        raise RuntimeError("Three-job plan contains duplicate names")
    return plans


def main():
    repo = Path(
        os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")
    ).resolve()
    plans = build_selected_plans(repo)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return
    os.chdir(repo)
    subprocess.run(
        [sys.executable, "scripts/smoke_test_obs_gate_mixer_coef001.py"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/smoke_test_trans9_multiscene.py"],
        env=dict(os.environ, SCENES="mmm2 3s5z", LABELS=LABEL),
        check=True,
    )

    user = run(["id", "-un"])
    names = {plan["job_name"] for plan in plans}
    retained = {}
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name not in names:
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

    manifest = logdir / "obs_gate_mixer_coef001_three_{}_{}.json".format(
        time.strftime("%Y%m%d_%H%M%S"), os.getpid()
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

    persist()
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
