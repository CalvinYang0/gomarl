#!/usr/bin/env python3
"""Preflight and submit the seven recent gate/relation controls on Counter.

The suite is append-only: an active job with the same name and repository
working directory is retained, while unrelated jobs are never cancelled.
DRY_RUN=YES prints the complete plan without invoking smoke tests or Slurm.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans, run


LABELS = (
    # Observation-conditioned masks grouped by instantaneous Advantage.
    "relation_advantage_kl80aux",
    # One learned global mask shared by all observations, agents and timesteps.
    "static_gate_kl80aux_thr05",
    "static_gate_kl80aux_thr08",
    "static_gate_kl80aux_sharp_thr05",
    # Full-observation main/test path; KL80 is training-only augmentation.
    "kl80aux_augmentation",
    # One fixed mask per agent, with trajectory-derived group supervision.
    "relation_trajectory_agent_kl80aux",
    # One independently learned fixed mask per agent, without group loss.
    "agent_static_gate_kl80aux",
)

SMOKE_TESTS = (
    "scripts/smoke_test_advantage_relation.py",
    "scripts/smoke_test_static_gate_controls.py",
    "scripts/smoke_test_kl80aux_augmentation.py",
    "scripts/smoke_test_trajectory_agent_relation.py",
    "scripts/smoke_test_agent_static_gate.py",
)


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    plans = build_plans(repo, LABELS)
    actual = tuple(plan["label"] for plan in plans)
    if actual != LABELS:
        raise RuntimeError("Counter control selection changed: {}".format(actual))
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    os.chdir(repo)
    smoke_failures = []
    for index, smoke_test in enumerate(SMOKE_TESTS, start=1):
        print(
            "[smoke {}/{}] START {}".format(
                index, len(SMOKE_TESTS), smoke_test
            ),
            flush=True,
        )
        result = subprocess.run([sys.executable, smoke_test])
        if result.returncode:
            smoke_failures.append((smoke_test, result.returncode))
            print(
                "[smoke {}/{}] FAILED {} (exit={})".format(
                    index, len(SMOKE_TESTS), smoke_test, result.returncode
                ),
                flush=True,
            )
        else:
            print(
                "[smoke {}/{}] PASSED {}".format(
                    index, len(SMOKE_TESTS), smoke_test
                ),
                flush=True,
            )
    if smoke_failures:
        details = ", ".join(
            "{}:exit={}".format(path, code)
            for path, code in smoke_failures
        )
        raise RuntimeError(
            "Counter control preflight failed after running every smoke test; "
            "no jobs submitted: " + details
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
            raise RuntimeError(
                "Same-name job has a different/unknown WorkDir: " + name
            )
        if name in retained:
            raise RuntimeError("Duplicate active job: " + name)
        retained[name] = job_id

    train_script = "scripts/ozstar_train_offline.sbatch"
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    for plan in plans:
        if plan["job_name"] in retained:
            continue
        subprocess.run(
            ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]),
            check=True,
        )

    manifest = logdir / "counter_gate_relation_controls_{}_{}.json".format(
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
    print(run([
        "squeue", "-u", user, "-o",
        "%.18i %.90j %.10T %.12M %.10m %R",
    ]))


if __name__ == "__main__":
    main()
