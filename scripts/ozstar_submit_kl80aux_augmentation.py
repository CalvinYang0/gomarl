#!/usr/bin/env python3
"""Preflight and append the mask-free Counter KL80 augmentation control."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans, run


LABEL = "kl80aux_augmentation"


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    plan, = build_plans(repo, [LABEL])
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps([plan], indent=2))
        return
    os.chdir(repo)
    subprocess.run(
        [sys.executable, "scripts/smoke_test_kl80aux_augmentation.py"],
        check=True,
    )

    user = run(["id", "-un"])
    retained = None
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name != plan["job_name"]:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same-name job has a different/unknown WorkDir")
        if retained is not None:
            raise RuntimeError("Duplicate active augmentation job")
        retained = job_id

    train_script = "scripts/ozstar_train_offline.sbatch"
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    if retained is None:
        subprocess.run(
            ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]), check=True,
        )
    manifest = logdir / "kl80aux_augmentation_{}_{}.json".format(
        time.strftime("%Y%m%d_%H%M%S"), os.getpid()
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "worktree_status": run(["git", "status", "--short"]),
        "plan": plan,
        "retained": retained,
        "submitted": None,
    }

    def persist():
        with manifest.open("w") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

    persist()
    if retained is not None:
        print("Retained {} job={}".format(plan["job_name"], retained), flush=True)
    else:
        result = run(
            ["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]),
        )
        record["submitted"] = result.split(";", 1)[0]
        persist()
        print("Submitted {} job={}".format(plan["job_name"], result), flush=True)
    print("Manifest: " + str(manifest))
    print(run(["squeue", "-u", user, "-o", "%.18i %.80j %.10T %.12M %R"]))


if __name__ == "__main__":
    main()
