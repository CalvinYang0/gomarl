#!/usr/bin/env python3
"""Append one no-relation control using the original nine-model settings."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans, run


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    os.chdir(repo)
    plan, = build_plans(repo, ["obs_gate_kl80aux"])
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plan, indent=2))
        return
    subprocess.run([sys.executable, "scripts/smoke_test_counter_kl80aux_no_relation.py"], check=True)
    user = run(["id", "-un"])
    active = run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"])
    for row in active.splitlines():
        job_id, name, state = row.split("|", 2)
        if name != plan["job_name"]:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same job name has a different/unknown WorkDir; refusing duplicate submission")
        print("Existing job={} state={} name={}; nothing submitted or cancelled".format(job_id, state, name))
        return
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    env = dict(os.environ, **plan["exports"])
    subprocess.run(["sbatch", "--test-only"] + plan["sbatch_args"] +
                   ["scripts/ozstar_train_offline.sbatch"], env=env, check=True)
    record = dict(commit=run(["git", "rev-parse", "HEAD"]), plan=plan, job_id=None)
    manifest = logdir / ("kl80aux_no_relation_{}_{}.json".format(time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
    def persist():
        with manifest.open("w") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    persist()  # Fail on storage quota before submitting any job.
    result = run(["sbatch", "--parsable"] + plan["sbatch_args"] +
                 ["scripts/ozstar_train_offline.sbatch"], env=env)
    record["job_id"] = result.split(";", 1)[0]
    print("Submitted job={} name={}".format(record["job_id"], plan["job_name"]), flush=True)
    persist()
    print("Manifest: {}; no existing jobs cancelled".format(manifest))


if __name__ == "__main__":
    main()
