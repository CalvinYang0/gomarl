#!/usr/bin/env python3
"""Stop only the six exact historical-attention ID names on the two maps."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from ozstar_submit_counter_transformer_nine import run

NAMES = {"{}_id_baseline_10m_s{}_valuediag".format(scene, seed)
         for scene in ("smac_5m6m", "smac_8m9m") for seed in (1, 2, 3)}


def resolve_jobs(user, repo):
    targets = []
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%200j|%T"]).splitlines():
        job_id, name, state = (item.strip() for item in row.split("|", 2))
        if name not in NAMES:
            continue
        if not job_id.isdigit():
            raise RuntimeError("Unexpected job ID: " + job_id)
        info = run(["scontrol", "show", "job", "-o", job_id])
        directory = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        canonical_name = re.search(r"(?:^|\s)JobName=(\S+)", info)
        if (not directory or Path(directory.group(1)).resolve() != repo
                or not canonical_name or canonical_name.group(1) != name):
            raise RuntimeError("Job name/work directory mismatch; nothing cancelled: " + job_id)
        targets.append(dict(job_id=job_id, name=name, state=state))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    targets = resolve_jobs(run(["id", "-un"]), repo)
    for job in targets:
        print("{} {} {}".format(job["job_id"], job["name"], job["state"]), flush=True)
    if not targets:
        print("No active historical ID jobs on 5m6m/8m9m; nothing to cancel.")
        return
    if not args.execute:
        print("Plan only; --execute cancels exactly the jobs listed above.")
        return
    logs = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch")) / "ozstar_logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = logs / "stop_historical_id_{}_{}.json".format(time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    record = dict(targets=targets, cancelled=[])
    for job in targets:
        manifest.write_text(json.dumps(record, indent=2))
        subprocess.run(["scancel", job["job_id"]], check=True)
        record["cancelled"].append(job["job_id"])
        manifest.write_text(json.dumps(record, indent=2))
        print("Cancelled {} {}".format(job["job_id"], job["name"]), flush=True)
    print("Cancellation record: " + str(manifest))


if __name__ == "__main__":
    main()
