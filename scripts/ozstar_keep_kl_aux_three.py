#!/usr/bin/env python3
"""Keep relation KL80/50/30 auxiliary jobs; cancel only other jobs in this repo."""
import json
import os
from pathlib import Path
import re
import subprocess
import time


KEEP = re.compile(r"grf_counter_trans9_relation_kl(?:80|50|30)aux_s\d+(?:_[A-Za-z0-9_-]+)?")


def read(argv):
    return subprocess.check_output(argv, text=True).strip()


def select_jobs(repo, user):
    selection = dict(keep=[], cancel=[], outside=[])
    rows = read(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"])
    for row in rows.splitlines():
        job_id, name, state = row.split("|", 2)
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            raise RuntimeError("Unsupported job ID; nothing cancelled: " + job_id)
        info = read(["scontrol", "show", "job", "-o", job_id])
        directory = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        owner = re.search(r"(?:^|\s)UserId=([^\s(]+)", info)
        current_name = re.search(r"(?:^|\s)JobName=(\S+)", info)
        if not directory or not owner or not current_name or current_name[1] != name:
            raise RuntimeError("Unverified job metadata; nothing cancelled: " + job_id)
        record = dict(job_id=job_id, name=name, state=state)
        if Path(directory[1]).resolve() != repo or owner[1] != user:
            selection["outside"].append(record)
        else:
            selection["keep" if KEEP.fullmatch(name) else "cancel"].append(record)
    return selection


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve(strict=True)
    if not (repo / "scripts/ozstar_train_offline.sbatch").is_file():
        raise RuntimeError("Not the expected training repository")
    user = read(["id", "-un"])
    selection = select_jobs(repo, user)  # Complete read-only verification first.
    print(json.dumps(selection, indent=2), flush=True)
    if os.environ.get("DRY_RUN") == "YES":
        print("Dry run: no jobs cancelled")
        return
    if selection["cancel"]:
        logdir = repo / "ozstar_logs"
        logdir.mkdir(exist_ok=True)
        manifest = logdir / ("keep_kl_aux_three_{}_{}.json".format(time.time_ns(), os.getpid()))
        with manifest.open("x") as handle:
            json.dump(selection, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())  # Fail on quota before cancellation.
        subprocess.run(["scancel"] + [job["job_id"] for job in selection["cancel"]], check=True)
        print("Cancellation requested; selection saved to " + str(manifest))
    print("Kept {} matching active jobs; no jobs submitted, no logs deleted.".format(len(selection["keep"])))
    print(read(["squeue", "-u", user, "-o", "%.18i %.90j %.10T %.12M %R"]))


if __name__ == "__main__":
    main()
