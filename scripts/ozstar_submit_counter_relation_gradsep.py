#!/usr/bin/env python3
"""Submit the gradient-isolated relcoef=.1 Counter control.

Mutable outputs use the personal home runtime. An active same-name job from
this repository is retained; this script never cancels another job.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans, run
from ozstar_submit_relation_advantage_mixer_nine import route_runtime


LABEL = "relation_all4_relcoef01_gradsep"


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    ))
    os.environ.setdefault("RUN_SUFFIX", "_home1")
    plans = build_plans(repo, (LABEL,))
    if len(plans) != 1 or plans[0]["label"] != LABEL:
        raise RuntimeError("Gradient-separation job selection changed")
    paths = route_runtime(plans, runtime_root)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    allowed_roots = ("/home/kyang/", "/fred/oz501/kyang/")
    if not runtime_root.is_absolute() or not any(
        str(runtime_root).startswith(root) for root in allowed_roots
    ):
        raise RuntimeError(
            "Runtime output must be below /home/kyang or /fred/oz501/kyang"
        )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))

    os.chdir(repo)
    subprocess.run(
        [sys.executable,
         "scripts/smoke_test_relation_all4_relcoef01_gradsep.py"],
        check=True,
    )
    plan = plans[0]
    retained = None
    user = run(["id", "-un"])
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name != plan["job_name"]:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same-name job has another WorkDir: " + name)
        if retained is not None:
            raise RuntimeError("Duplicate active job: " + name)
        retained = job_id

    train_script = "scripts/ozstar_train_offline.sbatch"
    if retained is None:
        subprocess.run(
            ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]),
            check=True,
        )

    manifest = paths["logs"] / (
        "counter_relation_gradsep_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "runtime_root": str(runtime_root),
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
        print("Retained {} job={}".format(plan["job_name"], retained))
    else:
        result = run(
            ["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]),
        )
        record["submitted"] = result.split(";", 1)[0]
        persist()
        print("Submitted {} job={}".format(plan["job_name"], result))
    print("Manifest: " + str(manifest))
    print(run([
        "squeue", "-u", user, "-o",
        "%.18i %.100j %.10T %.12M %.10m %R",
    ]))


if __name__ == "__main__":
    main()
