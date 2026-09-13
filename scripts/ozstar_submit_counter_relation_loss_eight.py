#!/usr/bin/env python3
"""Submit the eight Counter relation loss-composition controls."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans, run
from ozstar_submit_relation_advantage_mixer_nine import route_runtime


LABELS = (
    "relation_lnomask_lmask",
    "kl80_ltd_lkl_only",
    "relation_all4",
    "relation_lnomask_lmask_testopen",
    "relation_all4_testopen",
    "relation_all4_sigmoid",
    "relation_all4_relcoef10",
    "relation_all4_relcoef01",
)


def selected_plans(repo):
    plans = build_plans(repo, LABELS)
    if tuple(plan["label"] for plan in plans) != LABELS:
        raise RuntimeError("Eight-job loss-composition selection changed")
    if len({plan["job_name"] for plan in plans}) != len(LABELS):
        raise RuntimeError("Eight-job plan contains duplicate job names")
    return plans


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    ))
    plans = selected_plans(repo)
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
        [sys.executable, "scripts/smoke_test_relation_loss_eight.py"],
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
            raise RuntimeError("Same-name job has another WorkDir: " + name)
        if name in retained:
            raise RuntimeError("Duplicate active job: " + name)
        retained[name] = job_id

    train_script = "scripts/ozstar_train_offline.sbatch"
    for plan in plans:
        if plan["job_name"] not in retained:
            subprocess.run(
                ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
                env=dict(os.environ, **plan["exports"]),
                check=True,
            )

    manifest = paths["logs"] / (
        "counter_relation_loss_eight_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "runtime_root": str(runtime_root),
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
        "%.18i %.100j %.10T %.12M %.10m %R",
    ]))


if __name__ == "__main__":
    main()
