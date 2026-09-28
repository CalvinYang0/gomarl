#!/usr/bin/env python3
"""Submit matched Linear-only and Transformer-only KL80 gate controls."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_counter_transformer_nine import build_plans, run


LABELS = (
    "linear_bayesg_kl80_keep",
    "transformer_bayesg_kl80_keep",
)
MEMORY_BY_LABEL = {
    # The supplied Linear baseline trace peaks at roughly 13.3 GB.  Sixteen GB
    # leaves useful allocator/worker headroom without requesting the 24 GB
    # needed by the Transformer branch.
    "linear_bayesg_kl80_keep": "16G",
    "transformer_bayesg_kl80_keep": "24G",
}


def build_single_branch_plans(repo):
    plans = build_plans(repo, LABELS)
    for plan in plans:
        label = plan["label"]
        plan["job_name"] = "grf_counter_{}_s{}{}".format(
            label,
            os.environ.get("SEED", "1"),
            os.environ.get("RUN_SUFFIX", ""),
        )
        plan["run_name"] = "grf_counter_{}_10m_s{}{}".format(
            label,
            os.environ.get("SEED", "1"),
            os.environ.get("RUN_SUFFIX", ""),
        )
        plan["exports"]["RUN_NAME"] = plan["run_name"]
        plan["exports"]["GROUP_NAME"] = "counter_single_branch_bayesg_kl80"
        plan["exports"]["TEST_INTERVAL"] = os.environ.get(
            "TEST_INTERVAL", "50000"
        )
        # Keep diagnostics/checkpoints, but do not double evaluation cost with
        # a force-open policy.  Gate probabilities, masks, KL scale and PCA are
        # already emitted by the shared current framework.
        plan["exports"]["EXTRA_ARGS"] += (
            " clean_dual_gate_test=False test_nepisode=32"
        )
        memory = os.environ.get(
            "LINEAR_MEMORY" if label.startswith("linear_") else "TRANSFORMER_MEMORY",
            MEMORY_BY_LABEL[label],
        )
        plan["memory"] = memory
        plan["sbatch_args"] = [
            arg if not arg.startswith("--mem=") else "--mem=" + memory
            for arg in plan["sbatch_args"]
        ]
        plan["sbatch_args"] = [
            arg
            if not arg.startswith("--time=")
            else "--time=" + os.environ.get("TIME", "2-00:00:00")
            for arg in plan["sbatch_args"]
        ]
        plan["sbatch_args"] = [
            arg
            if not arg.startswith("--job-name=")
            else "--job-name=" + plan["job_name"]
            for arg in plan["sbatch_args"]
        ]
    return plans


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    plans = build_single_branch_plans(repo)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    os.chdir(repo)
    subprocess.run(
        [sys.executable,
         "scripts/smoke_test_counter_single_branch_bayesg_kl80.py"],
        check=True,
    )
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    manifest = logdir / (
        "counter_single_branch_bayesg_kl80_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    user = run(["id", "-un"])
    active = run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"])
    names = {plan["job_name"] for plan in plans}
    retained = {}
    for row in active.splitlines():
        job_id, name, state = row.split("|", 2)
        if name not in names or state not in {"RUNNING", "PENDING"}:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same-name job belongs to another checkout: " + name)
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

    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "seed": int(os.environ.get("SEED", "1")),
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
        print(
            "Submitted {} job={} memory={}".format(
                name, result, plan["memory"]
            ),
            flush=True,
        )

    print("Manifest: " + str(manifest))
    print(run([
        "squeue", "-u", user, "-o",
        "%.18i %.80j %.10T %.12M %.10l %.10m %R",
    ]))


if __name__ == "__main__":
    main()
