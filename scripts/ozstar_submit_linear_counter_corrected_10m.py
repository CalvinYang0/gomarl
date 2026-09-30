#!/usr/bin/env python3
"""Submit the 10M Counter Linear two-framework QME matrix, seed 1.

Five QME objectives are evaluated under matched Direct-KL and Aux-Multiply
KL80 frameworks. Each framework also has a MaskTD+NoMaskTD no-QME control.
The script can cancel only obsolete active jobs from superseded suites;
completed runs and valid standalone KL/baseline runs are kept.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_counter_transformer_nine import run  # noqa: E402
from ozstar_submit_linear_single_head_suite import (  # noqa: E402
    _load_profiles,
    _plan,
)
from ozstar_submit_relation_advantage_mixer_nine import (  # noqa: E402
    route_runtime,
)


QME_VARIANTS = (
    ("action_q", "action_q_masked"),
    ("action_q_episode_mean", "epmean_masked"),
    ("td_quality", "tdquality_masked"),
    ("joint_value", "joint_value_masked"),
    ("action_q_scaled", "q_scaled_masked"),
)

EXPERIMENTS = [
    ("linear_baseline", "paper_linear_singlehead", "16G"),
    (
        "linear_directkl_nomasktd_control",
        "directkl_nomasktd_control",
        "32G",
    ),
    (
        "linear_auxmultiply_nomasktd_control",
        "auxmul_nomasktd_control",
        "32G",
    ),
]
for _profile_suffix, _run_suffix in QME_VARIANTS:
    EXPERIMENTS.append((
        "linear_directkl_qme_" + _profile_suffix,
        "directkl_qme_" + _run_suffix,
        "32G",
    ))
    EXPERIMENTS.append((
        "linear_auxmultiply_qme_" + _profile_suffix,
        "auxmul_qme_" + _run_suffix,
        "32G",
    ))
EXPERIMENTS = tuple(EXPERIMENTS)

OBSOLETE_PATTERN = re.compile(
    r"^(?:grf_(?:counter|pass)|smac_(?:5m6m|mmm2))_linear_kl80_direct_5m_s[123]_controlled15(?:_retry1)?$"
    r"|^grf_counter_linear_(?:kl80_nomasktd_control|qme_(?:action_q_masked|epmean_masked|tdquality_masked|joint_value_masked|q_scaled_masked|dynamic_ready_masked|openwin_ready_masked|action_q_full))_5m_s1_controlled15(?:_retry1)?$"
    r"|^grf_counter_linear_(?:nomasktd_control_nokl|qme_.+_nokl)_10m_s1_corrected16$"
)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for label, short_name, memory in EXPERIMENTS:
        name = (
            "grf_counter_linear_paper_linear_singlehead_10m_s1_corrected16"
            if label == "linear_baseline"
            else "grf_counter_linear_{}_10m_s1_controlled17".format(short_name)
        )
        plan = _plan(
            repo,
            profiles,
            "grf_counter",
            "academy_counterattack_easy",
            "grf",
            label,
            1,
            name,
            memory,
            "counter_linear_two_kl80_qme_10m_s1",
        )
        plan["exports"]["T_MAX"] = os.environ.get("T_MAX", "10050000")
        plan["sbatch_args"] = [
            "--time=" + os.environ.get("TIME", "4-00:00:00")
            if arg.startswith("--time=") else arg
            for arg in plan["sbatch_args"]
        ]
        plans.append(plan)
    if len(plans) != 13:
        raise RuntimeError(
            "Expected one baseline, two framework controls and 10 QME jobs"
        )
    return plans


def cancel_obsolete_active_jobs(repo, user):
    cancelled = {}
    queue = run(["squeue", "-u", user, "-h", "-o", "%i|%j"])
    for row in queue.splitlines():
        job_id, name = row.split("|", 1)
        if not OBSOLETE_PATTERN.fullmatch(name):
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Refusing to cancel same-name job in another WorkDir: " + name)
        run(["scancel", job_id])
        cancelled[name] = job_id
        print("Cancelled obsolete {} job={}".format(name, job_id), flush=True)
    return cancelled


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    )).resolve()
    plans = build_plans(repo)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    paths = route_runtime(plans, runtime_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))

    os.chdir(repo)
    subprocess.run(
        [sys.executable, "scripts/smoke_test_linear_single_head_suite.py"],
        check=True,
    )

    user = run(["id", "-un"])
    names = {plan["job_name"] for plan in plans}
    retained = {}
    queue = run(["squeue", "-u", user, "-h", "-o", "%i|%j"])
    for row in queue.splitlines():
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
        if plan["job_name"] in retained:
            continue
        subprocess.run(
            ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
            env=dict(os.environ, **plan["exports"]),
            check=True,
        )

    cancelled = {}
    if os.environ.get("CANCEL_OBSOLETE", "NO") == "YES":
        cancelled = cancel_obsolete_active_jobs(repo, user)

    manifest = paths["logs"] / (
        "linear_counter_corrected_10m_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "t_max": 10050000,
        "cancelled": cancelled,
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
        "%.18i %.100j %.10T %.12M %.10l %.10m %R",
    ]))


if __name__ == "__main__":
    main()
