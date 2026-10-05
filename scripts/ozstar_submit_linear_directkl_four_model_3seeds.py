#!/usr/bin/env python3
"""Submit four matched single-Linear models on four maps with three seeds.

Completed historical runs are reused only for the known-valid model/version
names below. Legacy ``controlled15`` direct-KL runs used the wrong gate branch
and are deliberately never counted. No existing Slurm job is cancelled.
"""

import fcntl
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
    SCENES,
    _load_profiles,
    _plan,
)
from ozstar_submit_relation_advantage_mixer_nine import (  # noqa: E402
    route_runtime,
)


GROUP = "linear_directkl_four_models_5m_3seeds"
MODELS = (
    ("linear_baseline", "paper_linear_singlehead"),
    ("linear_bayesg_kl80_keep", "linear_kl80_direct"),
    ("linear_directkl_qme_action_q_episode_mean", "linear_directkl_qme_epmean_masked"),
    ("linear_directkl_qme_td_quality", "linear_directkl_qme_tdquality_masked"),
)
QME_LABELS = {MODELS[2][0], MODELS[3][0]}
EXPECTED_RUNS = 4 * 4 * 3


def historical_candidates(scene, label, seed):
    if label == "linear_baseline":
        return ["{}_paper_linear_singlehead_5m_s{}".format(scene, seed)]
    if scene != "grf_counter" or seed != 1:
        return []
    if label == "linear_bayesg_kl80_keep":
        return ["grf_counter_linear_kl80_direct_5m_s1_linearfix"]
    if label == "linear_directkl_qme_action_q_episode_mean":
        return ["grf_counter_linear_directkl_qme_epmean_masked_5m_s1_controlled20"]
    if label == "linear_directkl_qme_td_quality":
        return ["grf_counter_linear_directkl_qme_tdquality_masked_5m_s1_controlled20"]
    raise ValueError("Unknown model label: " + label)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for scene, map_name, domain, baseline_memory in SCENES:
        for label, suffix in MODELS:
            flags = profiles.ALL_PROFILES[label]
            assert flags.get("branch") == "linear"
            assert bool(flags.get("kl")) == (label != "linear_baseline")
            assert not flags.get("aux")
            if label in QME_LABELS:
                assert flags.get("nomask_td_coef") == 1.0
                assert flags.get("advantage_margin")
            for seed in (1, 2, 3):
                name = "{}_{}_5m_s{}_verify4maps".format(scene, suffix, seed)
                memory = "32G" if label in QME_LABELS else baseline_memory
                plan = _plan(
                    repo, profiles, scene, map_name, domain, label, seed,
                    name, memory, GROUP,
                )
                plan["exports"]["T_MAX"] = "5050000"
                days = 4 if label in QME_LABELS and scene == "smac_mmm2" else (
                    3 if label in QME_LABELS else 2
                )
                plan["sbatch_args"] = [
                    "--time={}-00:00:00".format(days)
                    if arg.startswith("--time=") else arg
                    for arg in plan["sbatch_args"]
                ]
                plan["historical_candidates"] = historical_candidates(
                    scene, label, seed
                )
                plans.append(plan)
    if len(plans) != EXPECTED_RUNS:
        raise RuntimeError("Expected {} runs".format(EXPECTED_RUNS))
    if len({plan["job_name"] for plan in plans}) != EXPECTED_RUNS:
        raise RuntimeError("Duplicate job name")
    return plans


def completed_jobs(user):
    result = run([
        "sacct", "-u", user, "-S", "2026-09-23", "-X", "-n", "-P",
        "--format=JobIDRaw,JobName%120,State,ExitCode",
    ])
    completed = {}
    for line in result.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        job_id, name, state, exit_code = (part.strip() for part in parts)
        if state == "COMPLETED" and exit_code == "0:0":
            completed[name] = job_id
    return completed


def active_jobs(user, repo, relevant_names):
    active = {}
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name not in relevant_names:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError("Same-name job is in another work directory: " + name)
        if name in active:
            raise RuntimeError("Duplicate active job: " + name)
        active[name] = job_id
    return active


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/fred/oz501/kyang/gomarl-runtime/gomarl-dual-branch"
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

    with (paths["logs"] / ".linear_four_model_submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        user = run(["id", "-un"])
        relevant_names = {
            name for plan in plans
            for name in [plan["job_name"]] + plan["historical_candidates"]
        }
        completed = completed_jobs(user)
        active = active_jobs(user, repo, relevant_names)
        chosen = {}
        to_submit = []
        for plan in plans:
            names = [plan["job_name"]] + plan["historical_candidates"]
            existing = next((name for name in names if name in active), None)
            if existing is not None:
                chosen[plan["job_name"]] = {
                    "source": "active", "name": existing, "job_id": active[existing]
                }
                continue
            existing = next((name for name in names if name in completed), None)
            if existing is not None:
                chosen[plan["job_name"]] = {
                    "source": "completed", "name": existing,
                    "job_id": completed[existing],
                }
                continue
            to_submit.append(plan)

        train_script = "scripts/ozstar_train_offline.sbatch"
        for plan in to_submit:
            subprocess.run(
                ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
                env=dict(os.environ, **plan["exports"]),
                check=True,
            )

        manifest = paths["logs"] / (
            "linear_four_model_3seeds_{}_{}.json".format(
                time.strftime("%Y%m%d_%H%M%S"), os.getpid()
            )
        )
        record = {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "group": GROUP,
            "expected_runs": EXPECTED_RUNS,
            "reused_or_active": chosen,
            "submitted": {},
            "new_plans": to_submit,
        }

        def persist():
            with manifest.open("w") as handle:
                json.dump(record, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

        persist()
        print("Reused or active: {} / {}".format(len(chosen), EXPECTED_RUNS))
        for name, details in chosen.items():
            print("{} {} job={}".format(details["source"], name, details["job_id"]))
        for plan in to_submit:
            result = run(
                ["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
                env=dict(os.environ, **plan["exports"]),
            )
            job_id = result.split(";", 1)[0]
            record["submitted"][plan["job_name"]] = job_id
            persist()
            print("Submitted {} job={}".format(plan["job_name"], job_id), flush=True)
        print("Manifest: " + str(manifest))


if __name__ == "__main__":
    main()
