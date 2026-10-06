#!/usr/bin/env python3
"""Fresh 10M Linear single-head baseline replicas on 5m_vs_6m.

Plan-only by default; SUBMIT=YES enables validated Slurm submission.
No cancellations or reuse of historical baseline names. Repeated invocation
retains active/completed jobs with this suite's exact names.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_counter_transformer_nine import run
from ozstar_submit_linear_single_head_suite import _load_profiles, _plan
from ozstar_submit_linear_directkl_four_model_3seeds import active_jobs, completed_jobs
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import (
    guard_and_route, home_quota_free_gib, validate_config_keys,
)

GROUP = "linear_singlehead_5m6m_10m_3seeds_recheck"
SMOKE_SCRIPTS = ("scripts/smoke_test_linear_single_head_suite.py",)
SUITE_FILE = Path(__file__)


def build_plans(repo):
    profiles = _load_profiles(repo)
    flags = profiles.ALL_PROFILES["linear_baseline"]
    if flags.get("branch") != "linear" or any(
        flags.get(key) for key in ("gate", "kl", "aux", "advantage_margin")
    ):
        raise RuntimeError("Expected ungated Linear baseline without auxiliary losses")
    plans = []
    for seed in (1, 2, 3):
        name = "smac_5m6m_linear_singlehead_baseline_10m_s{}_recheck".format(seed)
        plan = _plan(repo, profiles, "smac_5m6m", "5m_vs_6m", "smac",
                     "linear_baseline", seed, name, "24G", GROUP)
        plan["exports"]["T_MAX"] = "10050000"
        plan["exports"]["TEST_INTERVAL"] = "10000"
        plan["sbatch_args"] = [
            "--time=2-00:00:00" if arg.startswith("--time=") else arg
            for arg in plan["sbatch_args"]
        ]
        plans.append(plan)
    return plans


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    runtime = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    ))
    plans = build_plans(repo)
    paths = guard_and_route(plans, runtime)
    validate_config_keys(repo, plans)
    for plan in plans:
        print("{}: map={} seed={} t_max={} time=2 days memory={} model={}".format(
            plan["job_name"], plan["map_name"], plan["seed"],
            plan["exports"]["T_MAX"], plan["memory"], plan["exports"]["MODEL_TYPE"],
        ))
    if os.environ.get("SUBMIT") != "YES":
        print("Plan only: {} fresh runs; set SUBMIT=YES to submit.".format(len(plans)))
        return
    free_gib = home_quota_free_gib()
    if free_gib < 1.0:
        raise RuntimeError("Only {:.2f} GiB free in /home; require at least 1 GiB "
                           "for scalar-focused runs (not a capacity guarantee)".format(free_gib))
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))
    os.chdir(repo)
    for script in SMOKE_SCRIPTS:
        subprocess.run([sys.executable, script], check=True)
    with (paths["logs"] / ("." + GROUP + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        user = run(["id", "-un"])
        active = active_jobs(user, repo, {p["job_name"] for p in plans})
        completed = completed_jobs(user)
        retained = {
            p["job_name"]: active.get(p["job_name"], completed.get(p["job_name"]))
            for p in plans if p["job_name"] in active or p["job_name"] in completed
        }
        missing = [p for p in plans if p["job_name"] not in retained]
        train_script = "scripts/ozstar_train_offline.sbatch"
        for plan in missing:
            subprocess.run(["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
                           env=dict(os.environ, **plan["exports"]), check=True)
        manifest = paths["logs"] / "{}_{}_{}.json".format(
            GROUP, time.strftime("%Y%m%d_%H%M%S"), os.getpid())
        record = dict(commit=run(["git", "rev-parse", "HEAD"]), group=GROUP,
                      plans=plans, retained=retained, submitted={})
        # Preserve the submitter itself even if copied to a checkout without a commit.
        record["submitter_source"] = SUITE_FILE.read_text()

        def persist():
            with manifest.open("w") as handle:
                json.dump(record, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

        persist()
        for plan in missing:
            job_id = run(["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
                         env=dict(os.environ, **plan["exports"])).split(";", 1)[0]
            record["submitted"][plan["job_name"]] = job_id
            persist()
            print("Submitted {} job={}".format(plan["job_name"], job_id), flush=True)
        print("Retained: " + str(retained))
        print("Manifest: " + str(manifest))


if __name__ == "__main__":
    main()
