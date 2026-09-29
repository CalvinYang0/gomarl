#!/usr/bin/env python3
"""Complete four-map, three-seed results for two Linear single-head models.

Existing seed-one runs are deliberately not duplicated:
* Linear baseline: submit seeds 2/3 on all four maps (8 jobs).
* Linear BayesG KL80: Counter already has seed 1, so submit Counter seeds
  2/3 and seeds 1/2/3 on Pass, 5m6m and MMM2 (11 jobs).
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
    SCENES,
    _load_profiles,
    _plan,
)
from ozstar_submit_relation_advantage_mixer_nine import (  # noqa: E402
    route_runtime,
)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for scene_key, map_name, domain, memory in SCENES:
        for seed in (2, 3):
            name = "{}_paper_linear_singlehead_5m_s{}".format(
                scene_key, seed
            )
            plans.append(_plan(
                repo, profiles, scene_key, map_name, domain,
                "linear_baseline", seed, name, memory,
                "paper_linear_singlehead_5m_3seeds",
            ))

        kl_seeds = (2, 3) if scene_key == "grf_counter" else (1, 2, 3)
        for seed in kl_seeds:
            name = "{}_linear_kl80_direct_5m_s{}_controlled15".format(
                scene_key, seed
            )
            plans.append(_plan(
                repo, profiles, scene_key, map_name, domain,
                "linear_bayesg_kl80_keep", seed, name, memory,
                "paper_linear_kl80_direct_5m_3seeds",
            ))

    if len(plans) != 19:
        raise RuntimeError("Expected exactly 19 missing-seed jobs")
    if len({plan["job_name"] for plan in plans}) != len(plans):
        raise RuntimeError("Duplicate job name in two-model completion suite")
    return plans


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

    manifest = paths["logs"] / (
        "linear_two_models_missing_seeds_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "runtime_root": str(runtime_root),
        "expected_new_jobs": 19,
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
