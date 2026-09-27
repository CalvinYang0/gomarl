#!/usr/bin/env python3
"""Submit full HyperSelect on four paper scenes, 10M, seeds 1/2/3.

The suite is fixed to Counter, Pass, 5m6m and MMM2.  It never cancels other
jobs and retains an active same-name job from this checkout.  Test win rates
are evaluated with 32 episodes every 10k environment steps.
"""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_counter_transformer_nine import run
from ozstar_submit_relation_advantage_mixer_nine import route_runtime


PROFILE = "relation_advantage_qvalue_augtd_nomasktd"
SEEDS = (1, 2, 3)
SCENES = (
    ("grf_counter", "academy_counterattack_easy", "grf", "96G"),
    ("grf_pass", "academy_pass_and_shoot_with_keeper", "grf", "48G"),
    ("smac_5m6m", "5m_vs_6m", "smac", "160G"),
    ("smac_mmm2", "MMM2", "smac", "160G"),
)
T_MAX = "10050000"
WALLTIME = "5-00:00:00"


def _load_profiles(repo):
    spec = importlib.util.spec_from_file_location(
        "hyperselect_main_profiles",
        repo / "src/modules/agents/counter_transformer_suite.py",
    )
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    return profiles


def _extra_args(profiles, domain):
    overrides = profiles.experiment_overrides(PROFILE, domain)
    overrides.pop("clean_model_type")
    overrides.update(
        torch_num_threads=28,
        torch_num_interop_threads=1,
        learner_updates_per_collect=1,
        env_worker_startup_stagger=0.25,
        env_worker_reset_retries=5,
        env_worker_reset_retry_delay=2.0,
        env_worker_response_timeout=180.0,
        test_nepisode=32,
        save_model=True,
        save_model_interval=1000000,
        wandb_team="hjh331-sjtu",
        wandb_project="gomarl",
    )
    if domain == "grf":
        overrides["env_args.write_video"] = False
    else:
        overrides["save_battle_trace"] = False
    return " ".join(
        "{}={}".format(key, value) for key, value in overrides.items()
    )


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for scene_key, map_name, domain, memory in SCENES:
        model_type = profiles.model_type_for(PROFILE, domain)
        for seed in SEEDS:
            job_name = "{}_paper_hyperselect_10m_s{}".format(
                scene_key, seed
            )
            exports = {
                "REPO_DIR": str(repo),
                "PYTHON_BIN": sys.executable,
                "CONFIG": "clean_hyper",
                "ENV_CONFIG": "sc2" if domain == "smac" else map_name,
                "MAP_NAME": map_name,
                "MODEL_TYPE": model_type,
                "SEED": str(seed),
                "RUN_NAME": job_name,
                "GROUP_NAME": "hyperselect_paper_main_10m_3seeds",
                "T_MAX": T_MAX,
                "TEST_INTERVAL": "10000",
                "BATCH_SIZE_RUN": "8",
                "EXPECTED_BATCH_SIZE_RUN": "8",
                "BATCH_SIZE": "128",
                "BUFFER_SIZE": "5000",
                "USE_WANDB": "True",
                "WANDB_MODE": "offline",
                "USE_CUDA": "False",
                "OMP_NUM_THREADS": "28",
                "MKL_NUM_THREADS": "28",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "EXTRA_ARGS": _extra_args(profiles, domain),
            }
            sbatch_args = [
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=28",
                "--mem=" + memory,
                "--time=" + WALLTIME,
                "--job-name=" + job_name,
                "--chdir=" + str(repo),
                "--output=ozstar_logs/%x_%j.out",
                "--error=ozstar_logs/%x_%j.err",
                "--export=ALL",
            ]
            plans.append({
                "scene": scene_key,
                "map_name": map_name,
                "domain": domain,
                "seed": seed,
                "memory": memory,
                "job_name": job_name,
                "run_name": job_name,
                "exports": exports,
                "sbatch_args": sbatch_args,
            })
    if len(plans) != 12 or len({p["job_name"] for p in plans}) != 12:
        raise RuntimeError("Expected 12 unique HyperSelect main-result jobs")
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
    for command in (
        [sys.executable,
         "scripts/smoke_test_hyperselect_memory_optimizations.py"],
        [sys.executable, "scripts/smoke_test_hyperselect_paper_scenes.py"],
    ):
        subprocess.run(command, check=True)

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
        "hyperselect_main_10m_3seeds_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "profile": PROFILE,
        "seeds": list(SEEDS),
        "t_max": int(T_MAX),
        "walltime": WALLTIME,
        "test_interval": 10000,
        "test_nepisode": 32,
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

    print("Submitted/retained 12 HyperSelect jobs; seeds: 1, 2, 3")
    print("Manifest: " + str(manifest))
    print(run([
        "squeue", "-u", user, "-o",
        "%.18i %.80j %.10T %.12M %.10l %.10m %R",
    ]))


if __name__ == "__main__":
    main()
