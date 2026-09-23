#!/usr/bin/env python3
"""Submit selected 5M paper baselines on four selected scenes.

The suite is intentionally fixed to seeds 1, 2, and 3.  Every run evaluates
32 episodes every 10k environment steps, so the complete test win-rate curve
is retained in the offline W&B run.  Re-running the script retains an active
same-name job from this checkout and never cancels unrelated work.
"""

import json
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_counter_transformer_nine import run
from ozstar_submit_relation_advantage_mixer_nine import route_runtime


SEEDS = (1, 2, 3)
SCENES = (
    ("grf_counter", "academy_counterattack_easy", "grf"),
    ("grf_pass", "academy_pass_and_shoot_with_keeper", "grf"),
    ("smac_5m6m", "5m_vs_6m", "smac"),
    ("smac_mmm2", "MMM2", "smac"),
)
DEFAULT_METHODS = ("qmix", "id_hypernet", "obs_hypernet")
SUPPORTED_METHODS = DEFAULT_METHODS + ("vdn",)


def _load_profiles(repo):
    spec = importlib.util.spec_from_file_location(
        "paper_baseline_profiles",
        repo / "src/modules/agents/counter_transformer_suite.py",
    )
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    return profiles


def _extra_args(profiles, profile_label, method):
    # Use the same optimization/evaluation settings for both baselines.  The
    # only model difference is the individual action-value network.
    overrides = profiles.experiment_overrides(profile_label)
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
    if method == "vdn":
        overrides["mixer"] = "vdn"
    return " ".join("{}={}".format(key, value)
                    for key, value in overrides.items())


def build_plans(repo):
    profiles = _load_profiles(repo)
    methods = tuple(os.environ.get(
        "METHODS", " ".join(DEFAULT_METHODS)
    ).split())
    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - set(SUPPORTED_METHODS)
    ):
        raise ValueError(
            "METHODS must select unique values from "
            + " ".join(SUPPORTED_METHODS)
        )
    plans = []
    # Model-major ordering is intentional: submit all QMIX jobs first, then
    # all ID-HyperNet jobs, and finally all Obs-HyperNet jobs.
    for method in methods:
        for scene_key, map_name, domain in SCENES:
            profile_label = (
                "hyper_hypermarl_id"
                if method == "id_hypernet" else "baseline"
            )
            model_type = "qmix_minimal" if method in {"qmix", "vdn"} else (
                profiles.model_type_for(profile_label, domain)
            )
            fixed_head = method in {"qmix", "vdn"}
            walltime = "2-00:00:00" if fixed_head else "3-00:00:00"
            memory = (
                "24G" if domain == "grf" and fixed_head
                else "48G" if domain == "grf"
                else "48G" if fixed_head
                else "96G"
            )
            for seed in SEEDS:
                job_name = "{}_paper_{}_5m_s{}".format(
                    scene_key, method, seed
                )
                run_name = job_name
                exports = {
                    "REPO_DIR": str(repo),
                    "PYTHON_BIN": sys.executable,
                    "CONFIG": "clean_hyper",
                    "ENV_CONFIG": (
                        "sc2" if domain == "smac" else map_name
                    ),
                    "MAP_NAME": map_name,
                    "MODEL_TYPE": model_type,
                    "SEED": str(seed),
                    "RUN_NAME": run_name,
                    "GROUP_NAME": "paper_main_baselines_5m_3seeds",
                    # Run past the 5M evaluation boundary, then stop.
                    "T_MAX": "5050000",
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
                    "EXTRA_ARGS": _extra_args(
                        profiles, profile_label, method
                    ) + (
                        " env_args.write_video=False"
                        if domain == "grf" else ""
                    ),
                }
                sbatch_args = [
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=28",
                    "--mem=" + memory,
                    "--time=" + walltime,
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
                    "method": method,
                    "seed": seed,
                    "job_name": job_name,
                    "run_name": run_name,
                    "exports": exports,
                    "sbatch_args": sbatch_args,
                })
    expected_count = len(methods) * len(SCENES) * len(SEEDS)
    if (
        len(plans) != expected_count
        or len({p["job_name"] for p in plans}) != expected_count
    ):
        raise RuntimeError(
            "Expected {} unique paper-baseline jobs".format(expected_count)
        )
    return plans


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    ))
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
    selected_methods = {plan["method"] for plan in plans}
    if "id_hypernet" in selected_methods:
        subprocess.run(
            [sys.executable, "scripts/smoke_test_id_hypernet_smac.py"],
            check=True,
        )
    if "vdn" in selected_methods:
        subprocess.run(
            [sys.executable, "scripts/smoke_test_vdn_baseline.py"],
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
        "paper_baselines_3seeds_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "seeds": list(SEEDS),
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
    print("Seeds: 1, 2, 3")
    print("Manifest: " + str(manifest))


if __name__ == "__main__":
    main()
