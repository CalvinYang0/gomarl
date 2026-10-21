#!/usr/bin/env python3
"""Six fresh, pinned PyMARL2 QMIX runs; dry-run unless SUBMIT=YES.

No legacy profile inheritance, cancellations, historical renaming or recovery
from partial runs. All missing jobs pass Slurm preflight before submission.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ozstar_submit_counter_transformer_nine import run
from ozstar_submit_linear_directkl_four_model_3seeds import active_jobs, completed_jobs
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import (
    home_quota_free_gib, validate_config_keys,
)

GROUP = "official_pymarl2_qmix_10m_6jobs_corrected20261021"
COMMIT = "8ccac7c5aa134422a2e3009be735d23cdf8ce2f8"
MAPS = (("smac_5m6m", "5m_vs_6m", 5, 6), ("smac_8m9m", "8m_vs_9m", 8, 9))
TRAIN_SCRIPT = "scripts/ozstar_train_official_qmix.sbatch"


def build_plans(repo):
    plans = []
    extra = dict(
        torch_num_threads=28, torch_num_interop_threads=1,
        learner_updates_per_collect=1, env_worker_startup_stagger=0.25,
        env_worker_reset_retries=5, env_worker_reset_retry_delay=2.0,
        env_worker_response_timeout=180.0, env_worker_run_retries=2,
        env_worker_run_retry_delay=2.0, test_nepisode=32,
        test_greedy=True, save_model=False, save_model_at_end=False,
        wandb_save_model=False, wandb_team="hjh331-sjtu", wandb_project="gomarl",
    )
    for scene, map_name, _, _ in MAPS:
        for seed in (1, 2, 3):
            name = "{}_qmix_official_pymarl2_10m_s{}_corrected20261021".format(scene, seed)
            exports = dict(
                REPO_DIR=str(repo), PYTHON_BIN=sys.executable,
                CONFIG="official_pymarl2_qmix", ENV_CONFIG="sc2", MAP_NAME=map_name,
                MODEL_TYPE="official_pymarl2_rnn", SEED=str(seed), RUN_NAME=name,
                GROUP_NAME=GROUP, T_MAX="10050000", TEST_INTERVAL="10000",
                BATCH_SIZE_RUN="8", EXPECTED_BATCH_SIZE_RUN="8", BATCH_SIZE="128",
                BUFFER_SIZE="5000", USE_WANDB="True", WANDB_MODE="offline",
                USE_CUDA="False", OMP_NUM_THREADS="28", MKL_NUM_THREADS="28",
                OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                EXTRA_ARGS=" ".join("{}={}".format(k, v) for k, v in extra.items()),
            )
            plans.append(dict(
                scene=scene, map_name=map_name, domain="smac", seed=seed,
                label="qmix_official_pymarl2", job_name=name, run_name=name,
                memory="24G", upstream_commit=COMMIT, exports=exports,
                inventory_note="Fresh official PyMARL2 model and learner; not old MLP-head QMIX",
                sbatch_args=["--nodes=1", "--ntasks=1", "--cpus-per-task=28",
                             "--mem=24G", "--time=2-00:00:00", "--job-name=" + name,
                             "--chdir=" + str(repo), "--export=ALL"],
            ))
    assert len(plans) == len({p["job_name"] for p in plans}) == 6
    return plans


def route_runtime(plans, root):
    root = root.resolve()
    if Path("/home/kyang").resolve() not in root.parents:
        raise ValueError("Runtime root must be a subdirectory of /home/kyang")
    paths = {k: root / v for k, v in dict(
        logs="ozstar_logs", results="results", wandb="wandb", tmp="tmp",
        wandb_cache="wandb_cache", wandb_config="wandb_config", wandb_data="wandb_data",
        xdg_cache="xdg_cache", matplotlib="matplotlib",
    ).items()}
    env_paths = dict(WANDB_DIR="wandb", WANDB_CACHE_DIR="wandb_cache",
                     WANDB_CONFIG_DIR="wandb_config", WANDB_DATA_DIR="wandb_data",
                     GOMARL_RESULTS_PATH="results", TMPDIR="tmp",
                     XDG_CACHE_HOME="xdg_cache", MPLCONFIGDIR="matplotlib")
    for p in plans:
        p["exports"].update({k: str(paths[v]) for k, v in env_paths.items()})
        p["exports"].update(GOMARL_SACRED_CAPTURE_MODE="no", PYTHONDONTWRITEBYTECODE="1")
        p["exports"]["EXTRA_ARGS"] += " local_results_path=" + shlex.quote(str(paths["results"]))
        p["sbatch_args"] += ["--output=" + str(paths["logs"] / "%x_%j.out"),
                             "--error=" + str(paths["logs"] / "%x_%j.err")]
        p["runtime_root"] = str(root)
    return paths


def validate_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    for _, name, allies, enemies in MAPS:
        actual = registry.get(name, {})
        if (actual.get("n_agents"), actual.get("n_enemies"), actual.get("map_type")) != (allies, enemies, "marines"):
            raise RuntimeError("Unexpected installed map definition: " + name)
        if not any(maps_root.rglob(name + ".SC2Map")):
            raise RuntimeError("Missing SC2 map file: " + name)


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    runtime = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    plans = build_plans(repo)
    paths = route_runtime(plans, runtime)
    validate_config_keys(repo, plans)
    for p in plans:
        print("{}: map={} seed={} official={} 10M CPU28/24G/48h".format(
            p["job_name"], p["map_name"], p["seed"], COMMIT[:12]))
    if os.environ.get("SUBMIT") != "YES":
        print("Plan only: six new runs; no jobs submitted. Set SUBMIT=YES to submit.")
        return
    if run(["id", "-un"]) != "kyang":
        raise RuntimeError("This submitter is scoped to OzSTAR user kyang")
    if home_quota_free_gib() < 2.0:
        raise RuntimeError("Require 2 GiB free in /home; no jobs submitted")
    validate_maps()
    subprocess.run([sys.executable, str(repo / "scripts/smoke_test_official_qmix.py")], check=True, cwd=repo)
    subprocess.run([sys.executable, str(repo / "scripts/smoke_test_submit_official_qmix.py")], check=True, cwd=repo)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime path not writable: " + str(path))
    os.chdir(repo)
    with (paths["logs"] / ("." + GROUP + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = {p["job_name"] for p in plans}
        active = active_jobs("kyang", repo, names)
        completed = completed_jobs("kyang")
        retained = {n: active.get(n, completed.get(n)) for n in names if n in active or n in completed}
        missing = [p for p in plans if p["job_name"] not in retained]
        for p in missing:
            subprocess.run(["sbatch", "--test-only"] + p["sbatch_args"] + [TRAIN_SCRIPT],
                           env=dict(os.environ, **p["exports"]), check=True)
        manifest = paths["logs"] / (GROUP + time.strftime("_%Y%m%d_%H%M%S_") +
                                    str(os.getpid()) + "_" + str(time.time_ns()) + ".json")
        record = dict(commit=run(["git", "rev-parse", "HEAD"]), group=GROUP,
                      upstream_commit=COMMIT, plans=plans, retained=retained, submitted={},
                      provenance=json.loads((repo / "src/third_party/pymarl2_qmix/provenance.json").read_text()),
                      config_sha256=hashlib.sha256((repo / "src/config/algs/official_pymarl2_qmix.yaml").read_bytes()).hexdigest())

        def persist():
            with manifest.open("w") as handle:
                json.dump(record, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

        persist()
        for p in missing:
            job = run(["sbatch", "--parsable"] + p["sbatch_args"] + [TRAIN_SCRIPT],
                      env=dict(os.environ, **p["exports"])).split(";", 1)[0]
            record["submitted"][p["job_name"]] = job
            persist()
            print("Submitted {} job={}".format(p["job_name"], job), flush=True)
        print("Retained: " + str(retained))
        print("Manifest: " + str(manifest))


if __name__ == "__main__":
    main()
