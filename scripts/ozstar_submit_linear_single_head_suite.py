#!/usr/bin/env python3
"""Submit Linear single-head baselines and controlled Counter ablations.

Baseline: four paper maps, seed 1, 5M steps.
Counter: direct-vs-auxiliary KL80 plus no-KL NoMaskTD-teacher QME/sampling
controls, seed 1 by default. No Attention-only or RPG dual-head job is
constructed.
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


SCENES = (
    ("grf_counter", "academy_counterattack_easy", "grf", "16G"),
    ("grf_pass", "academy_pass_and_shoot_with_keeper", "grf", "16G"),
    ("smac_5m6m", "5m_vs_6m", "smac", "24G"),
    ("smac_mmm2", "MMM2", "smac", "24G"),
)
BASELINE_SEEDS = (1,)
COUNTER_LABELS = (
    "linear_bayesg_kl80_keep",
    "linear_obs_gate_kl80aux_multiply",
    "linear_bayesg_nomasktd_control",
    "linear_qme_action_q",
    "linear_qme_action_q_episode_mean",
    "linear_qme_td_quality",
    "linear_qme_joint_value",
    "linear_qme_action_q_scaled",
    "linear_qme_dynamic_readiness",
    "linear_qme_open_win_readiness",
    "linear_qme_full_behavior",
)
SHORT_NAMES = {
    "linear_bayesg_kl80_keep": "kl80_direct",
    "linear_obs_gate_kl80aux_multiply": "kl80_aux_multiply",
    "linear_bayesg_nomasktd_control": "nomasktd_control_nokl",
    "linear_qme_action_q": "qme_action_q_masked_nokl",
    "linear_qme_action_q_episode_mean": "qme_epmean_masked_nokl",
    "linear_qme_td_quality": "qme_tdquality_masked_nokl",
    "linear_qme_joint_value": "qme_joint_value_masked_nokl",
    "linear_qme_action_q_scaled": "qme_q_scaled_masked_nokl",
    "linear_qme_dynamic_readiness": "qme_dynamic_ready_masked_nokl",
    "linear_qme_open_win_readiness": "qme_openwin_ready_masked_nokl",
    "linear_qme_full_behavior": "qme_action_q_full_nokl",
}


def _load_profiles(repo):
    spec = importlib.util.spec_from_file_location(
        "linear_single_head_profiles",
        repo / "src/modules/agents/counter_transformer_suite.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extra_args(profiles, label, domain):
    overrides = profiles.experiment_overrides(label, domain)
    overrides.pop("clean_model_type")
    overrides.update(
        torch_num_threads=28,
        torch_num_interop_threads=1,
        learner_updates_per_collect=1,
        env_worker_startup_stagger=0.25,
        env_worker_reset_retries=5,
        env_worker_reset_retry_delay=2.0,
        env_worker_response_timeout=180.0,
        env_worker_run_retries=2,
        env_worker_run_retry_delay=2.0,
        test_nepisode=32,
        # Keep scalar and media histories, but do not write model checkpoints.
        save_model=False,
        save_model_at_end=False,
        wandb_save_model=False,
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


def _plan(repo, profiles, scene_key, map_name, domain, label, seed,
          job_name, memory, group):
    model_type = profiles.model_type_for(label, domain)
    if "single_linear" not in model_type:
        raise RuntimeError("Non-Linear model escaped suite: " + model_type)
    exports = {
        "REPO_DIR": str(repo),
        "PYTHON_BIN": sys.executable,
        "CONFIG": "clean_hyper",
        "ENV_CONFIG": "sc2" if domain == "smac" else map_name,
        "MAP_NAME": map_name,
        "MODEL_TYPE": model_type,
        "SEED": str(seed),
        "RUN_NAME": job_name,
        "GROUP_NAME": group,
        "T_MAX": os.environ.get("T_MAX", "5050000"),
        "TEST_INTERVAL": os.environ.get("TEST_INTERVAL", "10000"),
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
        "EXTRA_ARGS": _extra_args(profiles, label, domain),
    }
    return {
        "scene": scene_key,
        "map_name": map_name,
        "domain": domain,
        "label": label,
        "seed": seed,
        "memory": memory,
        "job_name": job_name,
        "run_name": job_name,
        "exports": exports,
        "sbatch_args": [
            "--nodes=1", "--ntasks=1", "--cpus-per-task=28",
            "--mem=" + memory,
            "--time=" + os.environ.get("TIME", "2-00:00:00"),
            "--job-name=" + job_name,
            "--chdir=" + str(repo),
            "--output=ozstar_logs/%x_%j.out",
            "--error=ozstar_logs/%x_%j.err",
            "--export=ALL",
        ],
    }


def build_plans(repo):
    profiles = _load_profiles(repo)
    selected = set(
        os.environ.get("SUITES", "baseline counter_ablation").split()
    )
    if not selected or selected - {"baseline", "counter_ablation"}:
        raise ValueError("SUITES accepts baseline and/or counter_ablation")
    plans = []
    if "baseline" in selected:
        for scene_key, map_name, domain, memory in SCENES:
            for seed in BASELINE_SEEDS:
                name = "{}_paper_linear_singlehead_5m_s{}".format(
                    scene_key, seed
                )
                plans.append(_plan(
                    repo, profiles, scene_key, map_name, domain,
                    "linear_baseline", seed, name, memory,
                    "paper_linear_singlehead_5m_seed1",
                ))
    if "counter_ablation" in selected:
        seed = int(os.environ.get("ABLATION_SEED", "1"))
        for label in COUNTER_LABELS:
            memory = (
                "16G" if label == "linear_bayesg_kl80_keep"
                else "32G" if label == "linear_obs_gate_kl80aux_multiply"
                else "32G"
            )
            name = "grf_counter_linear_{}_5m_s{}_controlled15".format(
                SHORT_NAMES[label], seed
            )
            plans.append(_plan(
                repo, profiles, "grf_counter",
                "academy_counterattack_easy", "grf", label, seed, name,
                memory,
                "counter_linear_kl80_qme_controlled15_s{}".format(seed),
            ))
    if len({plan["job_name"] for plan in plans}) != len(plans):
        raise RuntimeError("Duplicate Linear suite job name")
    return plans


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
            env=dict(os.environ, **plan["exports"]), check=True,
        )

    manifest = paths["logs"] / (
        "linear_single_head_suite_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
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
            print("Retained {} job={}".format(name, retained[name]))
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
