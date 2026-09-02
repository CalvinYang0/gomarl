#!/usr/bin/env python3
"""Preflight, precisely cancel old repository jobs, and submit the nine models.

DRY_RUN=YES prints the complete plan without Slurm calls or filesystem writes.
CANCEL_OLD=YES is required to cancel existing jobs. Existing suite jobs are
retained, so rerunning after an interrupted submission does not duplicate them.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def run(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    os.chdir(repo)
    spec = importlib.util.spec_from_file_location("nine_profiles", repo / "src/modules/agents/counter_transformer_suite.py")
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    seed = int(os.environ.get("SEED", "1"))
    suffix = os.environ.get("RUN_SUFFIX", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]*", suffix):
        raise ValueError("RUN_SUFFIX must contain only letters, digits, _ or -")
    plans = []
    for label in profiles.PROFILES:
        job_name = "grf_counter_trans9_{}_s{}{}".format(label, seed, suffix)
        run_name = "grf_counter_trans9_{}_10m_s{}{}".format(label, seed, suffix)
        extra = profiles.experiment_overrides(label)
        model = extra.pop("clean_model_type")
        extra.update(torch_num_threads=int(os.environ.get("CPUS_PER_TASK", "28")),
                     torch_num_interop_threads=1, learner_updates_per_collect=1,
                     env_worker_startup_stagger=0.25, env_worker_reset_retries=5,
                     env_worker_reset_retry_delay=2.0, env_worker_response_timeout=180.0,
                     save_model=True, save_model_interval=1000000,
                     wandb_team="hjh331-sjtu", wandb_project="gomarl")
        exports = dict(REPO_DIR=str(repo), PYTHON_BIN=sys.executable,
                       CONFIG="clean_hyper", ENV_CONFIG="academy_counterattack_easy",
                       MAP_NAME="academy_counterattack_easy", MODEL_TYPE=model,
                       SEED=str(seed), RUN_NAME=run_name, GROUP_NAME=run_name,
                       T_MAX=os.environ.get("T_MAX", "10050000"),
                       TEST_INTERVAL="50000", BATCH_SIZE_RUN="8", EXPECTED_BATCH_SIZE_RUN="8",
                       BATCH_SIZE="128", BUFFER_SIZE="5000", USE_WANDB="True",
                       WANDB_MODE="offline", USE_CUDA="False",
                       OMP_NUM_THREADS=str(extra["torch_num_threads"]),
                       MKL_NUM_THREADS=str(extra["torch_num_threads"]),
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                       EXTRA_ARGS=" ".join("{}={}".format(k, v) for k, v in extra.items()))
        args = ["--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(extra["torch_num_threads"]),
                "--mem=" + os.environ.get("MEMORY", "24G"),
                "--time=" + os.environ.get("TIME", "2-00:00:00"),
                "--job-name=" + job_name, "--chdir=" + str(repo),
                "--output=ozstar_logs/%x_%j.out", "--error=ozstar_logs/%x_%j.err", "--export=ALL"]
        plans.append(dict(label=label, job_name=job_name, run_name=run_name,
                          exports=exports, sbatch_args=args))
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return
    # No cancellation before all nine execute real learner updates and plots.
    subprocess.run([sys.executable, "scripts/smoke_test_counter_transformer_nine.py"], check=True)
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    manifest = logdir / ("transformer_nine_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    user = run(["id", "-un"])
    active = run(["squeue", "-u", user, "-h", "-o", "%i|%j|%T"])
    names = {p["job_name"] for p in plans}
    retained, cancel = {}, []
    for row in active.splitlines():
        job_id, name, state = row.split("|", 2)
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            continue  # Never cancel other projects or unknown job origins.
        if name in names:
            if name in retained:
                raise RuntimeError("Duplicate active suite job: " + name)
            retained[name] = job_id
        else:
            cancel.append(dict(job_id=job_id, name=name, state=state))
    if cancel and os.environ.get("CANCEL_OLD") != "YES":
        raise RuntimeError("Old repo jobs found. Set CANCEL_OLD=YES to replace them; nothing cancelled.")
    for plan in plans:
        if plan["job_name"] not in retained:
            subprocess.run(["sbatch", "--test-only"] + plan["sbatch_args"] +
                           ["scripts/ozstar_train_offline.sbatch"],
                           env=dict(os.environ, **plan["exports"]), check=True)
    record = dict(commit=run(["git", "rev-parse", "HEAD"]), plans=plans,
                  cancel=cancel, retained=retained, submitted={})
    def persist():
        with manifest.open("w") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())  # Fail on quota BEFORE cancelling.
    persist()
    if cancel:
        subprocess.run(["scancel"] + [j["job_id"] for j in cancel], check=True)
        print("Cancelled captured old jobs: " + ", ".join(j["job_id"] for j in cancel), flush=True)
    for plan in plans:
        if plan["job_name"] in retained:
            print("Retained " + plan["job_name"] + " job=" + retained[plan["job_name"]], flush=True)
            continue
        result = run(["sbatch", "--parsable"] + plan["sbatch_args"] +
                     ["scripts/ozstar_train_offline.sbatch"], env=dict(os.environ, **plan["exports"]))
        record["submitted"][plan["job_name"]] = result.split(";", 1)[0]
        persist()
        print("Submitted " + plan["job_name"] + " job=" + result, flush=True)
    print("Manifest: " + str(manifest))
    print(run(["squeue", "-u", user, "-o", "%.18i %.90j %.10T %.12M %.10m %R"]))


if __name__ == "__main__":
    main()
