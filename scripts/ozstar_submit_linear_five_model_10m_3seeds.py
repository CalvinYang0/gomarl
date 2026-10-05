#!/usr/bin/env python3
"""Submit five matched single-head Linear conditions, four maps, three seeds.

All 60 runs are new 10M runs: no 5M history is reused. Run without
SUBMIT=YES for a read-only plan summary. SUBMIT=YES performs smoke and Slurm
preflight checks for every missing job before submitting any of them.
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

from ozstar_submit_counter_transformer_nine import run  # noqa: E402
from ozstar_submit_linear_directkl_four_model_3seeds import (  # noqa: E402
    active_jobs, completed_jobs,
)
from ozstar_submit_linear_single_head_suite import (  # noqa: E402
    SCENES, _load_profiles, _plan,
)
from ozstar_submit_relation_advantage_mixer_nine import route_runtime  # noqa: E402


GROUP = "linear_five_model_10m_3seeds_home2d"
MODELS = (
    ("linear_bayesg_kl80_keep", "directkl"),
    ("linear_obs_gate_kl80aux_multiply", "auxmul"),
    ("linear_auxmultiply_qme_action_q_episode_mean", "auxmul_epmean"),
    ("linear_directkl_qme_action_q_episode_mean", "directkl_epmean"),
    ("linear_baseline", "singlehead_baseline"),
)
QME_LABELS = {MODELS[2][0], MODELS[3][0]}
EXPECTED_RUNS = len(SCENES) * len(MODELS) * 3
MIN_HOME_FREE_GIB = 5.0


def home_quota_free_gib():
    """Read the per-user /home quota, not filesystem-wide free space."""
    result = subprocess.run(
        ["quota", "-s"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] != "/home":
            continue

        def kib(token):
            token = token.rstrip("*")
            suffix = token[-1].upper()
            if suffix in "KMGTP":
                return float(token[:-1]) * (1024 ** "KMGTP".index(suffix))
            return float(token)

        used_kib, limit_kib = kib(fields[1]), kib(fields[3])
        if limit_kib <= 0:
            raise RuntimeError("/home has no readable hard quota: " + line)
        return (limit_kib - used_kib) / (1024 ** 2)
    raise RuntimeError("Could not verify /home quota; refusing 60 offline runs: " + result.stdout)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for scene, map_name, domain, baseline_memory in SCENES:
        for label, suffix in MODELS:
            flags = profiles.ALL_PROFILES[label]
            assert flags.get("branch") == "linear"
            assert bool(flags.get("gate")) == (label != "linear_baseline")
            assert bool(flags.get("kl")) == (
                label.startswith("linear_directkl_qme_")
                or label == "linear_bayesg_kl80_keep"
            )
            assert bool(flags.get("aux")) == (
                label.startswith("linear_auxmultiply_qme_")
                or label == "linear_obs_gate_kl80aux_multiply"
            )
            if label in QME_LABELS:
                assert flags.get("nomask_td_coef") == 1.0
                assert flags.get("advantage_objective") == "action_q_episode_mean"
            for seed in (1, 2, 3):
                name = "{}_{}_10m_s{}_home2d".format(scene, suffix, seed)
                memory = "32G" if label in QME_LABELS or label == "linear_obs_gate_kl80aux_multiply" else baseline_memory
                plan = _plan(repo, profiles, scene, map_name, domain, label,
                             seed, name, memory, GROUP)
                plan["exports"]["T_MAX"] = "10050000"
                plan["sbatch_args"] = [
                    "--time=2-00:00:00"
                    if arg.startswith("--time=") else arg
                    for arg in plan["sbatch_args"]
                ]
                plans.append(plan)
    if len(plans) != EXPECTED_RUNS or len({p["job_name"] for p in plans}) != EXPECTED_RUNS:
        raise RuntimeError("Expected {} unique 10M jobs".format(EXPECTED_RUNS))
    return plans


def guard_and_route(plans, runtime_root):
    expected_root = Path("/home/kyang")
    if runtime_root != expected_root and expected_root not in runtime_root.parents:
        raise RuntimeError("10M runtime must be under /home/kyang")
    if os.environ.get("MEDIA_INTERVAL", "100000") != "100000":
        raise RuntimeError("This suite fixes image intervals at 100000 steps")
    paths = route_runtime(plans, runtime_root)
    paths["xdg_cache"] = runtime_root / "xdg_cache"
    paths["matplotlib"] = runtime_root / "matplotlib"
    paths["football_dumps"] = runtime_root / "football_dumps"
    for plan in plans:
        exports = plan["exports"]
        exports.update(
            GOMARL_SACRED_CAPTURE_MODE="no",
            XDG_CACHE_HOME=str(paths["xdg_cache"]),
            MPLCONFIGDIR=str(paths["matplotlib"]),
            PYTHONDONTWRITEBYTECODE="1",
        )
        if plan["domain"] == "grf":
            exports["EXTRA_ARGS"] += " env_args.logdir={}".format(paths["football_dumps"])
        extras = exports["EXTRA_ARGS"]
        for required in (
            "save_model=False", "save_model_at_end=False",
            "wandb_save_model=False", "wandb_media_interval=100000",
            "clean_train_gate_image_interval=100000",
        ):
            if required not in extras:
                raise RuntimeError("Missing disk-control override: " + required)
        for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
                    "WANDB_DATA_DIR", "GOMARL_RESULTS_PATH", "TMPDIR",
                    "XDG_CACHE_HOME", "MPLCONFIGDIR"):
            if expected_root not in Path(exports[key]).parents:
                raise RuntimeError("Runtime path escaped /home: " + key)
        if "wandb_test_parameter_pca=True" not in exports["EXTRA_ARGS"]:
            raise RuntimeError("Expected PCA override is missing")
        exports["EXTRA_ARGS"] = exports["EXTRA_ARGS"].replace(
            "wandb_test_parameter_pca=True",
            "wandb_test_parameter_pca=False",
        )
        exports["EXTRA_ARGS"] += " wandb_test_gate_trajectory=False"
    return paths


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    )).resolve()
    plans = build_plans(repo)
    paths = guard_and_route(plans, runtime_root)
    for plan in plans:
        print("{} {} {} {} {}".format(
            plan["job_name"], plan["exports"]["T_MAX"],
            next(arg for arg in plan["sbatch_args"] if arg.startswith("--time=")),
            plan["memory"], paths["logs"]
        ))
    if os.environ.get("SUBMIT") != "YES":
        print("Plan only: {} jobs. Set SUBMIT=YES to validate and submit.".format(EXPECTED_RUNS))
        return

    free_gib = home_quota_free_gib()
    print("/home quota free: {:.2f} GiB".format(free_gib))
    if free_gib < MIN_HOME_FREE_GIB:
        raise RuntimeError(
            "Only {:.2f} GiB free in /home; at least {:.1f} GiB required "
            "before submitting 60 offline runs (not a guarantee against "
            "later quota exhaustion)".format(free_gib, MIN_HOME_FREE_GIB)
        )

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))
    os.chdir(repo)
    subprocess.run([sys.executable, "scripts/smoke_test_linear_single_head_suite.py"], check=True)
    with (paths["logs"] / ".linear_five_model_10m.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        user = run(["id", "-un"])
        names = {p["job_name"] for p in plans}
        previous_names = {name.replace("_home2d", "_fiveway") for name in names}
        previous_active = active_jobs(user, repo, previous_names)
        if previous_active:
            raise RuntimeError(
                "Earlier fiveway jobs are still active; inspect/cancel only "
                "those jobs before starting a differently configured rerun: "
                + str(previous_active)
            )
        completed = completed_jobs(user)
        active = active_jobs(user, repo, names)
        retained = {
            p["job_name"]: ("active", active[p["job_name"]])
            if p["job_name"] in active else ("completed", completed[p["job_name"]])
            for p in plans if p["job_name"] in active or p["job_name"] in completed
        }
        missing = [p for p in plans if p["job_name"] not in retained]
        train_script = "scripts/ozstar_train_offline.sbatch"
        for plan in missing:
            subprocess.run(
                ["sbatch", "--test-only"] + plan["sbatch_args"] + [train_script],
                env=dict(os.environ, **plan["exports"]), check=True,
            )
        manifest = paths["logs"] / "linear_five_model_10m_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S")
        )
        record = {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "group": GROUP, "expected_runs": EXPECTED_RUNS,
            "retained": retained, "submitted": {}, "plans": plans,
        }
        def persist():
            with manifest.open("w") as handle:
                json.dump(record, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
        persist()
        print("Preflight passed; retained {}, submitting {}.".format(len(retained), len(missing)))
        for plan in missing:
            job_id = run(
                ["sbatch", "--parsable"] + plan["sbatch_args"] + [train_script],
                env=dict(os.environ, **plan["exports"]),
            ).split(";", 1)[0]
            record["submitted"][plan["job_name"]] = job_id
            persist()
            print("Submitted {} job={}".format(plan["job_name"], job_id), flush=True)
        print("Manifest: " + str(manifest))


if __name__ == "__main__":
    main()
