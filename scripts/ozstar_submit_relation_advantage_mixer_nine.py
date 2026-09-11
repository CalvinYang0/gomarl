#!/usr/bin/env python3
"""Submit three relation/mixer methods on Counter, MMM2 and 3s5z.

All mutable runtime output is routed to /fred by default. Existing active jobs
with the same name and repository workdir are retained; nothing is cancelled.
DRY_RUN=YES prints the full nine-job plan without filesystem or Slurm writes.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import (
    build_plans as counter_plans,
    run,
)
from ozstar_submit_trans9_multiscene import build_plans as multiscene_plans


LABELS = (
    "relation_kl80aux_mixer_coef001",
    "relation_advantage_kl80aux_mixer_coef001",
    "relation_advantage_kl80aux",
)
SCENES = ("counter", "mmm2", "3s5z")


def build_selected_plans(repo):
    plans = counter_plans(repo, LABELS)
    for plan in plans:
        plan.update(
            scene="counter",
            map_name="academy_counterattack_easy",
            domain="grf",
        )

    previous = {key: os.environ.get(key) for key in ("SCENES", "LABELS")}
    os.environ.update(SCENES="mmm2 3s5z", LABELS=" ".join(LABELS))
    try:
        plans.extend(multiscene_plans(repo))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    expected = tuple(
        (scene, label) for scene in SCENES for label in LABELS
    )
    actual = tuple((plan["scene"], plan["label"]) for plan in plans)
    if actual != expected:
        raise RuntimeError("Nine-job selection changed: {}".format(actual))
    if len({plan["job_name"] for plan in plans}) != 9:
        raise RuntimeError("Nine-job plan contains duplicate job names")
    return plans


def route_runtime_to_fred(plans, runtime_root):
    if not runtime_root.is_absolute():
        raise ValueError("RUNTIME_ROOT must be an absolute path")
    paths = {
        "wandb": runtime_root / "wandb",
        "wandb_cache": runtime_root / "wandb_cache",
        "wandb_config": runtime_root / "wandb_config",
        "wandb_data": runtime_root / "wandb_data",
        "results": runtime_root / "results",
        "logs": runtime_root / "ozstar_logs",
        "tmp": runtime_root / "tmp",
    }
    for plan in plans:
        exports = plan["exports"]
        exports.update(
            # W&B treats WANDB_DIR as its root and creates the ``wandb``
            # directory beneath it. Keep that generated directory aligned
            # with paths["wandb"] so the sync loop has one exact root.
            WANDB_DIR=str(runtime_root),
            WANDB_CACHE_DIR=str(paths["wandb_cache"]),
            WANDB_CONFIG_DIR=str(paths["wandb_config"]),
            WANDB_DATA_DIR=str(paths["wandb_data"]),
            GOMARL_RESULTS_PATH=str(paths["results"]),
            TMPDIR=str(paths["tmp"]),
        )
        exports["EXTRA_ARGS"] += " local_results_path={}".format(
            paths["results"]
        )
        plan["sbatch_args"] = [
            arg
            for arg in plan["sbatch_args"]
            if not arg.startswith("--output=")
            and not arg.startswith("--error=")
        ] + [
            "--output=" + str(paths["logs"] / "%x_%j.out"),
            "--error=" + str(paths["logs"] / "%x_%j.err"),
        ]
        plan["runtime_root"] = str(runtime_root)
    return paths


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT",
        "/fred/oz501/kyang/gomarl-runtime/gomarl-dual-branch",
    ))
    plans = build_selected_plans(repo)
    paths = route_runtime_to_fred(plans, runtime_root)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    if not str(runtime_root).startswith("/fred/"):
        raise RuntimeError(
            "Refusing OzSTAR run outside /fred; set an explicit /fred RUNTIME_ROOT"
        )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))

    os.chdir(repo)
    smoke_commands = (
        [sys.executable,
         "scripts/smoke_test_relation_advantage_mixer_coef001.py"],
        [sys.executable, "scripts/smoke_test_trans9_multiscene.py"],
    )
    smoke_environments = (
        None,
        dict(os.environ, SCENES="mmm2 3s5z", LABELS=" ".join(LABELS)),
    )
    failures = []
    for index, (command, environment) in enumerate(
        zip(smoke_commands, smoke_environments), start=1
    ):
        print(
            "[smoke {}/2] START {}".format(index, " ".join(command)),
            flush=True,
        )
        result = subprocess.run(command, env=environment)
        if result.returncode:
            failures.append((" ".join(command), result.returncode))
            print("[smoke {}/2] FAILED exit={}".format(
                index, result.returncode
            ), flush=True)
        else:
            print("[smoke {}/2] PASSED".format(index), flush=True)
    if failures:
        raise RuntimeError(
            "All smoke tests ran; no jobs submitted: "
            + ", ".join("{}:exit={}".format(*item) for item in failures)
        )

    user = run(["id", "-un"])
    names = {plan["job_name"] for plan in plans}
    retained = {}
    for row in run(["squeue", "-u", user, "-h", "-o", "%i|%j"]).splitlines():
        job_id, name = row.split("|", 1)
        if name not in names:
            continue
        info = run(["scontrol", "show", "job", "-o", job_id])
        match = re.search(r"(?:^|\s)WorkDir=(\S+)", info)
        if not match or Path(match.group(1)).resolve() != repo:
            raise RuntimeError(
                "Same-name job has a different/unknown WorkDir: " + name
            )
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
        "relation_advantage_mixer_nine_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "worktree_status": run(["git", "status", "--short"]),
        "runtime_root": str(runtime_root),
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
