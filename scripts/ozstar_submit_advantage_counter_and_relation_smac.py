#!/usr/bin/env python3
"""Atomically preflight and submit five Counter plus two SMAC experiments."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import (
    build_plans as build_counter_plans,
    run,
)
from ozstar_submit_relation_advantage_mixer_nine import route_runtime
from ozstar_submit_trans9_multiscene import build_plans as build_scene_plans


COUNTER_LABELS = (
    "relation_advantage_margin_kl80aux",
    "relation_advantage_margin_kl80aux_gradsep",
    "relation_advantage_weighted_kl80aux",
    "relation_advantage_weighted_kl80aux_gradsep",
    "relation_kl80aux_kltd_gateonly",
)
SMAC_LABEL = "relation_all4_dualtest"
SMAC_SCENES = ("3s5z", "corridor")
SMOKE_TESTS = (
    "scripts/smoke_test_advantage_margin_gate.py",
    "scripts/smoke_test_relation_kl80aux_kltd_gateonly.py",
    "scripts/smoke_test_relation_all4_dualtest_multiscene.py",
)


def build_selected_plans(repo):
    preserved = {
        key: os.environ.get(key)
        for key in ("RUN_SUFFIX", "SCENES", "LABELS")
    }
    try:
        os.environ["RUN_SUFFIX"] = "_home1_advw"
        counter = build_counter_plans(repo, COUNTER_LABELS)

        os.environ["RUN_SUFFIX"] = "_home1_dual"
        os.environ["SCENES"] = " ".join(SMAC_SCENES)
        os.environ["LABELS"] = SMAC_LABEL
        smac = build_scene_plans(repo)
    finally:
        for key, value in preserved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    plans = counter + smac
    expected = (
        tuple(("counter", label) for label in COUNTER_LABELS)
        + tuple((scene, SMAC_LABEL) for scene in SMAC_SCENES)
    )
    actual = tuple(
        (plan.get("scene", "counter"), plan["label"])
        for plan in plans
    )
    if actual != expected or len(plans) != 7:
        raise RuntimeError("Seven-job selection changed: {}".format(actual))
    if len({plan["job_name"] for plan in plans}) != len(plans):
        raise RuntimeError("Seven-job plan contains duplicate names")
    return plans


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    ))
    plans = build_selected_plans(repo)
    paths = route_runtime(plans, runtime_root)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    allowed_roots = ("/home/kyang/", "/fred/oz501/kyang/")
    if not runtime_root.is_absolute() or not any(
        str(runtime_root).startswith(root) for root in allowed_roots
    ):
        raise RuntimeError(
            "Runtime output must be below /home/kyang or /fred/oz501/kyang"
        )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))

    os.chdir(repo)
    failures = []
    for index, smoke_test in enumerate(SMOKE_TESTS, start=1):
        print(
            "[smoke {}/{}] START {}".format(
                index, len(SMOKE_TESTS), smoke_test
            ),
            flush=True,
        )
        result = subprocess.run([sys.executable, smoke_test])
        if result.returncode:
            failures.append((smoke_test, result.returncode))
            print(
                "[smoke {}/{}] FAILED {} (exit={})".format(
                    index, len(SMOKE_TESTS), smoke_test, result.returncode
                ),
                flush=True,
            )
        else:
            print(
                "[smoke {}/{}] PASSED {}".format(
                    index, len(SMOKE_TESTS), smoke_test
                ),
                flush=True,
            )
    if failures:
        raise RuntimeError(
            "All smoke tests completed; no jobs submitted: "
            + ", ".join(
                "{}:exit={}".format(path, code)
                for path, code in failures
            )
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
        "counter5_smac2_{}_{}.json".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
    )
    record = {
        "commit": run(["git", "rev-parse", "HEAD"]),
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
