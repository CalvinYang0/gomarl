#!/usr/bin/env python3
"""Preflight and submit Direct-Q + AugTD + NoMaskTD on two SMAC maps."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import run
from ozstar_submit_relation_advantage_mixer_nine import route_runtime
from ozstar_submit_trans9_multiscene import build_plans


LABEL = "relation_advantage_qvalue_augtd_nomasktd"
SCENES = ("3s5z", "corridor")


def selected_plans(repo):
    previous = {
        key: os.environ.get(key)
        for key in ("SCENES", "LABELS", "RUN_SUFFIX")
    }
    try:
        os.environ["SCENES"] = " ".join(SCENES)
        os.environ["LABELS"] = LABEL
        os.environ["RUN_SUFFIX"] = "_home1_qvalue"
        plans = build_plans(repo)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    expected = tuple((scene, LABEL) for scene in SCENES)
    actual = tuple((plan["scene"], plan["label"]) for plan in plans)
    if actual != expected or len(plans) != 2:
        raise RuntimeError("Direct-Q SMAC selection changed: {}".format(actual))

    for plan in plans:
        exports = plan["exports"]
        expected_exports = {
            "T_MAX": "10050000",
            "TEST_INTERVAL": "10000",
            "BATCH_SIZE_RUN": "8",
            "EXPECTED_BATCH_SIZE_RUN": "8",
            "BATCH_SIZE": "128",
            "BUFFER_SIZE": "5000",
        }
        for key, value in expected_exports.items():
            if exports.get(key) != value:
                raise RuntimeError(
                    "{} changed from {} to {}".format(
                        key, value, exports.get(key)
                    )
                )
        extra = dict(
            item.split("=", 1)
            for item in shlex.split(exports["EXTRA_ARGS"])
            if "=" in item
        )
        if extra.get("learner_updates_per_collect") != "1":
            raise RuntimeError("learner_updates_per_collect must remain 1")
        plan["sbatch_args"] = [
            "--mem=" + os.environ.get("SMAC_MEMORY", "96G")
            if argument.startswith("--mem=")
            else argument
            for argument in plan["sbatch_args"]
        ]
    return plans


def main():
    repo = Path(os.environ.get(
        "REPO_DIR", "/home/kyang/code/gomarl-dual-branch"
    )).resolve()
    runtime_root = Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"
    ))
    plans = selected_plans(repo)
    paths = route_runtime(plans, runtime_root)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return

    allowed_roots = ("/home/kyang/", "/fred/oz501/kyang/")
    if not runtime_root.is_absolute() or not any(
        str(runtime_root).startswith(root) for root in allowed_roots
    ):
        raise RuntimeError("Runtime output is outside the approved roots")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(str(path), os.W_OK):
            raise RuntimeError("Runtime directory is not writable: " + str(path))

    os.chdir(repo)
    subprocess.run(
        [sys.executable, "scripts/smoke_test_advantage_qvalue_smac.py"],
        check=True,
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
        "advantage_qvalue_smac_{}_{}.json".format(
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
