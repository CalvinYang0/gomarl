#!/usr/bin/env python3
"""Submit the corrected additive HyperSelect Counter ablation, seed 1."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozstar_submit_counter_transformer_nine import build_plans as counter_plans, run
from ozstar_submit_relation_advantage_mixer_nine import route_runtime


DEFAULT_LABELS = (
    "hyperselect_gate",
    "hyperselect_gate_ltd_control",
    "hyperselect_gate_qme_additive",
    "hyperselect_gate_sme_additive",
    "hyperselect_gate_qme_sme_additive",
)
ALL_LABELS = DEFAULT_LABELS + ("hyperselect_qme_action_q_scaled",)
PAPER_NAMES = {
    "hyperselect_gate": "gate",
    "hyperselect_gate_ltd_control": "gate_ltd",
    "hyperselect_gate_qme_additive": "gate_qme",
    "hyperselect_gate_sme_additive": "gate_sme",
    "hyperselect_gate_qme_sme_additive": "full",
    "hyperselect_qme_action_q_scaled": "q_scaled",
}


def _replace_arg(arguments, prefix, value):
    return [prefix + value if item.startswith(prefix) else item for item in arguments]


def build_plans(repo):
    requested = tuple(
        item
        for item in os.environ.get("LABELS", " ".join(DEFAULT_LABELS)).split()
        if item
    )
    if not requested:
        raise ValueError("LABELS must select at least one additive ablation")
    unknown = sorted(set(requested) - set(ALL_LABELS))
    if unknown:
        raise ValueError("Unknown additive ablation labels: " + ", ".join(unknown))
    if len(set(requested)) != len(requested):
        raise ValueError("LABELS contains a duplicate additive ablation")
    previous = {
        key: os.environ.get(key)
        for key in ("SEED", "T_MAX", "RUN_SUFFIX", "MEMORY", "TIME")
    }
    os.environ.update(
        SEED="1",
        T_MAX="5050000",
        RUN_SUFFIX="_additive5m",
        MEMORY="96G",
        TIME="2-00:00:00",
    )
    try:
        plans = counter_plans(repo, requested)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    for plan in plans:
        paper_name = PAPER_NAMES[plan["label"]]
        plan["job_name"] = "grf_counter_additive_{}_s1".format(paper_name)
        plan["run_name"] = "grf_counter_additive_{}_5m_s1".format(paper_name)
        plan["exports"].update(
            T_MAX="5050000",
            RUN_NAME=plan["run_name"],
            GROUP_NAME="hyperselect_counter_additive_ablation_s1",
        )
        plan["sbatch_args"] = _replace_arg(
            plan["sbatch_args"], "--job-name=", plan["job_name"]
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
        [sys.executable, "scripts/smoke_test_hyperselect_memory_optimizations.py"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/smoke_test_hyperselect_additive_counter_ablation.py"],
        check=True,
    )
    if "hyperselect_qme_action_q_scaled" in {
        plan["label"] for plan in plans
    }:
        subprocess.run(
            [sys.executable, "scripts/smoke_test_hyperselect_stable_qme.py"],
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
        "hyperselect_counter_additive_seed1_{}_{}.json".format(
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
        "%.18i %.80j %.10T %.12M %.10m %R",
    ]))


if __name__ == "__main__":
    main()
