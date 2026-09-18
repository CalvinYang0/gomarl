#!/usr/bin/env python3
"""Submit the missing seed-1 ablations and five-scene HyperSelect set.

The suite contains only the two Counter ablations not covered by historical
runs, plus the full model on the five paper scenes other than Counter.  It
never cancels jobs and retains an active same-name job from this repository.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozstar_submit_counter_transformer_nine import (
    build_plans as counter_plans,
    run,
)
from ozstar_submit_relation_advantage_mixer_nine import route_runtime


FULL_LABEL = "relation_advantage_qvalue_augtd_nomasktd"
COUNTER_LABELS = (
    "hyperselect_gate_qme",
    "hyperselect_gate_sme",
)
PAPER_NAMES = {
    "hyperselect_gate_qme": "gate_qme",
    "hyperselect_gate_sme": "gate_sme",
}
TRANSFER_SCENES = (
    ("grf_pass", "academy_pass_and_shoot_with_keeper", "grf", "48G"),
    ("grf_3v1", "academy_3_vs_1_with_keeper", "grf", "48G"),
    # Keep ample headroom until the first optimized seed reports measured
    # MaxRSS. The failed Corridor control reached ~92 GB under a 96 GB limit.
    ("smac_3s5z", "3s5z_vs_3s6z", "smac", "160G"),
    ("smac_5m6m", "5m_vs_6m", "smac", "160G"),
    ("smac_mmm2", "MMM2", "smac", "160G"),
)


def _replace_arg(arguments, prefix, value):
    return [prefix + value if item.startswith(prefix) else item for item in arguments]


def build_plans(repo):
    from modules.agents.counter_transformer_suite import model_type_for

    previous = {
        key: os.environ.get(key)
        for key in ("SEED", "T_MAX", "RUN_SUFFIX", "MEMORY", "TIME")
    }
    os.environ.update(
        SEED="1",
        T_MAX="5050000",
        RUN_SUFFIX="_paper5m",
        MEMORY="96G",
        TIME="2-00:00:00",
    )
    try:
        counter = counter_plans(repo, COUNTER_LABELS)
        transfer_templates = [counter_plans(repo, (FULL_LABEL,))[0]
                              for _ in TRANSFER_SCENES]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    plans = []
    for plan in counter:
        paper_name = PAPER_NAMES[plan["label"]]
        plan["job_name"] = "grf_counter_paper_{}_s1".format(paper_name)
        plan["run_name"] = "grf_counter_paper_{}_5m_s1".format(paper_name)
        plan["exports"].update(
            T_MAX="5050000",
            RUN_NAME=plan["run_name"],
            GROUP_NAME="hyperselect_paper_counter_ablation_s1",
        )
        plan["sbatch_args"] = _replace_arg(
            plan["sbatch_args"], "--job-name=", plan["job_name"]
        )
        plan.update(scene="counter", map_name="academy_counterattack_easy",
                    domain="grf", suite="counter_ablation")
        plans.append(plan)

    for template, scene in zip(transfer_templates, TRANSFER_SCENES):
        scene_key, map_name, domain, memory = scene
        plan = template
        plan["job_name"] = "{}_paper_hyperselect_s1".format(scene_key)
        plan["run_name"] = "{}_paper_hyperselect_5m_s1".format(scene_key)
        env_config = "sc2" if domain == "smac" else map_name
        plan["exports"].update(
            ENV_CONFIG=env_config,
            MAP_NAME=map_name,
            MODEL_TYPE=model_type_for(FULL_LABEL, domain),
            T_MAX="5050000",
            RUN_NAME=plan["run_name"],
            GROUP_NAME="hyperselect_paper_transfer_s1",
        )
        plan["exports"]["EXTRA_ARGS"] += " test_nepisode=32"
        if domain == "grf":
            plan["exports"]["EXTRA_ARGS"] += " env_args.write_video=False"
        plan["sbatch_args"] = _replace_arg(
            plan["sbatch_args"], "--job-name=", plan["job_name"]
        )
        plan["sbatch_args"] = _replace_arg(
            plan["sbatch_args"], "--mem=", memory
        )
        plan.update(label=FULL_LABEL, scene=scene_key, map_name=map_name,
                    domain=domain, suite="paper_transfer")
        plans.append(plan)

    expected_count = len(COUNTER_LABELS) + len(TRANSFER_SCENES)
    if len(plans) != expected_count:
        raise RuntimeError("Paper suite plan count changed")
    if len({plan["job_name"] for plan in plans}) != expected_count:
        raise RuntimeError("Paper suite contains duplicate job names")
    for plan in plans:
        exports = plan["exports"]
        required = {
            "T_MAX": "5050000",
            "TEST_INTERVAL": "10000",
            "BATCH_SIZE_RUN": "8",
            "EXPECTED_BATCH_SIZE_RUN": "8",
            "BATCH_SIZE": "128",
            "BUFFER_SIZE": "5000",
        }
        for key, expected in required.items():
            if exports.get(key) != expected:
                raise RuntimeError(
                    "{} changed for {}: {}".format(
                        key, plan["job_name"], exports.get(key)
                    )
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
    smoke_commands = (
        [sys.executable,
         "scripts/smoke_test_hyperselect_memory_optimizations.py"],
        [sys.executable,
         "scripts/smoke_test_hyperselect_paper_counter_ablation.py"],
        [sys.executable, "scripts/smoke_test_hyperselect_paper_scenes.py"],
    )
    for command in smoke_commands:
        subprocess.run(command, check=True)

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
        "hyperselect_paper_seed1_{}_{}.json".format(
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
