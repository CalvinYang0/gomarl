#!/usr/bin/env python3
"""Submit five isolated QME diagnostics, defaulting to 3s5z seed 1."""
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


DEFAULT_LABELS = (
    "hyperselect_qme_joint_value",
    "hyperselect_qme_td_quality",
    "hyperselect_qme_action_rank",
    "hyperselect_qme_stable_teacher",
    "hyperselect_qme_dynamic_readiness",
)
ALL_LABELS = DEFAULT_LABELS + (
    "hyperselect_qme_action_q_scaled",
    "hyperselect_qme_full_behavior",
    "hyperselect_qme_mixed_behavior",
    "hyperselect_qme_open_win_ready",
)
ALL_SCENES = (
    ("3s5z", "3s5z_vs_3s6z"),
    ("5m6m", "5m_vs_6m"),
)
SHORT_NAMES = {
    "hyperselect_qme_joint_value": "joint_value",
    "hyperselect_qme_td_quality": "td_quality",
    "hyperselect_qme_action_rank": "action_rank",
    "hyperselect_qme_stable_teacher": "stable_teacher",
    "hyperselect_qme_dynamic_readiness": "dynamic_ready",
    "hyperselect_qme_action_q_scaled": "q_scaled",
    "hyperselect_qme_full_behavior": "full_behavior",
    "hyperselect_qme_mixed_behavior": "mixed_behavior",
    "hyperselect_qme_open_win_ready": "openwin_ready",
}


def _selected(values, environment_name, default=None):
    requested = tuple(
        item
        for item in os.environ.get(
            environment_name,
            " ".join(values) if default is None else default,
        ).split()
        if item
    )
    unknown = sorted(set(requested) - set(values))
    if not requested or unknown or len(requested) != len(set(requested)):
        raise ValueError(
            "Invalid {} selection: {}".format(environment_name, requested)
        )
    return requested


def _replace_arg(arguments, prefix, value):
    return [prefix + value if item.startswith(prefix) else item for item in arguments]


def build_plans(repo):
    from modules.agents.counter_transformer_suite import model_type_for

    labels = _selected(
        ALL_LABELS,
        "LABELS",
        default=" ".join(DEFAULT_LABELS),
    )
    scene_keys = _selected(
        tuple(item[0] for item in ALL_SCENES), "SCENES", default="3s5z"
    )
    scenes = tuple(item for item in ALL_SCENES if item[0] in scene_keys)
    previous = {
        key: os.environ.get(key)
        for key in ("SEED", "T_MAX", "RUN_SUFFIX", "MEMORY", "TIME")
    }
    os.environ.update(
        SEED="1",
        T_MAX="5050000",
        RUN_SUFFIX="_stableqme5m",
        MEMORY="160G",
        TIME="2-00:00:00",
    )
    try:
        templates = counter_plans(repo, labels)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    plans = []
    for template in templates:
        for scene_key, map_name in scenes:
            plan = dict(template)
            plan["exports"] = dict(template["exports"])
            plan["sbatch_args"] = list(template["sbatch_args"])
            short = SHORT_NAMES[template["label"]]
            plan["job_name"] = "smac_{}_hyperselect_{}_s1".format(
                scene_key, short
            )
            plan["run_name"] = "smac_{}_hyperselect_{}_5m_s1".format(
                scene_key, short
            )
            plan["exports"].update(
                ENV_CONFIG="sc2",
                MAP_NAME=map_name,
                MODEL_TYPE=model_type_for(template["label"], "smac"),
                T_MAX="5050000",
                RUN_NAME=plan["run_name"],
                GROUP_NAME="hyperselect_stable_qme_s1",
            )
            plan["exports"]["EXTRA_ARGS"] += " test_nepisode=32"
            plan["sbatch_args"] = _replace_arg(
                plan["sbatch_args"], "--job-name=", plan["job_name"]
            )
            plan["sbatch_args"] = _replace_arg(
                plan["sbatch_args"], "--mem=", "160G"
            )
            plan.update(
                scene=scene_key,
                map_name=map_name,
                domain="smac",
                suite="stable_qme",
            )
            plans.append(plan)
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
    for script in (
        "scripts/smoke_test_hyperselect_memory_optimizations.py",
        "scripts/smoke_test_hyperselect_stable_qme.py",
        "scripts/smoke_test_hyperselect_behavior_sampling.py",
    ):
        subprocess.run([sys.executable, script], check=True)

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
        "hyperselect_stable_qme_seed1_{}_{}.json".format(
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
