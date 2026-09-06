#!/usr/bin/env python3
"""Append paired baseline/KL80aux jobs; never cancel existing experiments.

DRY_RUN=YES prints plans without Slurm or filesystem writes. SCENES accepts
space-separated keys below. Both methods use identical resources per scene.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ozstar_submit_counter_transformer_nine import build_plans as counter_plans, run

SCENES = {
    "pass": ("academy_pass_and_shoot_with_keeper", "grf", "24G"),
    "corridor": ("corridor", "smac", "96G"),
    "mmm2": ("MMM2", "smac", "96G"),
    "3s5z": ("3s5z_vs_3s6z", "smac", "48G"),
}
LABELS = ("baseline", "relation_kl80aux")


def build_plans(repo):
    spec = importlib.util.spec_from_file_location(
        "suite_profiles", repo / "src/modules/agents/counter_transformer_suite.py")
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    selected = os.environ.get("SCENES", " ".join(SCENES)).split()
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(SCENES):
        raise ValueError("SCENES must select unique keys from " + " ".join(SCENES))
    plans = []
    for scene in selected:
        map_name, domain, memory = SCENES[scene]
        for plan in counter_plans(repo, LABELS):
            prefix = "{}_{}_trans9_".format(domain, scene)
            for key in ("job_name", "run_name"):
                plan[key] = plan[key].replace("grf_counter_trans9_", prefix)
            exports = plan["exports"]
            exports.update(ENV_CONFIG="sc2" if domain == "smac" else map_name,
                           MAP_NAME=map_name, RUN_NAME=plan["run_name"],
                           GROUP_NAME=plan["run_name"],
                           MODEL_TYPE=profiles.model_type_for(plan["label"], domain))
            exports["EXTRA_ARGS"] += " test_nepisode=32"
            if domain == "grf":
                exports["EXTRA_ARGS"] += " env_args.write_video=False"
            plan["sbatch_args"] = [
                "--job-name=" + plan["job_name"] if arg.startswith("--job-name=")
                else "--mem=" + os.environ.get(domain.upper() + "_MEMORY", memory)
                if arg.startswith("--mem=") else arg
                for arg in plan["sbatch_args"]]
            plan.update(scene=scene, map_name=map_name, domain=domain)
            plans.append(plan)
    return plans


def main():
    repo = Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")).resolve()
    plans = build_plans(repo)
    if os.environ.get("DRY_RUN") == "YES":
        print(json.dumps(plans, indent=2))
        return
    os.chdir(repo)
    subprocess.run([sys.executable, "scripts/smoke_test_trans9_multiscene.py"], check=True)
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
            raise RuntimeError("Same-name job has unknown/different workdir: " + name)
        if name in retained:
            raise RuntimeError("Duplicate active job: " + name)
        retained[name] = job_id
    script = "scripts/ozstar_train_offline.sbatch"
    logdir = repo / "ozstar_logs"
    logdir.mkdir(exist_ok=True)
    for plan in plans:
        if plan["job_name"] not in retained:
            subprocess.run(["sbatch", "--test-only"] + plan["sbatch_args"] + [script],
                           env=dict(os.environ, **plan["exports"]), check=True)
    manifest = logdir / ("trans9_multiscene_" + time.strftime("%Y%m%d_%H%M%S")
                         + "_" + str(os.getpid()) + ".json")
    record = dict(commit=run(["git", "rev-parse", "HEAD"]),
                  worktree_status=run(["git", "status", "--short"]),
                  plans=plans, retained=retained, submitted={})

    def persist():
        with manifest.open("w") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

    persist()
    for plan in plans:
        name = plan["job_name"]
        if name in retained:
            print("Retained " + name + " job=" + retained[name], flush=True)
            continue
        result = run(["sbatch", "--parsable"] + plan["sbatch_args"] + [script],
                     env=dict(os.environ, **plan["exports"]))
        record["submitted"][name] = result.split(";", 1)[0]
        persist()
        print("Submitted " + name + " job=" + result, flush=True)
    print("Manifest: " + str(manifest))


if __name__ == "__main__":
    main()
