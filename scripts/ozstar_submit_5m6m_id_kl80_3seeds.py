#!/usr/bin/env python3
"""Submit matched 5m_vs_6m Linear-ID and direct-KL80 runs.

Three seeds per condition, 5M environment steps, two-day limit. Fresh unique
names deliberately avoid any older KL80 attempts whose branch/gate may have
been misconfigured. Plan-only by default; SUBMIT=YES enables submission.
Nothing is cancelled. Requires at least 2 GiB free in /home.
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

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "smac_5m6m_linear_id_kl80_5m_3seeds"
SCENE = "smac_5m6m"
MAP = "5m_vs_6m"
CONDITIONS = (
    ("linear_id_baseline", "linear_id"),
    ("linear_bayesg_kl80_keep", "linear_kl80_direct_linearonly"),
)
SEEDS = (1, 2, 3)
MIN_HOME_FREE_GIB = 2.0


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for label, suffix in CONDITIONS:
        flags = profiles.ALL_PROFILES[label]
        if flags.get("branch") != "linear":
            raise RuntimeError("Refusing non-Linear branch: {} -> {}".format(label, flags))
        if label == "linear_id_baseline":
            if flags != {"branch": "linear", "hyper_condition": "agent_id_linear"}:
                raise RuntimeError("Unexpected Linear ID profile: " + repr(flags))
        elif label == "linear_bayesg_kl80_keep":
            if not flags.get("gate") or not flags.get("kl") or flags.get("aux"):
                raise RuntimeError("Direct KL80 must have gate+KL only: " + repr(flags))
        plans_for_label = []
        for seed in SEEDS:
            # Fresh names distinguish this corrected ID/direct-KL80 rerun from
            # historical attempts whose branch/gate may have been wrong.
            name = "{}_{}_5m_s{}_idkl80fix".format(SCENE, suffix, seed)
            plan = _plan(
                repo, profiles, SCENE, MAP, "smac", label, seed,
                name, "24G", GROUP,
            )
            plan["exports"]["T_MAX"] = "5050000"
            plan["exports"]["TEST_INTERVAL"] = "10000"
            plan["exports"]["EXTRA_ARGS"] = _extra_args(
                profiles, label, "smac"
            ) + " test_value_diagnostics=True test_value_diagnostics_interval=100000"
            plan["sbatch_args"] = [
                "--time=2-00:00:00" if arg.startswith("--time=") else arg
                for arg in plan["sbatch_args"]
            ]
            plan["profile_flags"] = flags
            plans_for_label.append(plan)
            plans.append(plan)
        expected_model = profiles.model_type_for(label, "smac")
        if any(plan["exports"]["MODEL_TYPE"] != expected_model for plan in plans_for_label):
            raise RuntimeError("Resolved model type changed within " + label)
        if "single_linear" not in expected_model:
            raise RuntimeError("Model name did not resolve to single-linear: " + expected_model)

    expected = len(CONDITIONS) * len(SEEDS)
    if len(plans) != expected or len({plan["job_name"] for plan in plans}) != expected:
        raise RuntimeError("Expected {} unique jobs".format(expected))
    return plans


def validate_installed_map():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry

    params = get_smac_map_registry().get(MAP)
    if params is None:
        raise RuntimeError("Map not registered in installed SMAC: " + MAP)
    actual = (params["n_agents"], params["n_enemies"], params["map_type"])
    if actual != (5, 6, "marines"):
        raise RuntimeError("Unexpected SMAC map definition: {}".format(actual))
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    if not any(maps_root.rglob(MAP + ".SC2Map")):
        raise RuntimeError("Missing map file under {}: {}".format(maps_root, MAP))
    print("Map preflight passed: {} {}".format(MAP, actual), flush=True)


def main():
    if os.environ.get("SUBMIT") == "YES":
        free = submitter.home_quota_free_gib()
        print("/home quota free: {:.2f} GiB".format(free), flush=True)
        if free < MIN_HOME_FREE_GIB:
            raise RuntimeError("Require at least 2 GiB free; no jobs submitted")
        validate_installed_map()

    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_linear_id_baseline.py",
        "scripts/smoke_test_linear_direct_kl80.py",
    )
    submitter.build_plans = build_plans
    submitter.main()


if __name__ == "__main__":
    main()
