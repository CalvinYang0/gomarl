#!/usr/bin/env python3
"""Submit matched Linear-ID controls on 8m_vs_9m and 6h_vs_8z.

Three seeds per map, 10M, GRU policy unchanged, main TD only. Plan-only
unless SUBMIT=YES. Existing exact-name active/completed runs are retained;
historical attention-ID runs are never reused and no jobs are cancelled.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "linear_id_8m9m_6h8z_value_diagnostics_10m_3seeds"
SCENES = (("smac_8m9m", "8m_vs_9m"), ("smac_6h8z", "6h_vs_8z"))
EXPECTED_MAPS = {
    "8m_vs_9m": (8, 9, "marines"),
    "6h_vs_8z": (6, 8, "hydralisks"),
}


def build_plans(repo):
    profiles = _load_profiles(repo)
    label = "linear_id_baseline"
    if profiles.ALL_PROFILES[label] != {"branch": "linear", "hyper_condition": "agent_id_linear"}:
        raise RuntimeError("Expected matched Linear-ID with no gates or auxiliary losses")
    plans = []
    for scene, map_name in SCENES:
        for seed in (1, 2, 3):
            # Reuse the previously defined corrected 8m9m names to avoid duplicates.
            name = "{}_linear_id_baseline_10m_s{}_valuediag".format(scene, seed)
            plan = _plan(repo, profiles, scene, map_name, "smac", label,
                         seed, name, "24G", GROUP)
            plan["exports"].update(T_MAX="10050000", TEST_INTERVAL="10000")
            plan["exports"]["EXTRA_ARGS"] = _extra_args(profiles, label, "smac") + (
                " test_value_diagnostics=True test_value_diagnostics_interval=100000"
            )
            plan["sbatch_args"] = [
                "--time=2-00:00:00" if arg.startswith("--time=") else arg
                for arg in plan["sbatch_args"]
            ]
            plans.append(plan)
    if len(plans) != 6 or len({p["job_name"] for p in plans}) != 6:
        raise RuntimeError("Expected six distinct Linear-ID jobs")
    return plans


def validate_installed_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    for name, expected in EXPECTED_MAPS.items():
        params = registry.get(name)
        if params is None:
            raise RuntimeError("Map is not registered: " + name)
        actual = (params["n_agents"], params["n_enemies"], params["map_type"])
        if actual != expected:
            raise RuntimeError("Unexpected map definition: {} {}".format(name, actual))
        if not any(maps_root.rglob(name + ".SC2Map")):
            raise RuntimeError("Missing map file: " + name)
        print("Map preflight passed: {} {}".format(name, actual), flush=True)


if __name__ == "__main__":
    if os.environ.get("SUBMIT") == "YES":
        free = submitter.home_quota_free_gib()
        if free < 2.0:
            raise RuntimeError("Require at least 2 GiB free in /home before submission")
        validate_installed_maps()
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_linear_id_8m9m_6h8z.py",
        "scripts/smoke_test_submit_linear_id_8m9m_6h8z.py",
    )
    submitter.build_plans = build_plans
    submitter.main()
