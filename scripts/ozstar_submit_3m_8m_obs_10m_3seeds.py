#!/usr/bin/env python3
"""Submit only 3m/8m obs-based Linear baselines, 3 seeds, 10M, 48h.

Same profile, optimizer, rollout and scalar diagnostics as the existing 5m6m
obs runs. No ID jobs. Exact-name active/completed jobs are retained. Require
at least 2 GiB of home quota before submission. Default is plan-only.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as value_plans

GROUP = "marine_3m_8m_obslinear_value_diagnostics_10m_3seeds"
SCENES = (("smac_3m", "3m"), ("smac_8m", "8m"))


def build_plans(repo):
    plans = []
    for scene, map_name in SCENES:
        plans.extend(plan for plan in value_plans(repo, scene, map_name, GROUP)
                     if plan["label"] == "linear_baseline")
    assert len(plans) == len({p["job_name"] for p in plans}) == 6
    assert all(p["exports"]["MODEL_TYPE"] == "smac_single_linear_suite_baseline_hypercond"
               for p in plans)
    return plans


def validate_installed_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    for name, count in (("3m", 3), ("8m", 8)):
        params = registry[name]
        if (params["n_agents"], params["n_enemies"], params["map_type"]) != (count, count, "marines"):
            raise RuntimeError("Unexpected SMAC map definition: " + name)
        if not any(maps_root.rglob(name + ".SC2Map")):
            raise RuntimeError("Missing map file: " + name)
        print("Map preflight passed: " + name, flush=True)


if __name__ == "__main__":
    if os.environ.get("SUBMIT") == "YES":
        free = submitter.home_quota_free_gib()
        print("/home quota free: {:.2f} GiB".format(free), flush=True)
        if free < 2.0:
            raise RuntimeError("Require at least 2 GiB free; no jobs submitted")
        validate_installed_maps()
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_3m_8m_obs_baseline.py",
        "scripts/smoke_test_value_diagnostics.py",
    )
    submitter.build_plans = build_plans
    submitter.main()
