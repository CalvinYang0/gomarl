#!/usr/bin/env python3
"""Fresh GRU + linear-ID-conditioned heads, two maps, 3 seeds, 10M.

Old historical-attention ID runs are preserved. No cancellation, checkpoints,
or media. SUBMIT=YES opts into preflight and submission; repeated calls retain
exact-name active/completed runs. Require >=2 GiB of home quota before admission.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "linear_id_5m6m_8m9m_value_diagnostics_10m_3seeds"
SCENES = (("smac_5m6m", "5m_vs_6m"), ("smac_8m9m", "8m_vs_9m"))


def build_plans(repo):
    profiles = _load_profiles(repo)
    label = "linear_id_baseline"
    if profiles.ALL_PROFILES[label] != {"branch": "linear", "hyper_condition": "agent_id_linear"}:
        raise RuntimeError("Expected the matched, main-TD-only Linear ID profile")
    plans = []
    for scene, map_name in SCENES:
        for seed in (1, 2, 3):
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
    assert len(plans) == len({p["job_name"] for p in plans}) == 6
    return plans


def validate_installed_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    for name, counts in (("5m_vs_6m", (5, 6)), ("8m_vs_9m", (8, 9))):
        params = registry[name]
        if (params["n_agents"], params["n_enemies"], params["map_type"]) != (*counts, "marines"):
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
        "scripts/smoke_test_linear_id_baseline.py",
        "scripts/smoke_test_id_hypernet_smac.py",
        "scripts/smoke_test_value_diagnostics.py",
    )
    submitter.build_plans = build_plans
    submitter.main()
