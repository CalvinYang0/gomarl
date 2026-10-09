#!/usr/bin/env python3
"""Matched fixed-head VDN/QMIX on 8m_vs_9m, three seeds, 10M.

Only the mixer differs between the two new groups. Match the existing
Linear Obs/ID rollout, optimizer and greedy evaluation settings. Keep value
diagnostics and ten test battle videos per 1M milestone. Plan-only unless
SUBMIT=YES; exact-name active/completed runs are retained, never cancelled.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "8m9m_vdn_qmix_value_diagnostics_10m_3seeds"
MAP = "8m_vs_9m"


def build_plans(repo):
    profiles = _load_profiles(repo)
    if profiles.ALL_PROFILES["baseline"] != {}:
        raise RuntimeError("Expected ungated baseline without auxiliary objectives")
    plans = []
    for method in ("vdn", "qmix"):
        for seed in (1, 2, 3):
            name = "smac_8m9m_{}_baseline_10m_s{}_valuediag".format(method, seed)
            plan = _plan(repo, profiles, "smac_8m9m", MAP, "smac",
                         "linear_baseline", seed, name, "24G", GROUP)
            plan["label"] = method
            exports = plan["exports"]
            # Same fixed two-layer ELU action head as the repository's paper
            # and Corridor controls. This is not a hyper-generated head.
            exports.update(MODEL_TYPE="qmix_minimal", T_MAX="10050000",
                           TEST_INTERVAL="10000")
            exports["EXTRA_ARGS"] = _extra_args(profiles, "baseline", "smac") + (
                " mixer={} test_greedy=True"
                " test_value_diagnostics=True test_value_diagnostics_interval=100000"
                " test_battle_videos=True test_battle_video_interval=1000000"
                " test_battle_video_episodes=10".format(method)
            )
            plan["sbatch_args"] = [
                "--time=2-00:00:00" if arg.startswith("--time=") else arg
                for arg in plan["sbatch_args"]
            ]
            plans.append(plan)
    if len(plans) != 6 or len({p["job_name"] for p in plans}) != 6:
        raise RuntimeError("Expected six distinct 8m_vs_9m fixed-head jobs")
    return plans


def validate_installed_map():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    params = get_smac_map_registry().get(MAP)
    if params is None or (params["n_agents"], params["n_enemies"], params["map_type"]) != (8, 9, "marines"):
        raise RuntimeError("Unexpected installed 8m_vs_9m map definition")
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    if not any(maps_root.rglob(MAP + ".SC2Map")):
        raise RuntimeError("Missing map file: " + MAP)
    print("Map preflight passed: 8m_vs_9m, 8 Marines vs. 9 Marines", flush=True)


def main():
    if os.environ.get("SUBMIT") == "YES":
        if submitter.home_quota_free_gib() < 2.0:
            raise RuntimeError("Require at least 2 GiB free in /home; no jobs submitted")
        validate_installed_map()
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_submit_8m9m_vdn_qmix.py",
        "scripts/smoke_test_8m9m_fixed_head_baselines.py",
        "scripts/smoke_test_periodic_battle_videos.py",
    )
    submitter.build_plans = build_plans
    submitter.main()


if __name__ == "__main__":
    main()
