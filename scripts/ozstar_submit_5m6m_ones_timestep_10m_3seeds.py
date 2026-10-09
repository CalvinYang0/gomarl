#!/usr/bin/env python3
"""Matched hypernetwork-input controls: all ones versus raw episode timestep.

5m_vs_6m, three seeds per control, 10M, main TD only. No job cancellation.
Plan-only unless SUBMIT=YES. Capturer architecture/initialization, policy GRU
and mixer are unchanged from Linear Obs. Future-run battle videos enabled.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "5m6m_ones_timestep_hyperobs_10m_3seeds"
CONTROLS = (
    ("linear_ones_baseline", "linear_ones", "ones"),
    ("linear_timestep_baseline", "linear_timestep", "episode_timestep"),
)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for label, suffix, fill in CONTROLS:
        if profiles.ALL_PROFILES[label] != {"branch": "linear", "hyper_obs_fill": fill}:
            raise RuntimeError("Unexpected hyper-observation control: " + label)
        for seed in (1, 2, 3):
            name = "smac_5m6m_{}_10m_s{}_signalcond".format(suffix, seed)
            plan = _plan(repo, profiles, "smac_5m6m", "5m_vs_6m", "smac",
                         label, seed, name, "24G", GROUP)
            plan["exports"].update(T_MAX="10050000", TEST_INTERVAL="10000")
            plan["exports"]["EXTRA_ARGS"] = _extra_args(profiles, label, "smac") + (
                " test_value_diagnostics=True test_value_diagnostics_interval=100000"
                " test_battle_videos=True test_battle_video_interval=1000000 test_battle_video_episodes=10"
            )
            plan["sbatch_args"] = [
                "--time=2-00:00:00" if arg.startswith("--time=") else arg
                for arg in plan["sbatch_args"]
            ]
            plans.append(plan)
    if len(plans) != 6 or len({p["job_name"] for p in plans}) != 6:
        raise RuntimeError("Expected six distinct signal-conditioning jobs")
    return plans


def main():
    if os.environ.get("SUBMIT") == "YES":
        from ozstar_submit_5m6m_global_state_3seeds import MAP, MIN_HOME_FREE_GIB
        from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
        free = submitter.home_quota_free_gib()
        if free < MIN_HOME_FREE_GIB:
            raise RuntimeError("Require at least 2 GiB free; no jobs submitted")
        params = get_smac_map_registry().get(MAP)
        if params is None or (params["n_agents"], params["n_enemies"], params["map_type"]) != (5, 6, "marines"):
            raise RuntimeError("Unexpected installed 5m_vs_6m map definition")
        maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
        if not any(maps_root.rglob(MAP + ".SC2Map")):
            raise RuntimeError("Missing map file: " + MAP)
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_5m6m_ones_timestep.py",
        "scripts/smoke_test_submit_5m6m_ones_timestep.py",
        "scripts/smoke_test_periodic_battle_videos.py",
    )
    submitter.build_plans = build_plans
    submitter.main()


if __name__ == "__main__":
    main()
