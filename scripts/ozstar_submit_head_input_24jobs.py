#!/usr/bin/env python3
"""Fresh 10M visualization-covered head-input replicas: 8 cohorts, 24 jobs.

5m6m ID; 3s_vs_5z and MMM2 Obs/ID/ones; 8m_vs_9m ones. Plan-only unless
SUBMIT=YES. Never cancel jobs or reuse old scalar-only/5M/attention-ID runs.
Repeated invocation retains exact-name pending/running/completed jobs.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _load_profiles, _plan

GROUP = "smac_head_inputs_four_maps_10m_3seeds_vizcoverage"
SCENES = (
    ("smac_5m6m", "5m_vs_6m", ("linear_id_baseline",)),
    ("smac_3svs5z", "3s_vs_5z", ("linear_baseline", "linear_id_baseline", "linear_ones_baseline")),
    ("smac_mmm2", "MMM2", ("linear_baseline", "linear_id_baseline", "linear_ones_baseline")),
    ("smac_8m9m", "8m_vs_9m", ("linear_ones_baseline",)),
)
SUFFIXES = {"linear_baseline": "linear_obs", "linear_id_baseline": "linear_id",
            "linear_ones_baseline": "linear_ones"}
EXPECTED_FLAGS = {
    "linear_baseline": {"branch": "linear"},
    "linear_id_baseline": {"branch": "linear", "hyper_condition": "agent_id_linear"},
    "linear_ones_baseline": {"branch": "linear", "hyper_obs_fill": "ones"},
}
EXPECTED_MAPS = {"5m_vs_6m": (5, 6, "marines"), "3s_vs_5z": (3, 5, "stalkers"),
                 "MMM2": (10, 12, "MMM"), "8m_vs_9m": (8, 9, "marines")}
PLOT_LABELS = {
    "linear_baseline_vizcoverage": ("Linear Obs (new 10M visualization cohort)", "#08519c"),
    "linear_id_baseline_vizcoverage": ("Linear ID (new 10M visualization cohort)", "#a55194"),
    "linear_ones_baseline_vizcoverage": ("Linear all-ones (new 10M visualization cohort)", "#a6761d"),
}
SMOKE_SCRIPTS = (
    "scripts/smoke_test_submit_head_input_24jobs.py",
    "scripts/smoke_test_head_input_four_maps.py",
    "scripts/smoke_test_policy_importance.py",
    "scripts/smoke_test_hyper_obs_importance.py",
    "scripts/smoke_test_periodic_battle_videos.py",
)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for scene, map_name, labels in SCENES:
        for label in labels:
            if profiles.ALL_PROFILES[label] != EXPECTED_FLAGS[label]:
                raise RuntimeError("Unexpected head-input profile: " + label)
            for seed in (1, 2, 3):
                name = "{}_{}_10m_s{}_vizcoverage".format(scene, SUFFIXES[label], seed)
                plan = _plan(repo, profiles, scene, map_name, "smac", label, seed,
                             name, "24G", GROUP)
                plan["exports"].update(T_MAX="10050000", TEST_INTERVAL="10000")
                plan["exports"]["EXTRA_ARGS"] += (
                    " test_value_diagnostics=True test_value_diagnostics_interval=100000"
                    " test_visualizations_required=True"
                    " test_battle_videos=True test_battle_video_interval=1000000 test_battle_video_episodes=10"
                    " test_policy_importance=True test_policy_importance_interval=1000000"
                    " test_policy_importance_episodes=10 test_policy_importance_samples=64"
                    " test_hyper_obs_importance=True test_hyper_obs_importance_interval=1000000"
                    " test_hyper_obs_importance_episodes=10"
                )
                plan["sbatch_args"] = ["--time=2-00:00:00" if arg.startswith("--time=") else arg
                                        for arg in plan["sbatch_args"]]
                plan["target_steps"] = 10000000
                plans.append(plan)
    if len(plans) != 24 or len({p["job_name"] for p in plans}) != 24:
        raise RuntimeError("Expected exactly 24 distinct jobs")
    return plans


def validate_installed_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    maps_root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    for name, expected in EXPECTED_MAPS.items():
        params = registry.get(name)
        actual = None if params is None else (params["n_agents"], params["n_enemies"], params["map_type"])
        if actual != expected:
            raise RuntimeError("Wrong installed map: {} actual={} expected={}".format(name, actual, expected))
        if not any(maps_root.rglob(name + ".SC2Map")):
            raise RuntimeError("Missing SC2 map file: " + name)
        print("Map verified: {} {}".format(name, actual), flush=True)


def main():
    if os.environ.get("SUBMIT") == "YES":
        if submitter.home_quota_free_gib() < 5.0:
            raise RuntimeError("Require at least 5 GiB free before 24 media-enabled jobs; no jobs submitted")
        validate_installed_maps()
    replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__), SMOKE_SCRIPTS=SMOKE_SCRIPTS,
                       build_plans=build_plans)
    original = {key: getattr(submitter, key) for key in replacement}
    try:
        for key, value in replacement.items():
            setattr(submitter, key, value)
        submitter.main()
    finally:
        for key, value in original.items():
            setattr(submitter, key, value)


if __name__ == "__main__":
    main()
