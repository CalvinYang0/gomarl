#!/usr/bin/env python3
"""Matched health-only hyper-Obs control: 5m6m, 10M, seeds 1/2/3.

Plan only unless SUBMIT=YES. Preserve active/completed exact-name jobs.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _load_profiles, _plan

GROUP = "5m6m_health_only_hyperobs_10m_3seeds"
LABEL = "linear_health_baseline"
SMOKE_SCRIPTS = (
    "scripts/smoke_test_5m6m_health_only.py",
    "scripts/smoke_test_submit_5m6m_health_only.py",
    "scripts/smoke_test_policy_importance.py",
    "scripts/smoke_test_hyper_obs_importance.py",
    "scripts/smoke_test_periodic_battle_videos.py",
)


def build_plans(repo):
    profiles = _load_profiles(repo)
    if profiles.ALL_PROFILES[LABEL] != {"branch": "linear", "hyper_obs_fill": "health_only"}:
        raise RuntimeError("Expected ungated single Linear health-only condition, main TD only")
    plans = []
    for seed in (1, 2, 3):
        name = "smac_5m6m_linear_health_only_10m_s{}_healthcond".format(seed)
        plan = _plan(repo, profiles, "smac_5m6m", "5m_vs_6m", "smac",
                     LABEL, seed, name, "24G", GROUP)
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
    return plans


def validate_installed_maps():
    from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
    validate_installed_map()


def main():
    if os.environ.get("SUBMIT") == "YES":
        if submitter.home_quota_free_gib() < 2.0:
            raise RuntimeError("Require at least 2 GiB free before media-enabled jobs; no jobs submitted")
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
