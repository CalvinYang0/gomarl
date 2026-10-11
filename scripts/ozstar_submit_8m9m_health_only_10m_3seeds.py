#!/usr/bin/env python3
"""Matched 8m_vs_9m health-only hyper-Obs, seeds 1/2/3, 10M.

Plan-only unless SUBMIT=YES; retain exact-name active/completed jobs.
Never cancel other experiments or reuse 5m6m health histories.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_health_only_10m_3seeds import LABEL, build_health_plans
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import validate_installed_map

GROUP = "8m9m_health_only_hyperobs_10m_3seeds"
SMOKE_SCRIPTS = (
    "scripts/smoke_test_8m9m_health_only.py",
    "scripts/smoke_test_submit_8m9m_health_only.py",
    "scripts/smoke_test_policy_importance.py",
    "scripts/smoke_test_hyper_obs_importance.py",
    "scripts/smoke_test_periodic_battle_videos.py",
)


def build_plans(repo):
    return build_health_plans(repo, "smac_8m9m", "8m_vs_9m", GROUP)


def main():
    if os.environ.get("SUBMIT") == "YES":
        if submitter.home_quota_free_gib() < 2.0:
            raise RuntimeError("Require at least 2 GiB free before media-enabled jobs; no jobs submitted")
        validate_installed_map()
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
