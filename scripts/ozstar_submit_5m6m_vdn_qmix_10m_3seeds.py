#!/usr/bin/env python3
"""Fresh matched 5m_vs_6m VDN/QMIX, seeds 1/2/3, 10M, with battle videos.

Plan-only unless SUBMIT=YES. Never extend/overwrite historical 5M runs,
cancel existing jobs, or resubmit exact-name active/completed jobs.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_fixed_head_plans
from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map

GROUP = "5m6m_vdn_qmix_value_diagnostics_10m_3seeds"


def build_plans(repo):
    return build_fixed_head_plans(repo, "smac_5m6m", "5m_vs_6m", GROUP)


def main():
    if os.environ.get("SUBMIT") == "YES":
        if submitter.home_quota_free_gib() < 2.0:
            raise RuntimeError("Require at least 2 GiB free in /home; no jobs submitted")
        validate_installed_map()
    replacement = dict(
        GROUP=GROUP, SUITE_FILE=Path(__file__), build_plans=build_plans,
        SMOKE_SCRIPTS=(
            "scripts/smoke_test_submit_5m6m_vdn_qmix.py",
            "scripts/smoke_test_5m6m_fixed_head_baselines.py",
            "scripts/smoke_test_periodic_battle_videos.py",
        ),
    )
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
