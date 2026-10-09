#!/usr/bin/env python3
"""5m6m Obs + absolute entity-ID hyper-input control, three seeds, 10M.

Only the affine hyper-observation input is enlarged. GRU/mixer/loss unchanged.
Plan-only unless SUBMIT=YES. Existing active/completed runs are retained.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "5m6m_obs_entity_id_hyperobs_10m_3seeds"
LABEL = "linear_obs_entity_id_baseline"


def build_plans(repo):
    profiles = _load_profiles(repo)
    if profiles.ALL_PROFILES[LABEL] != {"branch": "linear", "hyper_entity_ids": True}:
        raise RuntimeError("Expected ungated Linear Obs + entity IDs, TD only")
    plans = []
    for seed in (1, 2, 3):
        name = "smac_5m6m_linear_obs_entity_id_10m_s{}_entityidcond".format(seed)
        plan = _plan(repo, profiles, "smac_5m6m", "5m_vs_6m", "smac",
                     LABEL, seed, name, "24G", GROUP)
        plan["exports"].update(T_MAX="10050000", TEST_INTERVAL="10000")
        plan["exports"]["EXTRA_ARGS"] = _extra_args(profiles, LABEL, "smac") + (
            " test_value_diagnostics=True test_value_diagnostics_interval=100000"
            " test_battle_videos=True test_battle_video_interval=1000000 test_battle_video_episodes=10"
        )
        plan["sbatch_args"] = [
            "--time=2-00:00:00" if arg.startswith("--time=") else arg
            for arg in plan["sbatch_args"]
        ]
        plans.append(plan)
    if len(plans) != 3 or len({p["job_name"] for p in plans}) != 3:
        raise RuntimeError("Expected three distinct entity-ID jobs")
    return plans


def main():
    if os.environ.get("SUBMIT") == "YES":
        from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
        if submitter.home_quota_free_gib() < 2.0:
            raise RuntimeError("Require at least 2 GiB free in /home; no jobs submitted")
        validate_installed_map()
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_5m6m_obs_entity_id.py",
        "scripts/smoke_test_submit_5m6m_obs_entity_id.py",
        "scripts/smoke_test_periodic_battle_videos.py",
    )
    submitter.build_plans = build_plans
    submitter.main()


if __name__ == "__main__":
    main()
