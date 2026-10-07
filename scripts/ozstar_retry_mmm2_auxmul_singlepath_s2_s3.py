#!/usr/bin/env python3
"""Retry only the two startup-failed MMM2 single-TD auxmul seeds.

Keep exact experiment names/group so plots select the new attempts. Never
cancel jobs; retain exact-name active/completed retries. SUBMIT=YES submits.
"""
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _load_profiles
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import (
    GROUP, build_plans as suite_plans,
)


def build_plans(repo):
    plans = [
        plan for plan in suite_plans(repo)
        if plan["scene"] == "smac_mmm2"
        and plan["label"] == "linear_obs_gate_kl80aux_multiply_singlepath"
        and plan["seed"] in (2, 3)
    ]
    assert len(plans) == 2 and {p["seed"] for p in plans} == {2, 3}
    profiles = _load_profiles(repo)
    for plan in plans:
        model = plan["exports"]["MODEL_TYPE"]
        profile = profiles.MODEL_PROFILES.get(model)
        if not profile or profile.get("domain") != "smac":
            raise RuntimeError("SMAC agent model is not registered: " + model)
        assert profile["main_td_coef"] == 0.0
        assert profile["aux"] and profile["diagnostic_main_no_grad"]
    return plans


if __name__ == "__main__":
    submitter.GROUP = GROUP + "_retry_mmm2_s2_s3"
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = ("scripts/smoke_test_linear_single_head_suite.py",)
    submitter.build_plans = build_plans
    submitter.main()
