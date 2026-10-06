#!/usr/bin/env python3
"""Fresh Linear obs-based and historical ID-based baselines with scalar diagnostics.

5m_vs_6m, seeds 1/2/3, 10M steps, two days; no checkpoints/media.
The ID baseline retains its historical attention representation, not a new
Linear-ID architecture. SUBMIT=YES enables preflight and submission.
"""
import os
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "5m6m_obslinear_id_value_diagnostics_10m_3seeds"


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for label, suffix, memory in (
        ("linear_baseline", "linear_obs_baseline", "24G"),
        ("hyper_hypermarl_id", "id_baseline", "32G"),
    ):
        flags = profiles.ALL_PROFILES[label]
        if any(flags.get(k) for k in ("gate", "kl", "aux", "advantage_margin")):
            raise RuntimeError("This suite must contain ungated main-TD-only baselines")
        for seed in (1, 2, 3):
            name = "smac_5m6m_{}_10m_s{}_valuediag".format(suffix, seed)
            # Reuse matched rollout/optimization settings, then replace the
            # architecture/profile for the historical ID-conditioned model.
            plan = _plan(repo, profiles, "smac_5m6m", "5m_vs_6m", "smac",
                         "linear_baseline", seed, name, memory, GROUP)
            plan["label"] = label
            exports = plan["exports"]
            exports["MODEL_TYPE"] = profiles.model_type_for(label, "smac")
            exports["T_MAX"] = "10050000"
            exports["TEST_INTERVAL"] = "10000"
            exports["EXTRA_ARGS"] = _extra_args(profiles, label, "smac") + (
                " test_value_diagnostics=True test_value_diagnostics_interval=100000"
            )
            plan["sbatch_args"] = [
                "--time=2-00:00:00" if arg.startswith("--time=") else arg
                for arg in plan["sbatch_args"]
            ]
            plans.append(plan)
    assert len(plans) == 6 and len({p["job_name"] for p in plans}) == 6
    return plans


if __name__ == "__main__":
    submitter.GROUP = GROUP
    submitter.SUITE_FILE = Path(__file__)
    submitter.SMOKE_SCRIPTS = (
        "scripts/smoke_test_value_diagnostics.py",
        "scripts/smoke_test_linear_single_head_suite.py",
        "scripts/smoke_test_id_hypernet_smac.py",
    )
    submitter.build_plans = build_plans
    submitter.main()
