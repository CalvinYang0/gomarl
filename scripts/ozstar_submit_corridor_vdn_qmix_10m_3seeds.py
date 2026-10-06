#!/usr/bin/env python3
"""Corridor fixed-head VDN/QMIX, seeds 1/2/3, 10M, two days.

Match rollout/optimization and scalar diagnostics of the obs/ID suite.
Plan-only unless SUBMIT=YES. No checkpoints, media or job cancellations.
"""
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _extra_args, _load_profiles, _plan

GROUP = "corridor_vdn_qmix_value_diagnostics_10m_3seeds"


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for method in ("vdn", "qmix"):
        for seed in (1, 2, 3):
            name = "smac_corridor_{}_baseline_10m_s{}_valuediag".format(method, seed)
            plan = _plan(repo, profiles, "smac_corridor", "corridor", "smac",
                         "linear_baseline", seed, name, "24G", GROUP)
            plan["label"] = method
            exports = plan["exports"]
            # Use the historical fixed-head paper baseline, not a generated
            # Linear head. Only the mixer differs between VDN and QMIX.
            exports["MODEL_TYPE"] = "qmix_minimal"
            exports["T_MAX"] = "10050000"
            exports["TEST_INTERVAL"] = "10000"
            exports["EXTRA_ARGS"] = _extra_args(profiles, "baseline", "smac") + (
                " mixer={} test_value_diagnostics=True"
                " test_value_diagnostics_interval=100000".format(method)
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
        "scripts/smoke_test_corridor_fixed_head_baselines.py",
    )
    submitter.build_plans = build_plans
    submitter.main()
