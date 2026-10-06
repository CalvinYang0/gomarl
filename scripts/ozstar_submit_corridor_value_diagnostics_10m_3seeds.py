#!/usr/bin/env python3
"""Corridor obs-Linear and historical ID baselines, 3 seeds, 10M, 2 days.

Main TD only; no gate/KL/QME, checkpoints or media. Scalar value diagnostics
match the 5m6m suite. Plan-only unless SUBMIT=YES; no existing jobs cancelled.
The historical ID model retains its attention encoder.
"""
from pathlib import Path

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as value_plans

GROUP = "corridor_obslinear_id_value_diagnostics_10m_3seeds"


def build_plans(repo):
    return value_plans(repo, scene_key="smac_corridor", map_name="corridor", group=GROUP)


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
