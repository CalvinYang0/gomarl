#!/usr/bin/env python3
"""One-off completed 10M obs three-seed figures; no running-only filtering."""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as suite_plans


def build_plans(repo):
    plans = [p for p in suite_plans(repo) if p["label"] == "linear_baseline"]
    assert len(plans) == 3
    return plans


if __name__ == "__main__":
    charts.GROUP = "completed_5m6m_obs_10m_valuediag"
    charts.DEFAULT_OUTPUT_SUBDIR = "completed_5m6m_obs_10m"
    charts.RESULTS_TITLE = "completed obs baseline seeds"
    charts.SCENES = {"smac_5m6m": ("5m vs. 6m", "test_battle_won_mean")}
    charts.LABELS = {"linear_baseline": ("Linear obs-based single-head", "#1f77b4")}
    charts.build_plans = build_plans
    charts.main()
