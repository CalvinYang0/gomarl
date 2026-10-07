#!/usr/bin/env python3
"""Plot the fresh 10M value-diagnostic baselines on Corridor and 5m6m.

Only exact value-diagnostic suite run names are selected; no historical
5M runs or Counter/MMM2 jobs are mixed in. Missing seeds are reported, not filled.
Default uploads snapshot figures/inventory to W&B; --no-upload is local only.
"""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as five_plans
from ozstar_submit_corridor_value_diagnostics_10m_3seeds import build_plans as obs_id_plans
from ozstar_submit_corridor_vdn_qmix_10m_3seeds import build_plans as fixed_plans


def build_plans(repo):
    plans = five_plans(repo) + obs_id_plans(repo) + fixed_plans(repo)
    assert len(plans) == 18 and len({p["job_name"] for p in plans}) == 18
    return plans


if __name__ == "__main__":
    charts.GROUP = "corridor_5m6m_baselines_10m_3seeds_valuediag"
    charts.DEFAULT_OUTPUT_SUBDIR = "corridor_5m6m_baselines_10m"
    charts.RESULTS_TITLE = "current 10M baseline results"
    charts.SCENES = {
        "smac_5m6m": ("5m vs. 6m", "test_battle_won_mean"),
        "smac_corridor": ("Corridor", "test_battle_won_mean"),
    }
    charts.LABELS = {
        "linear_baseline": ("Linear obs-based single-head", "#1f77b4"),
        "hyper_hypermarl_id": ("ID-based (historical attention)", "#ff7f0e"),
        "vdn": ("VDN (fixed agent head)", "#2ca02c"),
        "qmix": ("QMIX (fixed agent head)", "#9467bd"),
    }
    charts.build_plans = build_plans
    charts.main()
