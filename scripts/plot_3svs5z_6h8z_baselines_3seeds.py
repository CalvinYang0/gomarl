#!/usr/bin/env python3
"""Upload fresh three-seed snapshots for 3s_vs_5z and 6h_vs_8z.

Latest exact-name attempt per seed; stopped ID histories remain available.
Uses test win rate. ID is the historical attention configuration. Uploads
aggregate and individual-seed PNGs plus CSV data/inventory by default.
--local-only disables cloud reads while keeping figure uploads enabled.
"""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as value_plans

GROUP = "3svs5z_6h8z_obs_attention_id_10m_3seeds"
SCENES = (
    ("smac_3svs5z", "3s_vs_5z", "3 Stalkers vs. 5 Zealots"),
    ("smac_6h8z", "6h_vs_8z", "6 Hydralisks vs. 8 Zealots"),
)


def build_plans(repo):
    plans = []
    for scene, map_name, _ in SCENES:
        for plan in value_plans(repo, scene_key=scene, map_name=map_name, group=GROUP):
            if plan["label"] == "hyper_hypermarl_id":
                plan["inventory_note"] = "Historical attention ID configuration; not Linear ID"
            plans.append(plan)
    assert len(plans) == len({p["job_name"] for p in plans}) == 12
    return plans


def configure():
    charts.GROUP = GROUP
    charts.DEFAULT_OUTPUT_SUBDIR = "3svs5z_6h8z_baselines_latest_3seeds"
    charts.RESULTS_TITLE = "current obs / attention ID results"
    charts.REPORT_SEED_COVERAGE = True
    charts.SCENES = {scene: (title, "test_battle_won_mean") for scene, _, title in SCENES}
    charts.LABELS = {
        "linear_baseline": ("Linear obs-based single-head", "#1f77b4"),
        "hyper_hypermarl_id": ("ID-based (historical attention)", "#ff7f0e"),
    }
    charts.build_plans = build_plans


if __name__ == "__main__":
    configure()
    charts.main()
