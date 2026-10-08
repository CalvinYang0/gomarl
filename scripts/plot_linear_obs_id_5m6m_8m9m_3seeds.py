#!/usr/bin/env python3
"""Upload latest three-seed snapshots for matched Linear obs/ID baselines.

Historical attention ID is a separately labelled reference. Missing fresh-ID
histories stay missing, never replaced by historical ID data. Produces PNG/PDF
figures and uploads PNGs plus CSV inventories through the existing uploader.
"""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as old_plans
from ozstar_submit_linear_id_5m6m_8m9m_10m_3seeds import build_plans as linear_id_plans

GROUP = "linear_obs_id_5m6m_8m9m_10m_3seeds"
SCENES = (("smac_5m6m", "5m_vs_6m", "5 Marines vs. 6 Marines"),
          ("smac_8m9m", "8m_vs_9m", "8 Marines vs. 9 Marines"))


def build_plans(repo):
    plans = linear_id_plans(repo)
    for scene, map_name, _ in SCENES:
        for plan in old_plans(repo, scene_key=scene, map_name=map_name, group=GROUP):
            if plan["label"] == "hyper_hypermarl_id":
                plan["inventory_note"] = "Historical attention ID reference; not matched Linear ID"
            plans.append(plan)
    assert len(plans) == len({p["job_name"] for p in plans}) == 18
    return plans


def configure():
    charts.GROUP = GROUP
    charts.DEFAULT_OUTPUT_SUBDIR = "linear_obs_id_5m6m_8m9m_10m"
    charts.RESULTS_TITLE = "Linear baselines + legacy ID reference"
    charts.REPORT_SEED_COVERAGE = True
    charts.SCENES = {scene: (title, "test_battle_won_mean") for scene, _, title in SCENES}
    charts.LABELS = {
        "linear_baseline": ("Linear obs-based", "#1f77b4"),
        "linear_id_baseline": ("Linear ID-based", "#ff7f0e"),
        "hyper_hypermarl_id": ("Legacy attention ID (reference)", "#7f7f7f"),
    }
    charts.build_plans = build_plans


if __name__ == "__main__":
    configure()
    charts.main()
