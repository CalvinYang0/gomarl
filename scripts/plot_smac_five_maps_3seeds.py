#!/usr/bin/env python3
"""One-shot seven-map snapshot; keep this legacy filename for existing commands.

Includes 3m, 8m (8 vs 8), 8m_vs_9m, 5m6m, 3s_vs_5z, 6h8z and MMM2.

Include 30 current cohorts and two explicitly labelled 5m6m 5M mixer fallbacks.
Exclude old replica and legacy attention-ID/5M ID references. Never mix
cohorts or budgets, or filter histories by Slurm state. Upload once per invocation.
"""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_3m_8m_obs_10m_3seeds import build_plans as marine_plans
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as value_plans
from plot_5m6m_head_condition_3seeds import build_plans as head_plans, LABELS, FALLBACK_MODELS
from ozstar_submit_linear_id_8m9m_6h8z_10m_3seeds import build_plans as id_plans
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_plans as mixer_plans
from ozstar_submit_head_input_24jobs import build_plans as covered_plans, PLOT_LABELS
from ozstar_submit_8m9m_health_only_10m_3seeds import build_plans as health8_plans

GROUP = "smac_seven_maps_obs_head_comparison_3seeds"
OUTPUT_SUBDIR = "smac_seven_maps_latest_3seeds"
SCENES = {
    "smac_3m": ("3m — 3 Marines vs. 3 Marines", "test_battle_won_mean"),
    "smac_8m": ("8m — 8 Marines vs. 8 Marines", "test_battle_won_mean"),
    "smac_8m9m": ("8m_vs_9m — 8 Marines vs. 9 Marines", "test_battle_won_mean"),
    "smac_5m6m": ("5 Marines vs. 6 Marines", "test_battle_won_mean"),
    "smac_3svs5z": ("3 Stalkers vs. 5 Zealots", "test_battle_won_mean"),
    "smac_6h8z": ("6 Hydralisks vs. 8 Zealots", "test_battle_won_mean"),
    "smac_mmm2": ("MMM2 — Marines, Marauders and Medivac", "test_battle_won_mean"),
}
# A single inventory drives both discovery and rendering. No separate
# Obs-only panel whitelist can silently hide newly registered groups.
SCENE_MODELS = {
    "smac_3m": ("linear_baseline",),
    "smac_8m": ("linear_baseline",),
    "smac_8m9m": ("linear_baseline", "linear_id_baseline", "vdn", "qmix",
                   "linear_ones_baseline_vizcoverage", "linear_health_baseline"),
    "smac_5m6m": tuple(LABELS),
    "smac_3svs5z": tuple(PLOT_LABELS),
    "smac_6h8z": ("linear_baseline", "linear_id_baseline"),
    "smac_mmm2": tuple(PLOT_LABELS),
}


def build_plans(repo):
    plans = marine_plans(repo) + head_plans(repo)
    for scene, map_name in (("smac_8m9m", "8m_vs_9m"),
                            ("smac_6h8z", "6h_vs_8z")):
        for plan in value_plans(repo, scene, map_name, GROUP):
            if plan["label"] == "linear_baseline":
                plans.append(plan)
    plans.extend(id_plans(repo))
    plans.extend(mixer_plans(repo))
    plans.extend(health8_plans(repo))
    # 5m6m's fresh ID cohort is already registered by head_plans. All other
    # fresh cohorts are selected without reading superseded replicas.
    plans.extend(dict(p, label=p["label"] + "_vizcoverage",
                      inventory_note="New visualization-covered 10M replica; not merged with older cohorts")
                 for p in covered_plans(repo) if p["scene"] != "smac_5m6m")
    for plan in plans:
        plan.setdefault("target_steps", 10000000)
    if len(plans) != 96 or len({p["job_name"] for p in plans}) != 96:
        raise RuntimeError("Expected 96 distinct runs across seven maps, including six labelled 5M fallback histories")
    for scene in SCENES:
        models = set(SCENE_MODELS[scene])
        selected = [p for p in plans if p["scene"] == scene]
        if scene in {"smac_8m", "smac_8m9m"}:
            expected_map = "8m" if scene == "smac_8m" else "8m_vs_9m"
            if any(p["map_name"] != expected_map for p in selected):
                raise RuntimeError("Marine map mismatch for " + scene)
        if {p["label"] for p in selected} != models:
            raise RuntimeError("Unexpected models for " + scene)
        for model in models:
            if {p["seed"] for p in selected if p["label"] == model} != {1, 2, 3}:
                raise RuntimeError("Expected three seeds for " + scene + ": " + model)
    return plans


def configure():
    charts.GROUP = GROUP
    charts.DEFAULT_OUTPUT_SUBDIR = OUTPUT_SUBDIR
    charts.RESULTS_TITLE = "current three-seed snapshot"
    charts.REPORT_SEED_COVERAGE = True
    charts.TARGET_STEPS = 10000000
    charts.SCENES = SCENES
    charts.LABELS = dict(LABELS, **PLOT_LABELS,
                         linear_id_baseline=("Linear ID-based (10M)", "#e6550d"))
    charts.SCENE_MODELS = SCENE_MODELS
    charts.SCENE_FALLBACK_MODELS = FALLBACK_MODELS
    charts.build_plans = build_plans


def main():
    configure()
    charts.main()


if __name__ == "__main__":
    main()
