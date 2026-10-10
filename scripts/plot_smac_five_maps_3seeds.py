#!/usr/bin/env python3
"""One-shot six-map snapshot; keep this legacy filename for existing commands.

Includes 3m, 8m (8 vs 8), 8m_vs_9m, 5m6m, 3s5z and 6h8z.

Include every configured control cohort in the current six-map study, not
only Obs: corrected Linear-ID on 5m6m/8m9m/6h8z, VDN/QMIX on 5m6m/8m9m,
and explicit legacy attention-ID references. Never substitute architectures
or budgets, and never filter histories by Slurm state. Upload once per invocation.
"""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_3m_8m_obs_10m_3seeds import build_plans as marine_plans
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as value_plans
from plot_5m6m_head_condition_3seeds import build_plans as head_plans, LABELS
from ozstar_submit_linear_id_8m9m_6h8z_10m_3seeds import build_plans as id_plans
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_plans as mixer_plans

GROUP = "smac_six_maps_obs_head_comparison_3seeds"
OUTPUT_SUBDIR = "smac_six_maps_latest_3seeds"
SCENES = {
    "smac_3m": ("3m — 3 Marines vs. 3 Marines", "test_battle_won_mean"),
    "smac_8m": ("8m — 8 Marines vs. 8 Marines", "test_battle_won_mean"),
    "smac_8m9m": ("8m_vs_9m — 8 Marines vs. 9 Marines", "test_battle_won_mean"),
    "smac_5m6m": ("5 Marines vs. 6 Marines", "test_battle_won_mean"),
    "smac_3svs5z": ("3 Stalkers vs. 5 Zealots", "test_battle_won_mean"),
    "smac_6h8z": ("6 Hydralisks vs. 8 Zealots", "test_battle_won_mean"),
}
# A single inventory drives both discovery and rendering. No separate
# Obs-only panel whitelist can silently hide newly registered groups.
SCENE_MODELS = {
    "smac_3m": ("linear_baseline",),
    "smac_8m": ("linear_baseline",),
    "smac_8m9m": ("linear_baseline", "linear_id_baseline", "vdn", "qmix", "hyper_hypermarl_id"),
    "smac_5m6m": tuple(LABELS),
    "smac_3svs5z": ("linear_baseline", "hyper_hypermarl_id"),
    "smac_6h8z": ("linear_baseline", "linear_id_baseline", "hyper_hypermarl_id"),
}


def build_plans(repo):
    plans = marine_plans(repo) + head_plans(repo)
    for scene, map_name in (("smac_8m9m", "8m_vs_9m"),
                            ("smac_3svs5z", "3s_vs_5z"),
                            ("smac_6h8z", "6h_vs_8z")):
        for plan in value_plans(repo, scene, map_name, GROUP):
            if plan["label"] == "hyper_hypermarl_id":
                plan["inventory_note"] = "Legacy attention-ID reference; may be stopped/partial; not corrected Linear ID"
            plans.append(plan)
    plans.extend(id_plans(repo))
    plans.extend(mixer_plans(repo))
    for plan in plans:
        plan.setdefault("target_steps", 10000000)
    if len(plans) != 78 or len({p["job_name"] for p in plans}) != 78:
        raise RuntimeError("Expected 78 distinct runs across six maps")
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
    charts.LABELS = LABELS
    charts.SCENE_MODELS = SCENE_MODELS
    charts.build_plans = build_plans


def main():
    configure()
    charts.main()


if __name__ == "__main__":
    main()
