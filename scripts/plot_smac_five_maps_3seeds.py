#!/usr/bin/env python3
"""One-shot six-map snapshot; keep this legacy filename for existing commands.

Includes 3m, 8m (8 vs 8), 8m_vs_9m, 5m6m, 3s5z and 6h8z.

All maps include the exact obs-based baseline runs. Only 5m6m also includes
the new linear-ID, linear-only KL80 and global-state experiments. Completed
and currently running histories are both eligible; Slurm state is not a
filter. No historical attention-ID substitution. Upload once per invocation.
"""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_3m_8m_obs_10m_3seeds import build_plans as marine_plans
from ozstar_submit_5m6m_value_diagnostics_10m_3seeds import build_plans as value_plans
from plot_5m6m_head_condition_3seeds import build_plans as head_plans, LABELS

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


def build_plans(repo):
    plans = marine_plans(repo) + head_plans(repo)
    for scene, map_name in (("smac_8m9m", "8m_vs_9m"),
                            ("smac_3svs5z", "3s_vs_5z"),
                            ("smac_6h8z", "6h_vs_8z")):
        plans.extend(p for p in value_plans(repo, scene, map_name, GROUP)
                     if p["label"] == "linear_baseline")
    for plan in plans:
        plan.setdefault("target_steps", 10000000)
    if len(plans) != 27 or len({p["job_name"] for p in plans}) != 27:
        raise RuntimeError("Expected 27 distinct runs across six maps")
    for scene in SCENES:
        models = set(LABELS) if scene == "smac_5m6m" else {"linear_baseline"}
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
    charts.SCENE_MODELS = {scene: tuple(LABELS) if scene == "smac_5m6m"
                           else ("linear_baseline",) for scene in SCENES}
    charts.build_plans = build_plans


def main():
    configure()
    charts.main()


if __name__ == "__main__":
    main()
