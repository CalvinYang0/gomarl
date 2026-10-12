#!/usr/bin/env python3
"""Only the nine state-control runs and matched QMIX (explicit 5M fallback)."""
import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as control_plans, PLOT_LABELS
from ozstar_submit_5m6m_vdn_qmix_10m_3seeds import build_plans as baseline_plans

_CLOUD_QUERY = charts.cloud_run_index


def build_plans(repo):
    plans = control_plans(repo)
    plans.extend(dict(p, target_steps=10000000) for p in baseline_plans(repo) if p["label"] == "qmix")
    plans.extend(dict(scene="smac_5m6m", map_name="5m_vs_6m", label="historical_qmix", seed=seed,
                      job_name="smac_5m6m_paper_qmix_5m_s{}".format(seed), target_steps=5000000,
                      inventory_note="Historical 5M QMIX only when no matched 10M seed has data; never mixed")
                 for seed in (1, 2, 3))
    return plans


def configure():
    charts.GROUP = "smac_5m6m_qmix_mixer_input_controls_3seeds"
    charts.DEFAULT_OUTPUT_SUBDIR = "5m6m_qmix_mixer_input_controls_3seeds"
    charts.RESULTS_TITLE = "ordinary agent / mixer state gate and groups"
    charts.REPORT_SEED_COVERAGE = True
    charts.SCENES = {"smac_5m6m": ("5 Marines vs. 6 Marines", "test_battle_won_mean")}
    charts.LABELS = dict(PLOT_LABELS, qmix=("QMIX ordinary fixed head (10M)", "#7f7f7f"),
                         historical_qmix=("QMIX historical 5M control", "#7f7f7f"))
    charts.SCENE_MODELS = {"smac_5m6m": tuple(charts.LABELS)}
    charts.SCENE_FALLBACK_MODELS = {"smac_5m6m": {"qmix": "historical_qmix"}}
    charts.build_plans = build_plans
    charts.cloud_run_index = lambda project, wanted: _CLOUD_QUERY(project, wanted, allow_full_scan=False)


if __name__ == "__main__":
    configure()
    charts.main()
