#!/usr/bin/env python3
"""11 current controls/baselines; cloud reads only six 10M VDN/QMIX runs.

Other histories come from local W&B/Sacred. Failed cloud discovery is fatal,
never silently presented as proof that a 10M baseline did not run.
"""
import plot_5m6m_head_condition_3seeds as previous
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as new_plans, PLOT_LABELS

charts = previous.charts
_QUERY = charts.cloud_run_index
MODELS = (
    "linear_baseline", "linear_id_baseline_vizcoverage", "vdn", "qmix",
    "linear_static10_nokl", "linear_group_forward_hierarchy",
    "linear_group_forward_dualtd", "linear_group_reverse_hierarchy",
    *PLOT_LABELS, "historical_vdn", "historical_qmix",
)


def build_plans(repo):
    plans = [p for p in previous.build_plans(repo) if p["label"] in MODELS]
    plans.extend(new_plans(repo))
    assert len(plans) == 39 and len({p["job_name"] for p in plans}) == 39
    return plans


def mixer_cloud_index(project, wanted):
    names = {"smac_5m6m_{}_baseline_10m_s{}_valuediag".format(method, seed)
             for method in ("vdn", "qmix") for seed in (1, 2, 3)} & wanted
    print("READ CLOUD: only {} exact 10M VDN/QMIX names; no other histories or full-project scan".format(len(names)), flush=True)
    return _QUERY(project, names, allow_full_scan=False)


def configure():
    previous.configure()
    charts.GROUP = "smac_5m6m_latest_gate_group_baselines_3seeds"
    charts.DEFAULT_OUTPUT_SUBDIR = "5m6m_latest_gate_group_baselines_3seeds"
    charts.RESULTS_TITLE = "latest agent-head and mixer-input controls / baselines"
    labels = dict(previous.LABELS, **PLOT_LABELS)
    charts.LABELS = {m: labels[m] for m in MODELS}
    charts.SCENE_MODELS = {"smac_5m6m": MODELS}
    charts.build_plans = build_plans
    charts.cloud_run_index = mixer_cloud_index
    charts.upload = previous._BASE_UPLOAD


if __name__ == "__main__":
    configure()
    charts.main()
