#!/usr/bin/env python3
"""Three ordinary-agent QMIX state controls; 5m6m, three seeds, 10M."""
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_8m9m_vdn_qmix_10m_3seeds import build_fixed_head_plans

GROUP = "5m6m_qmix_mixer_input_controls_10m_9jobs"
MIN_HOME_FREE_GIB = 2.0
VARIANTS = {
    "qmix_state_static10": "static_gate",
    "qmix_state_group_qchange": "group_qchange",
    "qmix_state_group_tderror": "group_tderror",
}
PLOT_LABELS = {
    "qmix_state_static10": ("QMIX state static gate p0=0.1, no KL (10M)", "#bcbd22"),
    "qmix_state_group_qchange": ("QMIX state groups / Q-change / dual TD (10M)", "#08519c"),
    "qmix_state_group_tderror": ("QMIX state groups / TD-error increase / dual TD (10M)", "#a50f15"),
}
SMOKE_SCRIPTS = (
    "scripts/smoke_test_mixer_input_controls.py",
    "scripts/smoke_test_submit_mixer_input_controls.py",
    "scripts/smoke_test_plot_mixer_input_controls.py",
    "scripts/smoke_test_periodic_battle_videos.py",
    "scripts/smoke_test_policy_importance.py",
)


def build_plans(repo):
    templates = [p for p in build_fixed_head_plans(repo, "smac_5m6m", "5m_vs_6m", GROUP) if p["label"] == "qmix"]
    plans = []
    from copy import deepcopy
    for label, variant in VARIANTS.items():
        for template in templates:
            p = deepcopy(template)
            old = p["job_name"]
            name = "smac_5m6m_{}_10m_s{}_mixercontrol".format(label, p["seed"])
            p.update(label=label, job_name=name, run_name=name, target_steps=10000000,
                     inventory_note="Ordinary fixed agent head; only global-state mixer inputs controlled; no historical merging")
            p["exports"]["RUN_NAME"] = name
            extra = dict(a.split("=", 1) for a in shlex.split(p["exports"]["EXTRA_ARGS"]))
            extra.update(learner="mixer_input_learner", mixer_input_control=variant,
                         mixer_input_initial_probability="0.1", mixer_input_gate_threshold="0.1",
                         mixer_input_gate_temperature="0.5", double_q="True",
                         slot_group_interval="1000", slot_importance_samples="256",
                         slot_importance_ema="0.8", slot_demote_ratio="0.5", slot_heatmap_interval="100000",
                         test_policy_importance="True", test_policy_importance_interval="1000000",
                         test_policy_importance_episodes="10", test_policy_importance_samples="64",
                         test_visualizations_required="True", test_hyper_obs_importance="False")
            p["exports"]["EXTRA_ARGS"] = " ".join("{}={}".format(k, v) for k, v in extra.items())
            p["sbatch_args"] = [a.replace(old, name) for a in p["sbatch_args"]]
            plans.append(p)
    if len(plans) != len({p["job_name"] for p in plans}) or len(plans) != 9:
        raise RuntimeError("Expected nine distinct mixer-input jobs")
    return plans


def main():
    if os.environ.get("SUBMIT") == "YES":
        free = submitter.home_quota_free_gib()
        print("/home quota free: {:.2f} GiB; minimum {:.2f} GiB".format(free, MIN_HOME_FREE_GIB), flush=True)
        if free < MIN_HOME_FREE_GIB:
            raise RuntimeError("Require at least 2 GiB before nine media-enabled jobs; no jobs submitted")
        print("Capacity warning: 2 GiB is only a startup threshold; videos and offline data will keep growing.", flush=True)
        from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
        validate_installed_map()
    replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__), SMOKE_SCRIPTS=SMOKE_SCRIPTS, build_plans=build_plans)
    original = {key: getattr(submitter, key) for key in replacement}
    try:
        for key, value in replacement.items():
            setattr(submitter, key, value)
        submitter.main()
    finally:
        for key, value in original.items():
            setattr(submitter, key, value)


if __name__ == "__main__":
    main()
