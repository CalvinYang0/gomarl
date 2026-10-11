#!/usr/bin/env python3
"""Four input controls, 5m_vs_6m, seeds 1/2/3, 10M; idempotent submission."""
import os
from pathlib import Path
import shlex

import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as submitter
from ozstar_submit_linear_single_head_suite import _load_profiles, _plan

GROUP = "5m6m_slot_input_controls_10m_12jobs"
VARIANTS = {
    "linear_static10_nokl": "smac_slot_static10_hypercond",
    "linear_group_forward_hierarchy": "smac_slot_forward_hierarchy_hypercond",
    "linear_group_forward_dualtd": "smac_slot_forward_dualtd_hypercond",
    "linear_group_reverse_hierarchy": "smac_slot_reverse_hierarchy_hypercond",
}
PLOT_LABELS = {
    "linear_static10_nokl": ("Linear static gate p0=0.1, no KL (10M)", "#bcbd22"),
    "linear_group_forward_hierarchy": ("Linear forward groups / hierarchical TD (10M)", "#00441b"),
    "linear_group_forward_dualtd": ("Linear forward groups / dual TD (10M)", "#08519c"),
    "linear_group_reverse_hierarchy": ("Linear reverse groups / hierarchical TD (10M)", "#a50f15"),
}
SMOKE_SCRIPTS = (
    "scripts/smoke_test_slot_input_controls.py",
    "scripts/smoke_test_submit_slot_input_controls.py",
    "scripts/smoke_test_value_diagnostics.py",
    "scripts/smoke_test_policy_importance.py",
    "scripts/smoke_test_periodic_battle_videos.py",
)


def build_plans(repo):
    profiles = _load_profiles(repo)
    plans = []
    for label, model in VARIANTS.items():
        for seed in (1, 2, 3):
            name = "smac_5m6m_{}_10m_s{}_slotcontrol".format(label, seed)
            plan = _plan(repo, profiles, "smac_5m6m", "5m_vs_6m", "smac",
                         "linear_baseline", seed, name,
                         "24G" if label == "linear_static10_nokl" else "48G", GROUP)
            plan["label"] = label
            plan["exports"].update(T_MAX="10050000", TEST_INTERVAL="10000", MODEL_TYPE=model)
            extras = dict(a.split("=", 1) for a in shlex.split(plan["exports"]["EXTRA_ARGS"]))
            extras.update(
                agent="slot_group_hyper", mac="slot_group_mac", learner="slot_group_learner",
                double_q="True", slot_group_interval="1000", slot_importance_samples="256",
                slot_importance_ema="0.8", slot_demote_ratio="0.5", slot_heatmap_interval="100000",
                test_value_diagnostics="True", test_value_diagnostics_interval="100000",
                test_visualizations_required="True", test_battle_videos="True",
                test_battle_video_interval="1000000", test_battle_video_episodes="10",
                test_policy_importance="True", test_policy_importance_interval="1000000",
                test_policy_importance_episodes="10", test_policy_importance_samples="64",
                # This new path has its own fixed-hidden slot diagnostic;
                # the old ungated-single-head attribution is not applicable.
                test_hyper_obs_importance="False")
            if label == "linear_static10_nokl":
                extras.update(clean_hard_gate_initial_keep_probability="0.1",
                              clean_hard_gate_threshold="0.1", clean_dynamic_branch_gate_static="True",
                              clean_dynamic_branch_gate_warmup_steps="0")
            plan["exports"]["EXTRA_ARGS"] = " ".join("{}={}".format(k, v) for k, v in extras.items())
            plan["sbatch_args"] = ["--time=2-00:00:00" if a.startswith("--time=") else a
                                    for a in plan["sbatch_args"]]
            plan["target_steps"] = 10000000
            plan["inventory_note"] = "New slot-control cohort; important-head behavior and Double-Q; no historical merging"
            plans.append(plan)
    if len(plans) != len({p["job_name"] for p in plans}) or len(plans) != 12:
        raise RuntimeError("Expected exactly twelve distinct slot-control runs")
    return plans


def validate_installed_maps():
    from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map
    validate_installed_map()


def main():
    if os.environ.get("SUBMIT") == "YES":
        if submitter.home_quota_free_gib() < 5:
            raise RuntimeError("Require at least 5 GiB free before 12 media-enabled jobs; no jobs submitted")
        validate_installed_maps()
    replacement = dict(GROUP=GROUP, SUITE_FILE=Path(__file__), SMOKE_SCRIPTS=SMOKE_SCRIPTS,
                       build_plans=build_plans)
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
