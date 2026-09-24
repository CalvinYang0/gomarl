#!/usr/bin/env python3
"""Verify ten HyperSelect QME trials against one historical control."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case
from smoke_test_trans9_multiscene import check_smac_semantics


LABELS = (
    "hyperselect_qme_joint_value",
    "hyperselect_qme_td_quality",
    "hyperselect_qme_action_rank",
    "hyperselect_qme_stable_teacher",
    "hyperselect_qme_dynamic_readiness",
    "hyperselect_qme_action_q_scaled",
    "hyperselect_qme_action_q_episode_mean",
    "hyperselect_qme_full_behavior",
    "hyperselect_qme_mixed_behavior",
    "hyperselect_qme_open_win_ready",
)

REFERENCE = "relation_advantage_qvalue_augtd_nomasktd"
ALLOWED_DIFFERENCES = {
    "hyperselect_qme_joint_value": {
        "clean_model_type", "clean_advantage_objective",
    },
    "hyperselect_qme_td_quality": {
        "clean_model_type", "clean_advantage_objective",
    },
    "hyperselect_qme_action_rank": {
        "clean_model_type", "clean_advantage_objective",
    },
    "hyperselect_qme_stable_teacher": {
        "clean_model_type",
        "clean_advantage_stable_target_teacher",
        "clean_nomask_independent_target",
    },
    "hyperselect_qme_dynamic_readiness": {
        "clean_model_type",
        "clean_advantage_dynamic_readiness",
        "clean_advantage_margin_warmup_steps",
        "clean_advantage_margin_ramp_steps",
    },
    "hyperselect_qme_action_q_scaled": {
        "clean_model_type", "clean_advantage_objective",
    },
    "hyperselect_qme_action_q_episode_mean": {
        "clean_model_type", "clean_advantage_objective",
    },
    "hyperselect_qme_full_behavior": {
        "clean_model_type", "clean_train_behavior_gate_mode",
    },
    "hyperselect_qme_mixed_behavior": {
        "clean_model_type", "clean_train_behavior_gate_mode",
    },
    "hyperselect_qme_open_win_ready": {
        "clean_model_type",
        "clean_advantage_open_win_readiness",
        "clean_advantage_margin_warmup_steps",
        "clean_advantage_margin_ramp_steps",
    },
}


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    reference = experiment_overrides(REFERENCE)
    for label in LABELS:
        for domain in ("grf", "smac"):
            overrides = experiment_overrides(label, domain)
            missing = set(overrides) - set(sacred)
            assert not missing, (label, domain, sorted(missing))
        overrides = experiment_overrides(label)
        assert overrides["clean_main_td_coef"] == 0.0
        assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        differences = {
            key for key in overrides
            if overrides[key] != reference[key]
        }
        assert differences == ALLOWED_DIFFERENCES[label], (
            label, sorted(differences)
        )
        expected_behavior = {
            "hyperselect_qme_full_behavior": "full",
            "hyperselect_qme_mixed_behavior": "mixed",
        }.get(label, "masked")
        assert (
            overrides["clean_train_behavior_gate_mode"]
            == expected_behavior
        )
        stable = label == "hyperselect_qme_stable_teacher"
        dynamic = label == "hyperselect_qme_dynamic_readiness"
        open_ready = label == "hyperselect_qme_open_win_ready"
        assert overrides["clean_advantage_stable_target_teacher"] is stable
        assert overrides["clean_nomask_independent_target"] is stable
        assert overrides["clean_advantage_dynamic_readiness"] is dynamic
        assert overrides["clean_advantage_open_win_readiness"] is open_ready
        assert overrides["clean_advantage_open_win_threshold"] == 0.1
        readiness = dynamic or open_ready
        assert overrides["clean_advantage_margin_warmup_steps"] == (
            0 if readiness else 250000
        )
        assert overrides["clean_advantage_margin_ramp_steps"] == (
            0 if readiness else 250000
        )
        if label == "hyperselect_qme_action_rank":
            assert overrides["clean_advantage_objective"] == "action_q_rank"

        check(label)
        _, learner, batch, logger = make_case(label)
        learner.train(batch, t_env=500000, episode_num=3)
        prefix = "train_gate/advantage_margin/"
        assert logger.stats[prefix + "action_agreement"]
        if dynamic:
            assert logger.stats[prefix + "readiness_weight"][-1][1] > 0.0
            assert logger.stats[prefix + "positive_return_ema"][-1][1] > 0.0
        if open_ready:
            assert logger.stats[prefix + "open_win_readiness_weight"][-1][1] == 0.0
            learner.update_qme_open_win_rate(0.1)
            assert learner.advantage_open_win_ready is False
            learner.update_qme_open_win_rate(0.1001)
            assert learner.advantage_open_win_ready is True
            learner.update_qme_open_win_rate(0.0)
            assert learner.advantage_open_win_ready is True
        if label == "hyperselect_qme_joint_value":
            assert logger.stats[prefix + "joint_masked_q"]
            assert logger.stats[prefix + "joint_full_q"]
            assert logger.stats[prefix + "joint_q_gain_mean"]
        if label == "hyperselect_qme_td_quality":
            assert logger.stats[prefix + "td_quality_gain_mean"]
        if stable:
            assert logger.stats["train_gate/qme_target/full_mean"]
            assert logger.stats["train_gate/qme_target/masked_mean"]
        if label == "hyperselect_qme_action_q_scaled":
            assert logger.stats[prefix + "q_scale"][-1][1] >= 1.0
            assert logger.stats[prefix + "raw_action_q_loss"]
            assert logger.stats[prefix + "scaled_action_q_loss"]
        if label == "hyperselect_qme_action_q_episode_mean":
            assert logger.stats[prefix + "episode_equal_action_q_loss"]
            assert logger.stats[prefix + "timestep_equal_action_q_loss"]

        for scene in ("3s5z_vs_3s6z", "5m_vs_6m"):
            check_smac_semantics(scene, label)

    print(
        "Ten controlled QME variants passed: objectives, teacher, readiness, "
        "scaling, aggregation and replay behaviour"
    )


if __name__ == "__main__":
    main()
