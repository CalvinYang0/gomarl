#!/usr/bin/env python3
"""Exercise five isolated HyperSelect QME improvements."""
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
)


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label in LABELS:
        for domain in ("grf", "smac"):
            overrides = experiment_overrides(label, domain)
            missing = set(overrides) - set(sacred)
            assert not missing, (label, domain, sorted(missing))
        overrides = experiment_overrides(label)
        assert overrides["clean_main_td_coef"] == 1.0
        assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
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
        assert overrides["clean_advantage_stable_target_teacher"] is stable
        assert overrides["clean_nomask_independent_target"] is stable
        assert overrides["clean_advantage_dynamic_readiness"] is dynamic

        check(label)
        _, learner, batch, logger = make_case(label)
        learner.train(batch, t_env=500000, episode_num=3)
        prefix = "train_gate/advantage_margin/"
        assert logger.stats[prefix + "action_agreement"]
        if dynamic:
            assert logger.stats[prefix + "readiness_weight"][-1][1] > 0.0
            assert logger.stats[prefix + "positive_return_ema"][-1][1] > 0.0
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
        "QME variants passed: joint-Q, TD-quality, action ranking, stable "
        "teacher, dynamic readiness, raw-Q scale normalization and "
        "episode-equal aggregation"
    )


if __name__ == "__main__":
    main()
