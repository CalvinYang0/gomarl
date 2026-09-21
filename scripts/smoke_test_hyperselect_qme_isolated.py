#!/usr/bin/env python3
"""Verify the five QME trials change one factor from the old paper model."""
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


REFERENCE = "relation_advantage_qvalue_augtd_nomasktd"
LABELS = (
    "hyperselect_qme_joint_value_isolated",
    "hyperselect_qme_td_quality_isolated",
    "hyperselect_qme_action_rank_isolated",
    "hyperselect_qme_stable_teacher_isolated",
    "hyperselect_qme_dynamic_readiness_isolated",
)
OBJECTIVES = {
    "hyperselect_qme_joint_value_isolated": "joint_q",
    "hyperselect_qme_td_quality_isolated": "td_quality",
    "hyperselect_qme_action_rank_isolated": "action_q_rank",
    "hyperselect_qme_stable_teacher_isolated": "action_q",
    "hyperselect_qme_dynamic_readiness_isolated": "action_q",
}
ALLOWED_DIFFERENCES = {
    "hyperselect_qme_joint_value_isolated": {"clean_model_type", "clean_advantage_objective"},
    "hyperselect_qme_td_quality_isolated": {"clean_model_type", "clean_advantage_objective"},
    "hyperselect_qme_action_rank_isolated": {"clean_model_type", "clean_advantage_objective"},
    "hyperselect_qme_stable_teacher_isolated": {
        "clean_model_type",
        "clean_advantage_stable_target_teacher",
        "clean_nomask_independent_target",
    },
    "hyperselect_qme_dynamic_readiness_isolated": {
        "clean_model_type",
        "clean_advantage_open_win_readiness",
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
        overrides = experiment_overrides(label)
        assert not (set(overrides) - set(sacred))
        assert overrides["clean_main_td_coef"] == 0.0
        assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        assert overrides["clean_advantage_objective"] == OBJECTIVES[label]
        differences = {
            key for key in overrides
            if overrides[key] != reference[key]
        }
        assert differences == ALLOWED_DIFFERENCES[label], (
            label, sorted(differences)
        )
        check(label)
        _, learner, batch, logger = make_case(label)
        learner.train(batch, t_env=500000, episode_num=3)
        prefix = "train_gate/advantage_margin/"
        assert logger.stats[prefix + "action_agreement"]
        if label.endswith("dynamic_readiness_isolated"):
            assert learner.advantage_open_win_ready is False
            learner.update_qme_open_win_rate(0.1001)
            assert learner.advantage_open_win_ready is True
    print("Five isolated Counter QME variants match the old paper TD paths")


if __name__ == "__main__":
    main()
