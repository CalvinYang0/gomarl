#!/usr/bin/env python3
"""Exercise the 3 objectives x 2 TD-path Advantage matrix."""
import logging
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


PROFILES = {
    "relation_advantage_masktd_augtd_teacheronly": (
        "margin", 1.0, 0.0, True
    ),
    "relation_advantage_actionadv_masktd_augtd_teacheronly": (
        "action_advantage", 1.0, 0.0, True
    ),
    "relation_advantage_qvalue_masktd_augtd_teacheronly": (
        "action_q", 1.0, 0.0, True
    ),
    "relation_advantage_augtd_nomasktd": (
        "margin", 0.0, 1.0, False
    ),
    "relation_advantage_actionadv_augtd_nomasktd": (
        "action_advantage", 0.0, 1.0, False
    ),
    "relation_advantage_qvalue_augtd_nomasktd": (
        "action_q", 0.0, 1.0, False
    ),
}


def main():
    logging.disable(logging.CRITICAL)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label, (objective, main_coef, nomask_coef, teacher_only) in (
        PROFILES.items()
    ):
        overrides = experiment_overrides(label)
        missing = set(overrides) - set(sacred)
        assert not missing, "{} has unregistered keys: {}".format(
            label, sorted(missing)
        )
        assert overrides["clean_advantage_objective"] == objective
        assert overrides["clean_main_td_coef"] == main_coef
        assert overrides["clean_nomask_td_auxiliary_coef"] == nomask_coef
        assert (
            overrides["clean_advantage_margin_teacher_only"]
            is teacher_only
        )
        assert overrides["clean_advantage_margin_auxiliary"] is True
        assert overrides["clean_mask_parameter_relation_coef"] == 0.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        assert (
            overrides["clean_random_drop_auxiliary_identity_warmup"]
            is True
        )
        assert overrides["clean_kl_auxiliary_force_main_open"] is False
        assert overrides["clean_importance_auxiliary_warmup_steps"] == 250000
        assert overrides["clean_advantage_margin_warmup_steps"] == 250000
        assert overrides["clean_dual_gate_test"] is True

        check(label)
        _, learner, batch, logger = make_case(label)
        assert learner.advantage_objective == objective
        learner.train(batch, t_env=10, episode_num=4)
        assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0.0
        learner.train(batch, t_env=500000, episode_num=5)
        for key in (
            "loss_advantage_margin",
            "train_gate/advantage_margin/margin_gain_mean",
            "train_gate/advantage_margin/action_advantage_gain_mean",
            "train_gate/advantage_margin/action_q_gain_mean",
        ):
            assert math.isfinite(logger.stats[key][-1][1]), (label, key)
    print("Advantage objective 3x2 matrix: OK")


if __name__ == "__main__":
    main()
