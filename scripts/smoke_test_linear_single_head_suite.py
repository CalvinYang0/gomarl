#!/usr/bin/env python3
"""Validate the four-scene Linear single-head baseline and Counter studies."""

from pathlib import Path
import sys

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import (  # noqa: E402
    ALL_PROFILES,
    experiment_overrides,
    model_type_for,
)
from smoke_test_counter_transformer_nine import make_case  # noqa: E402


SCENES = (
    "academy_counterattack_easy",
    "academy_pass_and_shoot_with_keeper",
    "5m_vs_6m",
    "MMM2",
)
QME = {
    "linear_qme_action_q": ("action_q", "masked"),
    "linear_qme_action_q_episode_mean": (
        "action_q_episode_mean", "masked"
    ),
    "linear_qme_td_quality": ("td_quality", "masked"),
    "linear_qme_joint_value": ("joint_q", "masked"),
    "linear_qme_action_q_scaled": ("action_q_scaled", "masked"),
    "linear_qme_dynamic_readiness": ("action_q", "masked"),
    "linear_qme_open_win_readiness": ("action_q", "masked"),
    "linear_qme_full_behavior": ("action_q", "full"),
}


def check_baselines():
    for scene in SCENES:
        domain = "smac" if not scene.startswith("academy_") else "grf"
        expected_model = model_type_for("linear_baseline", domain)
        assert "single_linear" in expected_model
        mac, learner, batch, logger = make_case("linear_baseline", scene)
        capturer = mac.agent.rpg_relation_capturer
        assert capturer.relation_encoder_style == "linear_only"
        assert len(capturer.transformer_layers) == 0
        assert capturer.dynamic_branch_gate is None
        assert not learner.random_drop_auxiliary_active
        assert not learner.advantage_margin_auxiliary_active
        learner.train(batch, t_env=10, episode_num=1)
        assert logger.stats["loss_td"][-1][1] >= 0.0
        assert all(th.isfinite(parameter).all() for parameter in mac.parameters())


def check_kl_forms():
    direct_mac, direct_learner, direct_batch, _ = make_case(
        "linear_bayesg_kl80_keep"
    )
    direct = direct_mac.agent.rpg_relation_capturer
    assert direct.relation_encoder_style == "linear_only"
    assert direct_learner.gate_regularization_active
    assert not direct_learner.random_drop_auxiliary_active
    direct_learner.train(direct_batch, t_env=300000, episode_num=1)

    aux_mac, aux_learner, aux_batch, aux_logger = make_case(
        "linear_obs_gate_kl80aux_multiply"
    )
    aux = aux_mac.agent.rpg_relation_capturer
    assert aux.relation_encoder_style == "linear_only"
    assert not aux_learner.gate_regularization_active
    assert aux_learner.kl80_random_drop_auxiliary
    assert aux_learner.random_drop_auxiliary_combine_mode == "multiply"
    aux_learner.train(aux_batch, t_env=300000, episode_num=1)
    assert aux_logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0.0


def check_qme():
    control = experiment_overrides("linear_bayesg_nomasktd_control")
    assert control["clean_main_td_coef"] == 1.0
    assert control["clean_nomask_td_auxiliary_coef"] == 1.0
    assert not control["clean_advantage_margin_auxiliary"]
    for label, (objective, behavior) in QME.items():
        flags = ALL_PROFILES[label]
        overrides = experiment_overrides(label)
        assert flags["branch"] == "linear"
        assert flags["kl"] and not flags.get("aux")
        assert not flags.get("memory_efficient_multi_path", False)
        assert overrides["clean_main_td_coef"] == 1.0
        assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert not overrides["clean_advantage_margin_teacher_only"]
        assert overrides["clean_advantage_objective"] == objective
        assert overrides["clean_train_behavior_gate_mode"] == behavior
        assert overrides["clean_dual_gate_test"] == (
            label == "linear_qme_open_win_readiness"
        )
        mac, learner, batch, logger = make_case(label)
        assert mac.agent.rpg_relation_capturer.relation_encoder_style == (
            "linear_only"
        )
        assert not learner.memory_efficient_multi_path
        learner.train(batch, t_env=500000, episode_num=1)
        assert logger.stats["loss_advantage_margin"][-1][1] == (
            logger.stats["loss_advantage_margin"][-1][1]
        )


def main():
    th.set_num_threads(1)
    th.manual_seed(41)
    check_baselines()
    check_kl_forms()
    check_qme()
    print(
        "Linear single-head suite passed: four scenes, KL80 forms, QME and "
        "masked/full sampling"
    )


if __name__ == "__main__":
    main()
