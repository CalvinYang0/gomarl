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

# Only genuine QME objective choices belong in the matched KL80 matrix.
# Readiness heuristics and behavior-sampling changes remain available as
# historical profiles, but are deliberately excluded from this experiment.
MATRIX_QME = {
    label: QME[label]
    for label in (
        "linear_qme_action_q",
        "linear_qme_action_q_episode_mean",
        "linear_qme_td_quality",
        "linear_qme_joint_value",
        "linear_qme_action_q_scaled",
    )
}

QME_FRAMEWORKS = {
    "directkl": (True, False),
    "auxmultiply": (False, True),
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
    for scene in SCENES:
        direct_mac, direct_learner, direct_batch, _ = make_case(
            "linear_bayesg_kl80_keep", scene
        )
        direct = direct_mac.agent.rpg_relation_capturer
        assert direct.relation_encoder_style == "linear_only"
        assert direct.dynamic_branch_gate is not None
        assert direct_learner.gate_regularization_active
        assert not direct_learner.random_drop_auxiliary_active
        direct_mac.init_hidden(direct_batch.batch_size)
        direct_mac.set_dynamic_branch_gate_t_env(300000)
        direct_mac.forward(direct_batch, t=0)
        final_gate_layer = direct.dynamic_branch_gate.gate_network[-1]
        branch_gradient = th.autograd.grad(
            direct_mac.latest_aux_loss,
            final_gate_layer.bias,
            retain_graph=True,
        )[0]
        group_count = direct.dynamic_branch_gate.group_count
        assert branch_gradient[:group_count].abs().sum() > 0.0
        assert branch_gradient[group_count:].abs().sum() == 0.0
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
    assert not ALL_PROFILES["linear_bayesg_nomasktd_control"].get("kl", False)
    assert not control["clean_advantage_margin_auxiliary"]
    for label, (objective, behavior) in QME.items():
        flags = ALL_PROFILES[label]
        overrides = experiment_overrides(label)
        assert flags["branch"] == "linear"
        assert not flags.get("kl", False) and not flags.get("aux")
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
        assert not learner.gate_regularization_active
        learner.train(batch, t_env=500000, episode_num=1)
        assert logger.stats["loss_advantage_margin"][-1][1] == (
            logger.stats["loss_advantage_margin"][-1][1]
        )

    for framework, (expect_direct_kl, expect_aux_kl) in QME_FRAMEWORKS.items():
        control_label = "linear_{}_nomasktd_control".format(framework)
        control_flags = ALL_PROFILES[control_label]
        control_overrides = experiment_overrides(control_label)
        assert bool(control_flags.get("kl")) == expect_direct_kl
        assert bool(control_flags.get("aux")) == expect_aux_kl
        assert control_overrides["clean_main_td_coef"] == 1.0
        assert control_overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert not control_overrides["clean_advantage_margin_auxiliary"]

        for base_label, (objective, behavior) in MATRIX_QME.items():
            suffix = base_label[len("linear_qme_"):]
            label = "linear_{}_qme_{}".format(framework, suffix)
            flags = ALL_PROFILES[label]
            overrides = experiment_overrides(label)
            assert bool(flags.get("kl")) == expect_direct_kl
            assert bool(flags.get("aux")) == expect_aux_kl
            assert overrides["clean_advantage_objective"] == objective
            assert overrides["clean_train_behavior_gate_mode"] == behavior
            mac, learner, batch, logger = make_case(label)
            assert learner.gate_regularization_active == expect_direct_kl
            assert learner.random_drop_auxiliary_active == expect_aux_kl
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
        "Linear single-head suite passed: four scenes, KL80 forms, five-QME "
        "matrix and historical sampling/readiness profiles"
    )


if __name__ == "__main__":
    main()
