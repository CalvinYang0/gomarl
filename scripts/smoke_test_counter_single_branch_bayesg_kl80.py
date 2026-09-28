#!/usr/bin/env python3
"""Regression checks for the matched single-branch KL80 gate controls."""

import sys
from pathlib import Path

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


CASES = {
    "linear_bayesg_kl80_keep": "linear_only",
    "transformer_bayesg_kl80_keep": "attention_only",
}


def check(label, encoder_style):
    flags = ALL_PROFILES[label]
    overrides = experiment_overrides(label)
    assert flags["gate"] and flags["kl"]
    assert not flags.get("aux")
    assert overrides["clean_model_type"] == model_type_for(label)
    assert overrides["clean_binary_concrete_temperature"] == 0.5
    assert overrides["clean_hard_gate_threshold"] == 0.5
    assert overrides["clean_hard_gate_initial_keep_probability"] == 0.95
    assert overrides["clean_dynamic_branch_gate_warmup_steps"] == 250000

    th.manual_seed(13)
    mac, learner, batch, logger = make_case(label)
    capturer = mac.agent.rpg_relation_capturer
    assert capturer.relation_encoder_style == encoder_style
    assert capturer.dynamic_branch_gate.mode == "binary_concrete"
    assert capturer.dynamic_branch_gate_regularizer == "bernoulli_kl"
    assert capturer.dynamic_branch_gate_prior_keep == 0.8
    assert learner.gate_regularization_active
    assert not learner.random_drop_auxiliary_active

    # Training uses differentiable stochastic Binary-Concrete values.
    mac.set_dynamic_branch_gate_t_env(300000)
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0, test_mode=False)
    train_gate = capturer.latest_dynamic_branch_gates_graph
    assert ((train_gate > 0.0) & (train_gate < 1.0)).all()

    # Evaluation converts the same observation-conditioned probabilities to
    # deterministic hard 0/1 decisions at threshold 0.5.
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0, test_mode=True)
    eval_gate = capturer.latest_dynamic_branch_gates_graph
    assert ((eval_gate == 0.0) | (eval_gate == 1.0)).all()

    learner.train(batch, t_env=300000, episode_num=1)
    assert logger.stats["loss_aux"][-1][1] > 0.0
    assert logger.stats["weighted_loss_dynamic_gate_regularizer"][-1][1] > 0.0
    assert not logger.stats.get("loss_random_drop_td_auxiliary")
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    print(label + ": soft-train/hard-test + single TD + gate KL80 OK")


def main():
    th.set_num_threads(1)
    for label, encoder_style in CASES.items():
        check(label, encoder_style)


if __name__ == "__main__":
    main()
