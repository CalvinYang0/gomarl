#!/usr/bin/env python3
"""Regression checks for observation-independent learned slot masks."""
import logging
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import ObservationConditionedBranchGate
from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABELS = (
    "static_gate_kl80aux_thr05",
    "static_gate_kl80aux_thr08",
    "static_gate_kl80aux_sharp_thr05",
)


def check_gate_primitive():
    ordinary = ObservationConditionedBranchGate(
        obs_dim=5, hidden_dim=8, mode="binary_concrete",
        initial_keep_probability=0.95, observation_independent=True,
        probability_temperature=1.0, hard_threshold=0.5,
    )
    sharp = ObservationConditionedBranchGate(
        obs_dim=5, hidden_dim=8, mode="binary_concrete",
        initial_keep_probability=0.95, observation_independent=True,
        probability_temperature=0.5, hard_threshold=0.5,
    )
    assert ordinary.gate_network is None and sharp.gate_network is None
    obs_a = th.randn(6, 5)
    obs_b = th.randn(6, 5) * 100.0
    _, probability_a = ordinary(obs_a, sample=False, deterministic_soft=True)
    _, probability_b = ordinary(obs_b, sample=False, deterministic_soft=True)
    assert th.equal(probability_a, probability_b)
    assert th.allclose(probability_a, th.full_like(probability_a, 0.95))

    with th.no_grad():
        ordinary.static_logits.fill_(0.4)
        sharp.static_logits.fill_(0.4)
    _, ordinary_probability = ordinary(
        obs_a, sample=False, deterministic_soft=True
    )
    _, sharp_probability = sharp(obs_a, sample=False, deterministic_soft=True)
    assert sharp_probability.mean() > ordinary_probability.mean()
    sharp_probability.mean().backward()
    assert sharp.static_logits.grad is not None
    assert sharp.static_logits.grad.abs().sum() > 0

    threshold05 = ObservationConditionedBranchGate(
        obs_dim=3, hidden_dim=4, mode="binary_concrete",
        observation_independent=True, hard_threshold=0.5,
    )
    threshold08 = ObservationConditionedBranchGate(
        obs_dim=3, hidden_dim=4, mode="binary_concrete",
        observation_independent=True, hard_threshold=0.8,
    )
    probability = 0.7
    logit = math.log(probability / (1.0 - probability))
    with th.no_grad():
        threshold05.static_logits.fill_(logit)
        threshold08.static_logits.fill_(logit)
    gate05, _ = threshold05(obs_a[:, :3], sample=False)
    gate08, _ = threshold08(obs_a[:, :3], sample=False)
    assert gate05.eq(1).all() and gate08.eq(0).all()


def check_profiles():
    sacred = yaml.safe_load((ROOT / "src/config/algs/clean_hyper.yaml").read_text())
    for label in LABELS:
        overrides = experiment_overrides(label)
        assert not set(overrides) - set(sacred)
        assert overrides["clean_dynamic_branch_gate_static"]
        assert overrides["clean_mask_parameter_relation_coef"] == 0.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        mac, _, batch, _ = make_case(label)
        gate = mac.agent.rpg_relation_capturer.dynamic_branch_gate
        assert gate.observation_independent
        assert gate.static_logits.numel() == mac.args.obs_shape
        assert gate.gate_network is None
        mac.set_dynamic_branch_gate_t_env(300000)
        mac.init_hidden(batch.batch_size)
        mac.forward(batch, t=0)
        first = mac.latest_dynamic_branch_probabilities_graph
        mac.forward(batch, t=1)
        second = mac.latest_dynamic_branch_probabilities_graph
        # The global learned vector is exactly invariant to observation,
        # agent, batch item and timestep.
        assert th.equal(first[:, :1].expand_as(first), first)
        assert th.equal(first, second)
        check(label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(31)
    check_gate_primitive()
    check_profiles()
    print("Static sigmoid threshold/sharpness controls passed on Counter")
