#!/usr/bin/env python3
"""Regression checks for trajectory-grouped, per-agent fixed masks."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learners.clean_learner import CleanLearner
from modules.agents.clean_hyper_agent import ObservationConditionedBranchGate
from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "relation_trajectory_agent_kl80aux"


def check_per_agent_gate():
    gate = ObservationConditionedBranchGate(
        obs_dim=5,
        hidden_dim=8,
        mode="binary_concrete",
        initial_keep_probability=0.8,
        gate_scope="shared",
        observation_independent=True,
        observation_independent_agent_count=3,
    )
    assert gate.static_logits.shape == (3, 5)
    with th.no_grad():
        gate.static_logits[0].fill_(-1.0)
        gate.static_logits[1].fill_(0.0)
        gate.static_logits[2].fill_(1.0)
    obs_a = th.randn(2, 3, 5)
    obs_b = th.randn(2, 3, 5) * 100.0
    _, probability_a = gate(obs_a, sample=False, deterministic_soft=True)
    _, probability_b = gate(obs_b, sample=False, deterministic_soft=True)
    assert th.equal(probability_a, probability_b)
    assert th.equal(probability_a[:, :1].expand_as(probability_a), probability_a)
    assert probability_a[0, 0, 0, 0] < probability_a[0, 0, 1, 0]
    assert probability_a[0, 0, 1, 0] < probability_a[0, 0, 2, 0]


def check_trajectory_objective():
    probe = type("TrajectoryRelationProbe", (), {})()
    probe.counter_branch_index = 1
    probe.advantage_relation_temperature = 1.0
    probe.advantage_relation_positive_threshold = 0.1
    probe.advantage_relation_negative_threshold = 0.4
    probe.advantage_relation_confidence_threshold = 0.05
    probe.advantage_relation_mask_margin = 0.2
    probe.advantage_relation_negative_coef = 1.0
    probe._trajectory_advantage_mask_relation = (
        CleanLearner._trajectory_advantage_mask_relation.__get__(probe)
    )
    # Agents 0/1 prefer action 1 throughout the trajectory; agent 2 prefers
    # action 0. Additive Q offsets across time must not change grouping.
    base = th.tensor([
        [0.0, 4.0, 0.0],
        [1.0, 5.0, 1.0],
        [4.0, 0.0, 0.0],
    ])
    q_values = th.stack([base, base + 10.0, base + 20.0], dim=0).unsqueeze(0)
    available = th.ones_like(q_values, dtype=th.bool)
    logits = th.tensor(
        [[0.40, 0.40], [0.20, 0.20], [0.30, 0.30]],
        requires_grad=True,
    )
    attention_probability = logits.sigmoid().unsqueeze(0)
    probabilities = [
        th.stack([th.ones_like(attention_probability), attention_probability])
        for _ in range(3)
    ]
    valid = [th.tensor([True])] * 3
    loss, stats = probe._trajectory_advantage_mask_relation(
        q_values, available, probabilities, valid
    )
    assert stats["positive_count"].item() > 0
    assert stats["negative_count"].item() > 0
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def check_profile(scene):
    mac, learner, batch, logger = make_case(LABEL, scene)
    gate = mac.agent.rpg_relation_capturer.dynamic_branch_gate
    assert gate.observation_independent
    assert gate.observation_independent_agent_count == mac.args.n_agents
    assert gate.static_logits.shape == (mac.args.n_agents, mac.args.obs_shape)
    learner.train(batch, t_env=300000, episode_num=1)
    assert "advantage_relation_candidate_count" in logger.stats
    check(LABEL, scene)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(41)
    sacred = yaml.safe_load((ROOT / "src/config/algs/clean_hyper.yaml").read_text())
    overrides = experiment_overrides(LABEL)
    assert not set(overrides) - set(sacred)
    assert overrides["clean_dynamic_branch_gate_static"]
    assert overrides["clean_dynamic_branch_gate_per_agent"]
    assert overrides["clean_mask_parameter_relation_objective"] == (
        "trajectory_advantage_contrastive"
    )
    check_per_agent_gate()
    check_trajectory_objective()
    check_profile("academy_counterattack_easy")
    check_profile("MMM2")
    check_profile("3s5z_vs_3s6z")
    print("Trajectory-grouped per-agent masks passed on Counter, MMM2, and 3s5z")
