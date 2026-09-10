#!/usr/bin/env python3
"""Regression checks for detached-Advantage dynamic mask grouping."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learners.clean_learner import CleanLearner
from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "relation_advantage_kl80aux"


def check_pair_objective():
    probe = type("AdvantageProbe", (), {})()
    probe.counter_branch_index = 1
    probe.advantage_relation_temperature = 1.0
    probe.advantage_relation_positive_threshold = 0.1
    probe.advantage_relation_negative_threshold = 0.4
    probe.advantage_relation_confidence_threshold = 0.05
    probe.advantage_relation_mask_margin = 0.2
    probe.advantage_relation_negative_coef = 1.0
    probe._random_relation_pair_indices = CleanLearner._random_relation_pair_indices
    probe._advantage_contrastive_mask_relation = (
        CleanLearner._advantage_contrastive_mask_relation.__get__(probe)
    )

    # Each agent keeps the same advantage structure over time (positive
    # temporal edges), while the two agents prefer different actions
    # (negative simultaneous edges). Absolute Q offsets deliberately differ.
    q_values = th.tensor([[[[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]],
                           [[10.0, 14.0, 10.0], [14.0, 10.0, 10.0]]]])
    available = th.ones_like(q_values, dtype=th.bool)
    logits = [
        th.tensor(
            [[[2.0, 2.0], [2.0, 2.0]],
             [[0.40, 0.40], [0.20, 0.20]]],
            requires_grad=True,
        )
        for _ in range(2)
    ]
    probabilities = [value.sigmoid() for value in logits]
    valid = [th.tensor([True]), th.tensor([True])]
    loss, stats = probe._advantage_contrastive_mask_relation(
        q_values, available, probabilities, valid
    )
    assert stats["positive_count"].item() > 0
    assert stats["negative_count"].item() > 0
    assert 0.0 < loss.item() < probe.advantage_relation_mask_margin
    loss.backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in logits)


def check_configuration():
    config = yaml.safe_load((ROOT / "src/config/algs/clean_hyper.yaml").read_text())
    overrides = experiment_overrides(LABEL)
    assert not set(overrides) - set(config)
    assert overrides["clean_mask_parameter_relation_objective"] == (
        "advantage_contrastive"
    )
    assert overrides["clean_mask_parameter_relation_coef"] == 0.1
    _, learner, batch, logger = make_case(LABEL)
    learner.train(batch, t_env=300000, episode_num=1)
    assert learner.mask_parameter_relation_objective == "advantage_contrastive"
    assert "advantage_relation_candidate_count" in logger.stats
    assert "advantage_relation_positive_count" in logger.stats
    assert "advantage_relation_negative_count" in logger.stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(23)
    check_pair_objective()
    check_configuration()
    check(LABEL)
    check(LABEL, "MMM2")
    check(LABEL, "3s5z_vs_3s6z")
    print("Advantage-induced mask grouping passed on Counter, MMM2, and 3s5z")
