#!/usr/bin/env python3
"""Regression checks for the five KL/relation/mixer follow-up runs."""
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from ozstar_submit_counter_transformer_nine import build_plans
from smoke_test_counter_kl_prior_aux import check_prior
from smoke_test_counter_transformer_nine import check, make_case


def check_registered(repo, labels):
    sacred = yaml.safe_load((repo / "src/config/algs/clean_hyper.yaml").read_text())
    for label in labels:
        missing = set(experiment_overrides(label)) - set(sacred)
        assert not missing, "{} has unregistered Sacred keys: {}".format(
            label, sorted(missing)
        )


def check_combined_mixer():
    reference = experiment_overrides("relation_kl80aux")
    combined = experiment_overrides("relation_kl80aux_mixer")
    assert {key for key in reference if reference[key] != combined[key]} == {
        "clean_model_type", "clean_mixer_kl80_auxiliary_coef"
    }
    th.manual_seed(19)
    reference_mac, reference_learner, _, _ = make_case("relation_kl80aux")
    th.manual_seed(19)
    combined_mac, combined_learner, _, _ = make_case("relation_kl80aux_mixer")
    assert all(
        th.equal(value, combined_mac.agent.state_dict()[key])
        for key, value in reference_mac.agent.state_dict().items()
    )
    assert all(
        th.equal(value, combined_learner.mixer.state_dict()[key])
        for key, value in reference_learner.mixer.state_dict().items()
    )
    assert combined_learner.random_drop_auxiliary_active
    assert combined_learner.mixer_kl80_auxiliary_active
    assert combined_learner.mixer_kl80_gate is not None
    check("relation_kl80aux_mixer")


def check_adjacent_random_agreement():
    label = "relation_kl80aux_adjrand_agreement"
    overrides = experiment_overrides(label)
    assert overrides["clean_mask_parameter_relation_pairing"] == "adjacent_random"
    assert overrides["clean_mask_parameter_relation_objective"] == "agreement"

    _, learner, batch, logger = make_case(label)
    learner.train(batch, t_env=300000, episode_num=1)
    assert logger.stats["mask_parameter_relation_pair_count"][-1][1] > 4
    assert "mask_parameter_relation_mask_distance_mean" in logger.stats
    assert "mask_parameter_relation_parameter_target_mean" in logger.stats

    probe = type(learner).__new__(type(learner))
    probe.mask_parameter_relation_scale = 0.1
    probe.temporal_param_scale_eps = 1e-6
    probe.mask_parameter_relation_group_distance = False
    probe.mask_parameter_relation_group_ids = None
    probe.mask_parameter_relation_stop_side = "parameter"
    probe.mask_parameter_relation_objective = "agreement"
    probe.counter_transformer_profile = {"relation": True}
    probe.counter_branch_index = 1
    previous_parameter = th.zeros(1, 2, requires_grad=True)
    current_parameter = th.ones(1, 2, requires_grad=True)
    previous_logits = th.full((2, 1, 1, 2), math.log(0.2 / 0.8), requires_grad=True)
    current_logits = th.full((2, 1, 1, 2), math.log(0.4 / 0.6), requires_grad=True)
    total, count, a_sum, b_sum = probe._mask_parameter_relation_pair(
        (previous_parameter,), (current_parameter,),
        previous_logits.sigmoid(), current_logits.sigmoid(),
        th.tensor([True]), 1, 1,
    )
    a, b = a_sum / count, b_sum / count
    assert b.item() > 0.5
    assert th.allclose(total / count, a * (1.0 - b) + (1.0 - a) * b)
    (total / count).backward()
    assert current_logits.grad[1].max() < 0
    assert previous_parameter.grad is None and current_parameter.grad is None
    check(label)


if __name__ == "__main__":
    th.set_num_threads(1)
    repo = ROOT
    labels = (
        "relation_kl90aux",
        "relation_kl80aux_mixer",
        "relation_kl80aux_adjrand_agreement",
    )
    check_registered(repo, labels)
    assert len(build_plans(repo, labels)) == 3
    check_prior(0.9)
    check_combined_mixer()
    check_adjacent_random_agreement()
    print("KL90, combined mixer, and adjacent-random agreement controls passed")
