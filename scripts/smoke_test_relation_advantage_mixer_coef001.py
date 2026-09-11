#!/usr/bin/env python3
"""Matched relation/Advantage plus low-weight mixer KL80 checks."""
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABELS = (
    "relation_kl80aux_mixer_coef001",
    "relation_advantage_kl80aux_mixer_coef001",
    "relation_advantage_kl80aux",
)


def check_matched_pair(reference_label, combined_label, seed):
    reference = experiment_overrides(reference_label)
    combined = experiment_overrides(combined_label)
    assert {key for key in reference if reference[key] != combined[key]} == {
        "clean_model_type",
        "clean_mixer_kl80_auxiliary_coef",
    }
    assert math.isclose(combined["clean_mixer_kl80_auxiliary_coef"], 0.01)

    th.manual_seed(seed)
    reference_mac, reference_learner, _, _ = make_case(reference_label)
    th.manual_seed(seed)
    combined_mac, combined_learner, batch, logger = make_case(combined_label)
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
    assert math.isclose(combined_learner.mixer_kl80_auxiliary_coef, 0.01)
    combined_learner.train(batch, t_env=300000, episode_num=1)
    raw = logger.stats["loss_mixer_kl80_td_auxiliary"][-1][1]
    weighted = logger.stats["weighted_loss_mixer_kl80_td_auxiliary"][-1][1]
    assert raw > 0
    assert math.isclose(weighted, 0.01 * raw, rel_tol=1e-6, abs_tol=1e-6)


def main():
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label in LABELS:
        missing = set(experiment_overrides(label)) - set(sacred)
        assert not missing, "{} has unregistered keys: {}".format(
            label, sorted(missing)
        )

    relation = experiment_overrides("relation_kl80aux_mixer_coef001")
    advantage = experiment_overrides(
        "relation_advantage_kl80aux_mixer_coef001"
    )
    standalone = experiment_overrides("relation_advantage_kl80aux")
    assert relation["clean_mask_parameter_relation_objective"] == "l1"
    assert advantage["clean_mask_parameter_relation_objective"] == (
        "advantage_contrastive"
    )
    assert standalone["clean_mask_parameter_relation_objective"] == (
        "advantage_contrastive"
    )
    assert math.isclose(
        advantage["clean_mask_parameter_relation_coef"], 0.1
    )

    check_matched_pair(
        "relation_kl80aux", "relation_kl80aux_mixer_coef001", 61
    )
    check_matched_pair(
        "relation_advantage_kl80aux",
        "relation_advantage_kl80aux_mixer_coef001",
        67,
    )
    for label in LABELS:
        check(label)
    print("Relation/Advantage + mixer-KL80(0.01): 3/3 matched and OK")


if __name__ == "__main__":
    main()
