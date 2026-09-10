#!/usr/bin/env python3
"""Matched obs-gate KL80 plus mixer-KL80(0.01) regression check."""
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


def main():
    th.set_num_threads(1)
    label = "obs_gate_kl80aux_mixer_coef001"
    reference = experiment_overrides("obs_gate_kl80aux")
    combined = experiment_overrides(label)
    assert {key for key in reference if reference[key] != combined[key]} == {
        "clean_model_type",
        "clean_mixer_kl80_auxiliary_coef",
    }
    assert math.isclose(combined["clean_mixer_kl80_auxiliary_coef"], 0.01)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    assert not set(combined) - set(sacred)

    th.manual_seed(29)
    reference_mac, reference_learner, _, _ = make_case("obs_gate_kl80aux")
    th.manual_seed(29)
    combined_mac, combined_learner, batch, logger = make_case(label)
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
    check(label)
    print("Observation KL80 gate + mixer KL80 coefficient 0.01: matched and OK")


if __name__ == "__main__":
    main()
