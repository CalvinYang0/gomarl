#!/usr/bin/env python3
"""Verify full-observation execution with training-only KL80 augmentation."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "kl80aux_augmentation"


def main():
    sacred = yaml.safe_load((ROOT / "src/config/algs/clean_hyper.yaml").read_text())
    overrides = experiment_overrides(LABEL)
    assert not set(overrides) - set(sacred)
    assert overrides["clean_mask_parameter_relation_coef"] == 0.0
    assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
    assert not overrides["clean_dynamic_branch_gate_static"]

    mac, learner, batch, logger = make_case(LABEL)
    capturer = mac.agent.rpg_relation_capturer
    assert capturer.dynamic_branch_gate is None
    assert capturer.kl80_auxiliary_gate is not None
    assert learner.random_drop_auxiliary_active
    learner.train(batch, t_env=300000, episode_num=1)
    assert logger.stats["loss_kl80_random_auxiliary"][-1][1] > 0
    # The auxiliary gate must be disabled again after its second rollout, so
    # ordinary execution cannot accidentally retain augmentation masking.
    assert not capturer.kl80_auxiliary_enabled
    check(LABEL)
    print("Full-observation baseline + training-only KL80 augmentation passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(37)
    main()
