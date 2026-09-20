#!/usr/bin/env python3
"""Validate the corrected additive HyperSelect Counter ablations."""
import logging
from pathlib import Path
import sys

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import (
    ALL_PROFILES,
    experiment_overrides,
)
from smoke_test_counter_transformer_nine import check, make_case


EXPECTED = {
    # main MaskTD, clean LTD, SME AugTD, QME
    "hyperselect_gate": (1.0, 0.0, 0.0, False),
    "hyperselect_gate_ltd_control": (1.0, 1.0, 0.0, False),
    "hyperselect_gate_qme_additive": (1.0, 1.0, 0.0, True),
    "hyperselect_gate_sme_additive": (1.0, 0.0, 1.0, False),
    "hyperselect_gate_qme_sme_additive": (1.0, 1.0, 1.0, True),
}


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    for label, expected in EXPECTED.items():
        overrides = experiment_overrides(label)
        actual = (
            overrides["clean_main_td_coef"],
            overrides["clean_nomask_td_auxiliary_coef"],
            overrides["clean_random_drop_auxiliary_coef"],
            overrides["clean_advantage_margin_auxiliary"],
        )
        assert actual == expected, (label, actual, expected)
        assert overrides["clean_dynamic_branch_gate_warmup_steps"] == 250000
        assert overrides["clean_mask_parameter_relation_coef"] == 0.0
        if expected[2] > 0.0:
            assert ALL_PROFILES[label]["aux"] == "kl80"
            assert overrides["clean_random_drop_auxiliary_combine_mode"] == "multiply"

        check(label)
        _, learner, batch, logger = make_case(label)
        capturer = learner.mac.agent.rpg_relation_capturer
        if expected[2] > 0.0:
            auxiliary_gate = capturer.kl80_auxiliary_gate
            assert auxiliary_gate is not None
            assert auxiliary_gate.observation_independent is False
        learner.train(batch, t_env=500000, episode_num=1)
        assert learner.main_td_coef == expected[0]
        assert learner.nomask_td_auxiliary_coef == expected[1]
        assert learner.random_drop_auxiliary_coef == expected[2]
        assert learner.advantage_margin_auxiliary_active is expected[3]
        qme_key = "train_gate/advantage_margin/action_q_gain_mean"
        assert bool(logger.stats.get(qme_key)) is expected[3]

    print(
        "Corrected additive HyperSelect ablations preserve MaskTD and use "
        "observation-conditioned multiplicative SME masks"
    )


if __name__ == "__main__":
    main()
