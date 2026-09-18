#!/usr/bin/env python3
"""Validate the five paper-facing HyperSelect Counter ablations."""
import logging
from pathlib import Path
import sys

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABELS = (
    "baseline",
    "hyperselect_gate",
    "hyperselect_gate_qme",
    "hyperselect_gate_sme",
    "relation_advantage_qvalue_augtd_nomasktd",
)

EXPECTED = {
    "baseline": (1.0, 0.0, 0.0, False),
    "hyperselect_gate": (1.0, 0.0, 0.0, False),
    "hyperselect_gate_qme": (0.0, 1.0, 0.0, True),
    "hyperselect_gate_sme": (0.0, 1.0, 1.0, False),
    "relation_advantage_qvalue_augtd_nomasktd": (0.0, 1.0, 1.0, True),
}


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    for label in LABELS:
        overrides = experiment_overrides(label)
        expected = EXPECTED[label]
        actual = (
            overrides["clean_main_td_coef"],
            overrides["clean_nomask_td_auxiliary_coef"],
            overrides["clean_random_drop_auxiliary_coef"],
            overrides["clean_advantage_margin_auxiliary"],
        )
        assert actual == expected, (label, actual, expected)
        assert overrides["clean_mask_parameter_relation_coef"] == 0.0
        assert overrides["clean_dynamic_branch_gate_warmup_steps"] == 250000
        check(label)

        _, learner, batch, logger = make_case(label)
        learner.train(batch, t_env=500000, episode_num=1)
        assert learner.main_td_coef == expected[0]
        assert learner.nomask_td_auxiliary_coef == expected[1]
        assert learner.random_drop_auxiliary_coef == expected[2]
        assert learner.advantage_margin_auxiliary_active is expected[3]
        if expected[3]:
            key = "train_gate/advantage_margin/action_q_gain_mean"
            assert logger.stats[key], (label, key)
        else:
            assert not logger.stats.get(
                "train_gate/advantage_margin/action_q_gain_mean"
            )
    print("Five HyperSelect Counter paper ablations passed")


if __name__ == "__main__":
    main()
