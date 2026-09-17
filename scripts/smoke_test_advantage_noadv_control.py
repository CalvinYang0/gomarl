#!/usr/bin/env python3
"""Exercise the no-Advantage AugTD + NoMaskTD control."""
import logging
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


LABEL = "relation_noadv_augtd_nomasktd"


def main():
    logging.disable(logging.CRITICAL)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    overrides = experiment_overrides(LABEL)
    missing = set(overrides) - set(sacred)
    assert not missing, "unregistered keys: {}".format(sorted(missing))
    assert overrides["clean_main_td_coef"] == 0.0
    assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
    assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
    assert overrides["clean_random_drop_auxiliary_identity_warmup"] is True
    assert overrides["clean_kl_auxiliary_force_main_open"] is False
    assert overrides["clean_advantage_margin_auxiliary"] is False
    assert overrides["clean_mask_parameter_relation_coef"] == 0.0
    assert overrides["clean_importance_auxiliary_warmup_steps"] == 250000
    assert overrides["clean_dual_gate_test"] is True

    check(LABEL)
    _, learner, batch, logger = make_case(LABEL)
    assert learner.advantage_margin_auxiliary_active is False
    learner.train(batch, t_env=10, episode_num=4)
    assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] > 0.0
    learner.train(batch, t_env=500000, episode_num=5)
    for key in (
        "loss_td",
        "loss_nomask_td_auxiliary",
        "loss_random_drop_td_auxiliary",
    ):
        assert math.isfinite(logger.stats[key][-1][1]), key
    assert "loss_advantage_margin" not in logger.stats
    print("No-Advantage AugTD + NoMaskTD control: OK")


if __name__ == "__main__":
    main()
