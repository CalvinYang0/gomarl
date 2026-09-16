#!/usr/bin/env python3
"""Exercise the three matched Advantage-margin TD-path profiles."""
import logging
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case


PROFILES = {
    "relation_advantage_augtd_teacheronly": (0.0, 0.0, True),
    "relation_advantage_masktd_augtd_teacheronly": (1.0, 0.0, True),
    "relation_advantage_augtd_nomasktd": (0.0, 1.0, False),
}


def main():
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(47)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label, (main_coef, nomask_coef, teacher_only) in PROFILES.items():
        overrides = experiment_overrides(label)
        missing = set(overrides) - set(sacred)
        assert not missing, "{} has unregistered keys: {}".format(
            label, sorted(missing)
        )
        assert math.isclose(overrides["clean_main_td_coef"], main_coef)
        assert math.isclose(
            overrides["clean_nomask_td_auxiliary_coef"], nomask_coef
        )
        assert overrides["clean_advantage_margin_teacher_only"] is teacher_only
        assert overrides["clean_advantage_margin_auxiliary"] is True
        assert overrides["clean_mask_parameter_relation_coef"] == 0.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        assert overrides["clean_kl_auxiliary_force_main_open"] is False
        assert overrides["clean_importance_auxiliary_warmup_steps"] == 0
        assert overrides["clean_dual_gate_test"] is True

        check(label)
        _, learner, batch, logger = make_case(label)
        assert learner.advantage_margin_teacher_only is teacher_only
        learner.train(batch, t_env=500000, episode_num=5)
        assert logger.stats["loss_random_drop_td_auxiliary"][-1][1] >= 0.0
        assert logger.stats["loss_advantage_margin"][-1][1] >= 0.0
        has_nomask_log = "loss_nomask_td_auxiliary" in logger.stats
        assert has_nomask_log is (nomask_coef > 0.0)
        gain = logger.stats[
            "train_gate/advantage_margin/margin_gain_mean"
        ][-1][1]
        assert math.isfinite(gain)
    print("Advantage augmented/mask/no-mask TD-path profiles: OK")


if __name__ == "__main__":
    main()
