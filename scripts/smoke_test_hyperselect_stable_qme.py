#!/usr/bin/env python3
"""Exercise stable joint-Q and TD-quality HyperSelect objectives."""
import logging
from pathlib import Path
import sys

import torch as th
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.agents.counter_transformer_suite import experiment_overrides
from smoke_test_counter_transformer_nine import check, make_case
from smoke_test_trans9_multiscene import check_smac_semantics


LABELS = (
    "hyperselect_qme_joint_rank_stable",
    "hyperselect_qme_tdquality_rank_stable",
)


def main():
    logging.disable(logging.CRITICAL)
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label in LABELS:
        for domain in ("grf", "smac"):
            overrides = experiment_overrides(label, domain)
            missing = set(overrides) - set(sacred)
            assert not missing, (label, domain, sorted(missing))
        overrides = experiment_overrides(label)
        assert overrides["clean_main_td_coef"] == 1.0
        assert overrides["clean_nomask_td_auxiliary_coef"] == 1.0
        assert overrides["clean_random_drop_auxiliary_coef"] == 1.0
        assert overrides["clean_advantage_stable_target_teacher"] is True
        assert overrides["clean_nomask_independent_target"] is True
        assert overrides["clean_advantage_dynamic_readiness"] is True

        check(label)
        _, learner, batch, logger = make_case(label)
        learner.train(batch, t_env=500000, episode_num=3)
        prefix = "train_gate/advantage_margin/"
        assert logger.stats[prefix + "readiness_weight"][-1][1] > 0.0
        assert logger.stats[prefix + "positive_return_ema"][-1][1] > 0.0
        assert logger.stats[prefix + "action_agreement"]
        if "joint_q" in label:
            assert logger.stats[prefix + "joint_masked_q"]
            assert logger.stats[prefix + "joint_full_q"]
            assert logger.stats[prefix + "joint_q_gain_mean"]
        else:
            assert logger.stats[prefix + "td_quality_gain_mean"]

        for scene in ("3s5z_vs_3s6z", "5m_vs_6m"):
            check_smac_semantics(scene, label)

    print(
        "Stable target teacher, independent full targets, dynamic QME "
        "readiness, action ranking, joint-Q and TD-quality objectives passed"
    )


if __name__ == "__main__":
    main()
