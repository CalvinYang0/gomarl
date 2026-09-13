#!/usr/bin/env python3
"""Exercise the eight Counter relation loss-composition controls."""
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
    "relation_lnomask_lmask",
    "kl80_ltd_lkl_only",
    "relation_all4",
    "relation_lnomask_lmask_testopen",
    "relation_all4_testopen",
    "relation_all4_sigmoid",
    "relation_all4_relcoef10",
    "relation_all4_relcoef01",
)


def main():
    th.set_num_threads(1)
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    for label in LABELS:
        overrides = experiment_overrides(label)
        missing = set(overrides) - set(sacred)
        assert not missing, "{} has unregistered keys: {}".format(
            label, sorted(missing)
        )

    expected = {
        "relation_lnomask_lmask": (1.0, 1.0, False, "l1", 1.0),
        "kl80_ltd_lkl_only": (0.0, 0.0, False, "l1", 0.0),
        "relation_all4": (1.0, 1.0, True, "l1", 1.0),
        "relation_lnomask_lmask_testopen": (1.0, 1.0, False, "l1", 1.0),
        "relation_all4_testopen": (1.0, 1.0, True, "l1", 1.0),
        "relation_all4_sigmoid": (1.0, 1.0, True, "l1_sigmoid", 1.0),
        "relation_all4_relcoef10": (1.0, 1.0, True, "l1", 10.0),
        "relation_all4_relcoef01": (1.0, 1.0, True, "l1", 0.1),
    }
    for label in LABELS:
        overrides = experiment_overrides(label)
        main, nomask, force_open, objective, relation_coef = expected[label]
        assert math.isclose(overrides["clean_main_td_coef"], main)
        assert math.isclose(
            overrides["clean_nomask_td_auxiliary_coef"], nomask
        )
        assert overrides["clean_kl_auxiliary_force_main_open"] is force_open
        assert overrides["clean_mask_parameter_relation_objective"] == objective
        assert math.isclose(
            overrides["clean_mask_parameter_relation_coef"], relation_coef
        )
        check(label)

        _, learner, batch, logger = make_case(label)
        learner.train(batch, t_env=300000, episode_num=1)
        assert math.isclose(learner.main_td_coef, main)
        assert math.isclose(learner.nomask_td_auxiliary_coef, nomask)
        assert learner.kl_auxiliary_force_main_open is force_open
        assert math.isclose(
            logger.stats["weighted_loss_main_td"][-1][1],
            main * logger.stats["loss_td"][-1][1],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        if nomask:
            raw = logger.stats["loss_nomask_td_auxiliary"][-1][1]
            weighted = logger.stats[
                "weighted_loss_nomask_td_auxiliary"
            ][-1][1]
            assert raw > 0.0
            assert math.isclose(weighted, nomask * raw, rel_tol=1e-6)
        else:
            assert not logger.stats.get("loss_nomask_td_auxiliary")

    print("Counter relation loss composition: 8/8 profiles and updates OK")


if __name__ == "__main__":
    main()
