#!/usr/bin/env python3
"""Regression checks for the 0.1 and 0.01 mixer-KL80 TD weights."""
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from ozstar_submit_counter_transformer_nine import build_plans
from smoke_test_counter_transformer_nine import check, make_case


def main():
    th.set_num_threads(1)
    repo = ROOT
    labels_and_coefficients = (
        ("mixer_kl80aux_coef01", 0.1),
        ("mixer_kl80aux_coef001", 0.01),
    )
    plans = build_plans(repo, [label for label, _ in labels_and_coefficients])
    assert len(plans) == 2
    sacred = yaml.safe_load(
        (repo / "src/config/algs/clean_hyper.yaml").read_text()
    )

    for (label, expected_coefficient), plan in zip(
        labels_and_coefficients, plans
    ):
        overrides = experiment_overrides(label)
        assert not set(overrides) - set(sacred)
        assert math.isclose(
            overrides["clean_mixer_kl80_auxiliary_coef"],
            expected_coefficient,
        )
        assert plan["exports"]["MODEL_TYPE"].endswith(
            "suite_{}_hypercond".format(label)
        )

        _, learner, batch, logger = make_case(label)
        assert learner.mixer_kl80_auxiliary_active
        assert math.isclose(
            learner.mixer_kl80_auxiliary_coef, expected_coefficient
        )
        learner.train(batch, t_env=300000, episode_num=1)
        raw = logger.stats["loss_mixer_kl80_td_auxiliary"][-1][1]
        weighted = logger.stats[
            "weighted_loss_mixer_kl80_td_auxiliary"
        ][-1][1]
        assert raw > 0
        assert math.isclose(
            weighted,
            expected_coefficient * raw,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        check(label)

    print("Mixer KL80 TD coefficients 0.1 and 0.01: updates and logs OK")


if __name__ == "__main__":
    main()
