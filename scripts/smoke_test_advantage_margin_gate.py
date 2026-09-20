#!/usr/bin/env python3
"""Exercise Advantage-margin gate supervision and its gradient routing."""
import logging
import math
from pathlib import Path
import sys

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.counter_transformer_suite import experiment_overrides
from runners.episode_runner import EpisodeRunner
from runners.parallel_runner import ParallelRunner
from run.run import _run_force_open_test
from smoke_test_counter_transformer_nine import check, make_case
from utils.logging import Logger


LABELS = (
    "relation_advantage_margin_kl80aux",
    "relation_advantage_margin_kl80aux_gradsep",
    "relation_advantage_weighted_kl80aux",
    "relation_advantage_weighted_kl80aux_gradsep",
)


def check_gradient_router(learner, separated):
    if separated:
        non_gate = learner.gradient_separation_non_gate_parameters[0]
        gate = learner.gradient_separation_gate_parameters[0]
    else:
        non_gate = learner.importance_non_gate_parameters[0]
        gate = learner.importance_gate_parameters[0]
    learner.optimiser.zero_grad()
    teacher_or_main = non_gate.reshape(-1)[0] + 2.0 * gate.reshape(-1)[0]
    masked_or_aux = (
        11.0 * non_gate.reshape(-1)[0]
        + 3.0 * gate.reshape(-1)[0]
    )
    if separated:
        learner._backward_mask_nomask_gradient_separated(
            teacher_or_main, masked_or_aux
        )
        expected_gate = 3.0
    else:
        learner._backward_main_and_gate_only_auxiliary(
            teacher_or_main, masked_or_aux
        )
        expected_gate = 5.0
    assert th.allclose(
        non_gate.grad.reshape(-1)[0], non_gate.new_tensor(1.0)
    )
    assert th.allclose(
        gate.grad.reshape(-1)[0], gate.new_tensor(expected_gate)
    )
    learner.optimiser.zero_grad()


def check_teacher_advantage_weighting(learner):
    full_values = th.tensor([[[[4.0, 0.0, 0.0]]]])
    masked_values = th.tensor(
        [[[[1.0, 0.9, 0.0]]]], requires_grad=True
    )
    available = th.ones_like(full_values, dtype=th.int)
    valid = th.ones(1, 1, 1, dtype=th.bool)
    learner.advantage_margin_weight_by_teacher = False
    plain_loss, plain_stats = learner._advantage_margin_gate_loss(
        masked_values, full_values, available, valid
    )
    learner.advantage_margin_weight_by_teacher = True
    weighted_loss, weighted_stats = learner._advantage_margin_gate_loss(
        masked_values, full_values, available, valid
    )
    teacher_advantage = weighted_stats["teacher_margin"]
    assert teacher_advantage.item() > 1.0
    assert th.allclose(
        weighted_loss,
        plain_loss * teacher_advantage,
        rtol=1e-5,
        atol=1e-6,
    )
    assert weighted_stats["sample_weight"] > plain_stats["sample_weight"]


def check_dual_test_logging_contract():
    for runner_type in (EpisodeRunner, ParallelRunner):
        runner = object.__new__(runner_type)
        runner.test_log_prefix = "test_"
        runner.set_test_log_prefix("test_open_")
        assert runner.test_log_prefix == "test_open_"
        runner.set_test_log_prefix("test_")
        try:
            runner.set_test_log_prefix("open_")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid test prefix was accepted")
    assert Logger._wandb_metric_allowed("test_open_game_win_mean")
    assert Logger._wandb_metric_allowed("test_open_battle_won_mean")
    assert not Logger._wandb_metric_allowed("test_open_gate_heatmap")

    class FakeMac:
        def __init__(self):
            self.force_open = False

        def set_dynamic_branch_gate_force_open(self, enabled):
            self.force_open = bool(enabled)

    class FakeRunner:
        def __init__(self):
            self.mac = FakeMac()
            self.test_log_prefix = "test_"
            self.calls = []
            self.logger = type("Log", (), {
                "console_logger": logging.getLogger("dual-test-smoke"),
                "stats": {},
            })()

        def set_test_log_prefix(self, prefix):
            self.test_log_prefix = prefix

        def run(self, test_mode=False):
            self.calls.append(
                (test_mode, self.test_log_prefix, self.mac.force_open)
            )
            if len(self.calls) == 2:
                self.logger.stats["test_open_game_win_mean"] = [
                    (123, 0.125)
                ]

    class FakeLearner:
        def __init__(self):
            self.open_win_rate = None

        def update_qme_open_win_rate(self, value):
            self.open_win_rate = value

    fake = FakeRunner()
    learner = FakeLearner()
    args = type("Args", (), {"clean_dual_gate_test": True})()
    open_win_rate = _run_force_open_test(args, fake, 2, learner=learner)
    assert fake.calls == [
        (True, "test_open_", True),
        (True, "test_open_", True),
    ]
    assert fake.test_log_prefix == "test_"
    assert fake.mac.force_open is False
    assert open_win_rate == 0.125
    assert learner.open_win_rate == 0.125


def check_profile(label):
    sacred = yaml.safe_load(
        (ROOT / "src/config/algs/clean_hyper.yaml").read_text()
    )
    overrides = experiment_overrides(label)
    missing = set(overrides) - set(sacred)
    assert not missing, "{} has unregistered keys: {}".format(
        label, sorted(missing)
    )
    assert overrides["clean_advantage_margin_auxiliary"] is True
    assert overrides["clean_advantage_margin_weight_by_teacher"] is (
        "_weighted_" in label
    )
    assert overrides["clean_dual_gate_test"] is True
    assert math.isclose(overrides["clean_main_td_coef"], 1.0)
    assert math.isclose(overrides["clean_nomask_td_auxiliary_coef"], 1.0)
    assert math.isclose(overrides["clean_random_drop_auxiliary_coef"], 1.0)
    assert math.isclose(overrides["clean_mask_parameter_relation_coef"], 1.0)
    assert overrides["clean_kl_auxiliary_force_main_open"] is True
    assert overrides["clean_mask_nomask_gradient_separation"] is (
        label.endswith("_gradsep")
    )

    check(label)
    _, learner, batch, logger = make_case(label)
    assert learner.advantage_margin_auxiliary_active
    check_gradient_router(learner, label.endswith("_gradsep"))
    if "_weighted_" in label:
        check_teacher_advantage_weighting(learner)
        learner.advantage_margin_weight_by_teacher = True
    learner.train(batch, t_env=500000, episode_num=3)
    assert logger.stats["loss_advantage_margin"][-1][1] >= 0.0
    assert logger.stats["weighted_loss_advantage_margin"][-1][1] >= 0.0
    assert math.isclose(
        logger.stats[
            "train_gate/advantage_margin/effective_coef"
        ][-1][1],
        learner.advantage_margin_auxiliary_coef,
    )
    for name in (
        "teacher_margin",
        "masked_margin",
        "confidence",
        "sample_weight",
        "margin_gain_mean",
        "margin_shortfall_mean",
        "margin_improve_rate",
        "margin_harm_rate",
        "margin_target_met_rate",
        "confidence_weighted_margin_gain",
        "high_confidence_fraction",
        "high_confidence_margin_gain",
        "high_confidence_improve_rate",
        "high_confidence_target_met_rate",
        "action_agreement",
    ):
        value = logger.stats[
            "train_gate/advantage_margin/" + name
        ][-1][1]
        assert math.isfinite(value)
        if name.endswith("_rate") or name.endswith("_fraction"):
            assert 0.0 <= value <= 1.0


def main():
    logging.basicConfig(level=logging.WARNING)
    th.set_num_threads(1)
    th.manual_seed(31)
    check_dual_test_logging_contract()
    for label in LABELS:
        check_profile(label)
    print(
        "Counter Advantage-margin/weighted, joint/gradient-separated "
        "profiles and dual test logging: OK"
    )


if __name__ == "__main__":
    main()
