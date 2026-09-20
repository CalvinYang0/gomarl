#!/usr/bin/env python3
"""Verify full and exactly alternating mixed rollout behaviour selection."""
from pathlib import Path
from types import SimpleNamespace
import logging
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run.run import _run_force_open_test, _training_behavior_force_open


def sequence(mode, count=6):
    args = SimpleNamespace(clean_train_behavior_gate_mode=mode)
    return [
        _training_behavior_force_open(args, index)
        for index in range(count)
    ]


def main():
    assert sequence("masked") == [False] * 6
    assert sequence("full") == [True] * 6
    assert sequence("mixed") == [True, False, True, False, True, False]
    try:
        sequence("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown behaviour mode was accepted")

    class FakeMac:
        def __init__(self):
            self.force_open = False

        def set_dynamic_branch_gate_force_open(self, enabled):
            self.force_open = bool(enabled)

    class FakeRunner:
        def __init__(self):
            self.mac = FakeMac()
            self.test_log_prefix = "test_"
            self.calls = 0
            self.logger = SimpleNamespace(
                console_logger=logging.getLogger("open-win-smoke"),
                stats={},
            )

        def set_test_log_prefix(self, prefix):
            self.test_log_prefix = prefix

        def run(self, test_mode=False):
            assert test_mode
            assert self.test_log_prefix == "test_open_"
            assert self.mac.force_open
            self.calls += 1
            if self.calls == 2:
                self.logger.stats["test_open_game_win_mean"] = [
                    (123, 0.125)
                ]

    class FakeLearner:
        def __init__(self):
            self.open_win_rate = None

        def update_qme_open_win_rate(self, value):
            self.open_win_rate = float(value)

    runner = FakeRunner()
    learner = FakeLearner()
    rate = _run_force_open_test(
        SimpleNamespace(clean_dual_gate_test=True),
        runner,
        2,
        learner=learner,
    )
    assert rate == 0.125
    assert learner.open_win_rate == 0.125
    assert runner.test_log_prefix == "test_"
    assert runner.mac.force_open is False
    print(
        "HyperSelect behaviour sampling and open-win readiness passed: "
        "masked, full, mixed 1:1"
    )


if __name__ == "__main__":
    main()
