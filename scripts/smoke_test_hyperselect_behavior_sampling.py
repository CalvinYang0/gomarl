#!/usr/bin/env python3
"""Verify full and exactly alternating mixed rollout behaviour selection."""
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run.run import _training_behavior_force_open


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
    print("HyperSelect behaviour sampling passed: masked, full, mixed 1:1")


if __name__ == "__main__":
    main()
