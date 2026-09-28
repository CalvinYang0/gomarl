#!/usr/bin/env python3
"""Offline unit smoke test for multi-seed curve aggregation."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_wandb_seed_aggregate import (
    align_curves,
    collapse_duplicate_steps,
    common_grid,
    ema,
    parse_series,
    summarize,
)


def main():
    assert parse_series(["A=run_s{seed}"]) == [("A", "run_s{seed}")]
    x, y = collapse_duplicate_steps(
        np.array([20.0, 10.0, 10.0]), np.array([3.0, 1.0, 2.0])
    )
    np.testing.assert_allclose(x, [10.0, 20.0])
    np.testing.assert_allclose(y, [1.5, 3.0])
    np.testing.assert_allclose(ema(np.array([0.0, 1.0, 1.0]), 3), [0.0, 0.5, 0.75])

    curves = [
        (np.array([0.0, 10.0, 20.0]), np.array([0.0, 1.0, 2.0])),
        (np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 10.0, 20.0]), np.array([2.0, 3.0, 4.0])),
    ]
    grid = common_grid(curves, 10.0)
    aligned = align_curves(curves, grid)
    center, lower, upper, count = summarize(aligned, "iqr")
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(lower, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(upper, [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(count, [3, 3, 3])

    center, lower, upper, count = summarize(aligned, "ci95")
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    expected_radius = 4.303 / np.sqrt(3.0)
    np.testing.assert_allclose(upper - center, expected_radius, rtol=1e-3)
    np.testing.assert_allclose(center - lower, expected_radius, rtol=1e-3)
    print("multi-seed aggregation smoke test passed")


if __name__ == "__main__":
    main()
