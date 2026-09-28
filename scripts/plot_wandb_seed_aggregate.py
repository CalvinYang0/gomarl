#!/usr/bin/env python3
"""Plot aligned multi-seed W&B learning curves with uncertainty bands.

Examples
--------
Reproduce the statistic used by the GoMARL paper (median and IQR)::

    python scripts/plot_wandb_seed_aggregate.py \
      --project hjh331-sjtu/gomarl \
      --metric test_game_win_mean \
      --series 'HyperSelect=grf_counter_paper_hyperselect_10m_s{seed}' \
      --seeds 1 2 3 --band iqr --percent \
      --output figures/counter_hyperselect_iqr.pdf

Plot a mean curve and a 95% Student-t confidence interval::

    python scripts/plot_wandb_seed_aggregate.py \
      --project hjh331-sjtu/gomarl \
      --metric test_game_win_mean \
      --series 'HyperSelect=grf_counter_paper_hyperselect_10m_s{seed}' \
      --seeds 1 2 3 --band ci95 --percent \
      --output figures/counter_hyperselect_ci95.pdf

Each ``--series`` value is ``LABEL=RUN_NAME_TEMPLATE``.  The template must
contain ``{seed}``.  Curves are interpolated onto a common environment-step
grid without extrapolation.  By default a point is shown only when every
requested seed contributes to it, so the uncertainty band never silently
changes from three seeds to one seed near the end of training.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_series(values: Sequence[str]) -> List[Tuple[str, str]]:
    parsed = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--series must be LABEL=RUN_NAME_TEMPLATE: " + value)
        label, template = value.split("=", 1)
        label, template = label.strip(), template.strip()
        if not label or not template or "{seed}" not in template:
            raise ValueError(
                "Each --series needs a label and a {seed} placeholder: " + value
            )
        if label in labels:
            raise ValueError("Duplicate series label: " + label)
        labels.add(label)
        parsed.append((label, template))
    return parsed


def collapse_duplicate_steps(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort points and average repeated x values."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return x, y
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.zeros(unique.size, dtype=float)
    counts = np.zeros(unique.size, dtype=float)
    np.add.at(sums, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return unique, sums / counts


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Causal exponential moving average; span=1 leaves data unchanged."""
    if span <= 1 or values.size == 0:
        return values.copy()
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, values.size):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def common_grid(curves: Sequence[Tuple[np.ndarray, np.ndarray]], step_size: float) -> np.ndarray:
    starts = [curve[0][0] for curve in curves if curve[0].size]
    ends = [curve[0][-1] for curve in curves if curve[0].size]
    if not starts or not ends:
        raise ValueError("No finite history points were loaded")
    start = math.floor(min(starts) / step_size) * step_size
    end = math.ceil(max(ends) / step_size) * step_size
    return np.arange(start, end + step_size * 0.5, step_size, dtype=float)


def align_curves(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> np.ndarray:
    aligned = np.full((len(curves), grid.size), np.nan, dtype=float)
    for row, (x, y) in enumerate(curves):
        if x.size == 0:
            continue
        inside = (grid >= x[0]) & (grid <= x[-1])
        aligned[row, inside] = np.interp(grid[inside], x, y)
    return aligned


def _t_critical_975(degrees_of_freedom: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, degrees_of_freedom))
    except ImportError:
        # Exact/common values followed by the normal approximation.  In
        # particular, n=3 uses df=2 and t*=4.303 rather than 1.96.
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        }
        return table.get(degrees_of_freedom, 1.96)


def summarize(aligned: np.ndarray, band: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(np.isfinite(aligned), axis=0)
    if band == "iqr":
        center = np.nanmedian(aligned, axis=0)
        lower = np.nanpercentile(aligned, 25, axis=0)
        upper = np.nanpercentile(aligned, 75, axis=0)
        return center, lower, upper, counts

    center = np.nanmean(aligned, axis=0)
    lower = np.full(center.shape, np.nan)
    upper = np.full(center.shape, np.nan)
    for index, n_value in enumerate(counts):
        n = int(n_value)
        if n < 2:
            continue
        values = aligned[:, index]
        values = values[np.isfinite(values)]
        std = float(np.std(values, ddof=1))
        if band == "std":
            radius = std
        elif band == "sem":
            radius = std / math.sqrt(n)
        elif band == "ci95":
            radius = _t_critical_975(n - 1) * std / math.sqrt(n)
        else:
            raise ValueError("Unknown band: " + band)
        lower[index], upper[index] = center[index] - radius, center[index] + radius
    return center, lower, upper, counts


def fetch_run_curve(api, project: str, run_name: str, metric: str) -> Tuple[np.ndarray, np.ndarray, str]:
    matches = list(api.runs(project, filters={"display_name": run_name}))
    if not matches:
        raise RuntimeError("No W&B run named {!r} in {}".format(run_name, project))

    candidates = []
    for run in matches:
        rows = list(run.scan_history(keys=["_step", metric], page_size=10000))
        points = [
            (row.get("_step"), row.get(metric))
            for row in rows
            if row.get("_step") is not None and row.get(metric) is not None
        ]
        if points:
            x = np.asarray([point[0] for point in points], dtype=float)
            y = np.asarray([point[1] for point in points], dtype=float)
            x, y = collapse_duplicate_steps(x, y)
            candidates.append((float(x[-1]), x.size, x, y, run.id))

    if not candidates:
        raise RuntimeError(
            "Run {!r} exists but has no finite {!r} history".format(run_name, metric)
        )
    # Retries can create duplicate display names.  Prefer the run that reached
    # the greatest environment step, then the one containing more evaluations.
    _, _, x, y, run_id = max(candidates, key=lambda item: (item[0], item[1]))
    if len(candidates) > 1:
        print(
            "WARNING: {} has {} copies; selected run {} ending at step {:g}".format(
                run_name, len(candidates), run_id, x[-1]
            ),
            file=sys.stderr,
        )
    return x, y, run_id


def write_csv(path: Path, grid: np.ndarray, summaries: Dict[str, Tuple[np.ndarray, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(summaries)
    fields = ["step"]
    for label in labels:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        fields.extend([slug + "_center", slug + "_lower", slug + "_upper", slug + "_n"])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, step in enumerate(grid):
            row: List[object] = [step]
            for label in labels:
                center, lower, upper, counts = summaries[label]
                row.extend([center[index], lower[index], upper[index], int(counts[index])])
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="W&B ENTITY/PROJECT")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--series", action="append", required=True,
                        help="LABEL=RUN_NAME_TEMPLATE; repeat for comparisons")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--band", choices=("iqr", "ci95", "std", "sem"), default="iqr")
    parser.add_argument("--step-size", type=float, default=10000.0)
    parser.add_argument("--max-step", type=float)
    parser.add_argument("--min-seeds", type=int,
                        help="Minimum contributing seeds; default is all requested seeds")
    parser.add_argument("--smooth-span", type=int, default=1,
                        help="Causal EMA span applied to each seed before aggregation")
    parser.add_argument("--percent", action="store_true",
                        help="Multiply y values by 100 and label axis as percent")
    parser.add_argument("--title", default="")
    parser.add_argument("--xlabel", default="Environment Steps")
    parser.add_argument("--ylabel", default="Test Win Rate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path,
                        help="Aggregated values; default is OUTPUT with .csv suffix")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    series = parse_series(args.series)
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.smooth_span < 1:
        raise ValueError("--smooth-span must be at least 1")
    required = args.min_seeds if args.min_seeds is not None else len(args.seeds)
    if required < 1 or required > len(args.seeds):
        raise ValueError("--min-seeds must be between 1 and the number of seeds")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install/login to wandb before using this script") from exc
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    api = wandb.Api()
    loaded: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    all_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for label, template in series:
        curves = []
        for seed in args.seeds:
            run_name = template.format(seed=seed)
            x, y, run_id = fetch_run_curve(api, args.project, run_name, args.metric)
            y = ema(y, args.smooth_span)
            if args.percent:
                y = y * 100.0
            curves.append((x, y))
            all_curves.append((x, y))
            print(
                "loaded label={!r} seed={} run={} id={} points={} end_step={:g}".format(
                    label, seed, run_name, run_id, x.size, x[-1]
                )
            )
        loaded[label] = curves

    grid = common_grid(all_curves, args.step_size)
    if args.max_step is not None:
        grid = grid[grid <= args.max_step]

    summaries: Dict[str, Tuple[np.ndarray, ...]] = {}
    for label, curves in loaded.items():
        aligned = align_curves(curves, grid)
        center, lower, upper, counts = summarize(aligned, args.band)
        insufficient = counts < required
        center[insufficient] = np.nan
        lower[insufficient] = np.nan
        upper[insufficient] = np.nan
        summaries[label] = (center, lower, upper, counts)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (label, _) in enumerate(series):
        center, lower, upper, _ = summaries[label]
        color = colors[index % len(colors)]
        axis.plot(grid, center, color=color, linewidth=2.0, label=label)
        axis.fill_between(grid, lower, upper, color=color, alpha=0.20, linewidth=0)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axis.set_xlabel(args.xlabel)
    ylabel = args.ylabel + (" (%)" if args.percent else "")
    axis.set_ylabel(ylabel)
    if args.percent:
        axis.set_ylim(0.0, 100.0)
    if args.title:
        axis.set_title(args.title)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "{:.0f}M".format(value / 1e6)
                      if abs(value) >= 1e6 else "{:.0f}K".format(value / 1e3))
    )
    axis.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")
    write_csv(csv_path, grid, summaries)
    print("wrote figure: " + str(args.output))
    print("wrote aggregate data: " + str(csv_path))
    print("statistic: " + ("median + 25/75 percentiles" if args.band == "iqr" else "mean + " + args.band))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
