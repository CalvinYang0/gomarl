#!/usr/bin/env python3
"""Plot aligned multi-seed W&B learning curves with uncertainty bands.

The defaults reproduce CASH's public plotting pipeline: compute the mean and
sample standard deviation across seeds at each timestep, smooth those two
statistics separately with a centered rolling window, downsample, and draw
``mean +/- std``.

Examples
--------
Use the CASH defaults (mean +/- sample std, centered windows of 100, then
10x downsampling)::

    python scripts/plot_wandb_seed_aggregate.py \
      --project hjh331-sjtu/gomarl \
      --metric test_game_win_mean \
      --series 'HyperSelect=grf_counter_paper_hyperselect_10m_s{seed}' \
      --seeds 1 2 3 --percent \
      --output figures/counter_hyperselect_cash.pdf

Plot a mean curve and a 95% Student-t confidence interval without CASH-style
post-aggregation smoothing::

    python scripts/plot_wandb_seed_aggregate.py \
      --project hjh331-sjtu/gomarl \
      --metric test_game_win_mean \
      --series 'HyperSelect=grf_counter_paper_hyperselect_10m_s{seed}' \
      --seeds 1 2 3 --band ci95 --mean-window 1 --std-window 1 \
      --downsample-factor 1 --percent \
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
import hashlib
import json
import math
import os
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


def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Pandas-compatible centered rolling mean with ``min_periods=1``."""
    if window <= 1 or values.size == 0:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    # pandas ``rolling(window=..., center=True)`` assigns the extra point of
    # an even-sized window to the left side of the labelled position.
    left = window // 2
    right = (window - 1) // 2
    for index in range(values.size):
        start = max(0, index - left)
        stop = min(values.size, index + right + 1)
        chunk = values[start:stop]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            result[index] = float(np.mean(finite))
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


def _local_run_files(wandb_root: Path, run_name: str) -> List[Path]:
    matches = []
    for directory in wandb_root.glob("offline-run-*"):
        config = directory / "files" / "config.yaml"
        if not config.is_file():
            continue
        try:
            if run_name not in config.read_text(errors="ignore"):
                continue
        except OSError:
            continue
        files = list(directory.glob("run-*.wandb"))
        if len(files) == 1:
            matches.append(files[0])
    return matches


def fetch_local_run_curve_by_paths(
    wandb_root: Path, paths: Sequence[Path], metric: str
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read one metric from exact retained W&B files, preferring longest history."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    candidates = []
    for path in paths:
        stat = path.stat()
        cache_key = hashlib.sha256(
            "{}\0{}\0{}\0{}".format(
                path.resolve(), stat.st_size, stat.st_mtime_ns, metric
            ).encode("utf-8")
        ).hexdigest()
        cache_dir = Path(os.environ.get(
            "PLOT_CACHE_DIR", str(wandb_root / ".plot-curve-cache")
        )).resolve()
        cache_path = cache_dir / (cache_key + ".npz")
        if cache_path.is_file():
            with np.load(cache_path) as cached:
                x = np.asarray(cached["x"], dtype=float)
                y = np.asarray(cached["y"], dtype=float)
            if x.size and y.size:
                candidates.append((float(x[-1]), x.size, x, y, path))
                continue

        points = []
        scanner = DataStore()
        scanner.open_for_scan(str(path))
        try:
            while True:
                data = scanner.scan_data()
                if data is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if not record.HasField("history"):
                    continue
                step = (
                    float(record.history.step.num)
                    if record.history.HasField("step") else None
                )
                value = None
                for item in record.history.item:
                    if item.nested_key:
                        item_key = ".".join(item.nested_key)
                    else:
                        item_key = item.key
                    if item_key not in ("_step", metric):
                        continue
                    try:
                        parsed = json.loads(item.value_json)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if item_key == "_step" and step is None:
                        if isinstance(parsed, (int, float)):
                            step = float(parsed)
                    elif item_key == metric:
                        if isinstance(parsed, (int, float)):
                            value = float(parsed)
                if step is not None and value is not None:
                    points.append((step, value))
        finally:
            if hasattr(scanner, "close"):
                scanner.close()
        if points:
            x = np.asarray([point[0] for point in points], dtype=float)
            y = np.asarray([point[1] for point in points], dtype=float)
            x, y = collapse_duplicate_steps(x, y)
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, x=x, y=y)
            temporary.replace(cache_path)
            candidates.append((float(x[-1]), x.size, x, y, path))

    if not candidates:
        raise RuntimeError("No retained offline history for metric {!r}".format(metric))
    _, _, x, y, path = max(candidates, key=lambda item: (item[0], item[1]))
    return x, y, str(path)


def fetch_local_run_curve(
    wandb_root: Path, run_name: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read a scalar history directly from retained offline W&B records."""
    try:
        return fetch_local_run_curve_by_paths(
            wandb_root, _local_run_files(wandb_root, run_name), metric
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "No retained offline history for {!r} metric {!r}".format(
                run_name, metric
            )
        ) from exc


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
    parser.add_argument("--wandb-root", type=Path,
                        help="Retained offline-run root; preferred over cloud W&B")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--series", action="append", required=True,
                        help="LABEL=RUN_NAME_TEMPLATE; repeat for comparisons")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--band", choices=("iqr", "ci95", "std", "sem"), default="std",
                        help="Uncertainty band; CASH uses std (default)")
    parser.add_argument("--step-size", type=float, default=10000.0)
    parser.add_argument("--max-step", type=float)
    parser.add_argument("--min-seeds", type=int,
                        help="Minimum contributing seeds; default is all requested seeds")
    parser.add_argument("--smooth-span", type=int, default=1,
                        help="Optional pre-aggregation per-seed EMA; CASH leaves this at 1")
    parser.add_argument("--mean-window", type=int, default=100,
                        help="Centered rolling window for the aggregate center (CASH: 100)")
    parser.add_argument("--std-window", type=int, default=100,
                        help="Centered rolling window for band radius (CASH: 100)")
    parser.add_argument("--downsample-factor", type=int, default=10,
                        help="Keep every Nth aggregate point after smoothing (CASH: 10)")
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
    if args.mean_window < 1 or args.std_window < 1:
        raise ValueError("--mean-window and --std-window must be at least 1")
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be at least 1")
    required = args.min_seeds if args.min_seeds is not None else len(args.seeds)
    if required < 1 or required > len(args.seeds):
        raise ValueError("--min-seeds must be between 1 and the number of seeds")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install/login to wandb before using this script") from exc
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    api = None
    loaded: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    all_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for label, template in series:
        curves = []
        for seed in args.seeds:
            run_name = template.format(seed=seed)
            source = None
            if args.wandb_root is not None:
                try:
                    x, y, source = fetch_local_run_curve(
                        args.wandb_root.resolve(), run_name, args.metric
                    )
                except RuntimeError as exc:
                    print("WARNING: {}; falling back to cloud".format(exc), file=sys.stderr)
            if source is None:
                if api is None:
                    api = wandb.Api()
                x, y, source = fetch_run_curve(
                    api, args.project, run_name, args.metric
                )
            y = ema(y, args.smooth_span)
            if args.percent:
                y = y * 100.0
            curves.append((x, y))
            all_curves.append((x, y))
            print(
                "loaded label={!r} seed={} run={} id={} points={} end_step={:g}".format(
                    label, seed, run_name, source, x.size, x[-1]
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

    # CASH smooths the already aggregated mean and standard deviation
    # separately, using centered rolling averages, and downsamples afterwards.
    for label in summaries:
        center, lower, upper, counts = summaries[label]
        smooth_center = centered_rolling_mean(center, args.mean_window)
        if args.band == "iqr":
            smooth_lower = centered_rolling_mean(lower, args.std_window)
            smooth_upper = centered_rolling_mean(upper, args.std_window)
        else:
            radius = (upper - lower) / 2.0
            smooth_radius = centered_rolling_mean(radius, args.std_window)
            smooth_lower = smooth_center - smooth_radius
            smooth_upper = smooth_center + smooth_radius
        valid = counts >= required
        smooth_center[~valid] = np.nan
        smooth_lower[~valid] = np.nan
        smooth_upper[~valid] = np.nan
        summaries[label] = (smooth_center, smooth_lower, smooth_upper, counts)

    keep = np.arange(0, grid.size, args.downsample_factor)
    grid = grid[keep]
    summaries = {
        label: tuple(array[keep] for array in summary)
        for label, summary in summaries.items()
    }

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
    statistic = "median + 25/75 percentiles" if args.band == "iqr" else "mean + " + args.band
    print(
        "statistic: {}; centered windows center={} band={}; downsample={}".format(
            statistic, args.mean_window, args.std_window, args.downsample_factor
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
