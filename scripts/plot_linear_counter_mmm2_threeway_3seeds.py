#!/usr/bin/env python3
"""Plot only the 18 Counter/MMM2 threeway runs; latest attempt per seed.

Read local W&B, cloud W&B, or Sacred records. Never join restarted histories
or reuse the older 5M suites. Aggregate only each model's shared seed interval.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import (  # noqa: E402
    GROUP, build_plans,
)
from plot_linear_directkl_four_model_3seeds import (  # noqa: E402
    cloud_curve, cloud_run_index, local_run_index,
)

SCENES = {
    "grf_counter": ("Counterattack Easy", "test_game_win_mean"),
    "smac_mmm2": ("MMM2", "test_battle_won_mean"),
}
LABELS = {
    "linear_baseline": ("Single-head Linear (no gate)", "#1f77b4"),
    "linear_bayesg_kl80_keep": ("Linear + direct KL80", "#ff7f0e"),
    "linear_obs_gate_kl80aux_multiply": ("Linear + auxiliary KL80 multiply", "#9467bd"),
}
FIELDS = (
    "scene", "model", "seed", "run_name", "run_id", "source",
    "attempt_start", "points", "start_step", "end_step", "coverage",
    "older_wandb_run_ids", "note",
)
TARGET_STEPS = 10000000


def timestamp(value):
    if not value:
        return 0.0
    date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.timestamp()


def discover_sacred(root, wanted):
    result = {}
    if not root.is_dir():
        return result
    for path in root.glob("*/*/*/config.json"):
        try:
            config = json.loads(path.read_text())
            name = config.get("wandb_run_name")
            if name not in wanted:
                continue
            run = json.loads((path.parent / "run.json").read_text())
            started = timestamp(run.get("start_time"))
            result.setdefault(name, []).append((started, path.parent))
        except (OSError, ValueError, TypeError) as exc:
            print("WARNING: Sacred metadata {}: {}".format(path, exc), file=sys.stderr)
    return result


def select_latest(plan, metric, local, cloud, sacred, wandb_root, plotter):
    name = plan["job_name"]
    attempts = []
    for path in local.get(name, []):
        stamp = path.parent.name.split("-")[2]
        started = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        run_id = path.stem[len("run-"):]
        attempts.append((started, run_id, "local", path))
    for run in cloud.get(name, []):
        attempts.append((timestamp(run.created_at), run.id, "cloud", run))
    for started, path in sacred.get(name, []):
        attempts.append((started, "sacred:" + str(path), "sacred", path))
    if not attempts:
        return None, {}, "No matching run found"

    # A new attempt without test points stays missing; an older successful
    # attempt must not silently stand in for the restarted experiment.
    newest = max(attempts, key=lambda item: item[0])
    same_run = [item for item in attempts if item[1] == newest[1]]
    same_run.sort(key=lambda item: {"local": 0, "cloud": 1, "sacred": 2}[item[2]])
    details = {
        "run_id": newest[1], "source": newest[2],
        "attempt_start": datetime.fromtimestamp(newest[0], timezone.utc).isoformat(),
        "older_wandb_run_ids": ",".join(sorted({
            item[1] for item in attempts
            if item[2] != "sacred" and item[1] != newest[1]
        })),
    }
    errors = []
    for _, run_id, source, payload in same_run:
        try:
            if source == "local":
                x, y, _ = plotter.fetch_local_run_curve_by_paths(wandb_root, [payload], metric)
            elif source == "cloud":
                x, y = cloud_curve(payload, metric, plotter)
            else:
                import numpy as np
                info = json.loads((payload / "info.json").read_text())
                x, y = plotter.collapse_duplicate_steps(
                    np.asarray(info.get(metric + "_T", []), dtype=float),
                    np.asarray(info.get(metric, []), dtype=float),
                )
            import numpy as np
            valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 10050000)
            x, y = x[valid], y[valid]
            if not x.size:
                raise ValueError("Latest attempt has no test-win points")
            if np.any((y < 0) | (y > 1)):
                raise ValueError("Expected test-win fractions in [0, 1]")
            details.update(source=source, run_id=run_id)
            return (x, y), details, ""
        except Exception as exc:
            errors.append(str(exc))
    return None, details, "; ".join(errors)


def render(scene, entries, output_dir, plotter, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    title, _ = SCENES[scene]
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    rows = []
    for model, (label, color) in LABELS.items():
        seeds = entries[model]
        if not seeds:
            continue
        curves = [curve for _, curve in seeds]
        start = max(x[0] for x, _ in curves)
        end = min(TARGET_STEPS, min(x[-1] for x, _ in curves))
        if start > end:
            continue
        grid = np.arange(np.ceil(start / 10000) * 10000, end + 0.5, 10000)
        if not grid.size:
            continue
        aligned = plotter.align_curves(curves, grid)
        center = aligned.mean(axis=0)
        radius = aligned.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(grid.size)
        center = plotter.centered_rolling_mean(center, window)
        radius = plotter.centered_rolling_mean(radius, window)
        keep = np.unique(np.r_[np.arange(0, grid.size, 10), grid.size - 1])
        legend = "{} ({}/3 seeds; to {:.2f}M)".format(label, len(seeds), end / 1e6)
        axis.plot(grid[keep], center[keep] * 100, color=color, linewidth=2,
                  marker="o" if grid.size == 1 else None, label=legend)
        if len(curves) > 1:
            axis.fill_between(grid[keep], np.clip((center - radius)[keep] * 100, 0, 100),
                              np.clip((center + radius)[keep] * 100, 0, 100),
                              color=color, alpha=0.18, linewidth=0)
        rows.extend(zip([model] * grid.size, grid, center, radius, [len(seeds)] * grid.size))
    axis.set(title=title + " — current threeway results", xlabel="Environment Steps",
             ylabel="Test Win Rate (%)", xlim=(0, TARGET_STEPS), ylim=(0, 100))
    if rows:
        axis.set_xlim(0, min(TARGET_STEPS, max(100000, np.ceil(max(row[1] for row in rows) / 100000) * 100000)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}M".format(v / 1e6)))
    axis.grid(alpha=0.35, linestyle="--")
    if rows:
        axis.legend(frameon=False, fontsize=8)
    else:
        axis.text(0.5, 0.5, "No test-win data / shared seed interval yet",
                  ha="center", va="center", transform=axis.transAxes)
    fig.suptitle("Mean ± sample std; centered window={} test points\n"
                 "Latest attempt per seed; shared seed interval only".format(window), fontsize=9)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / (scene + "_three_seed." + suffix)
        fig.savefig(path, dpi=250)
        paths.append(path)
    plt.close(fig)
    with (output_dir / (scene + "_aggregate.csv")).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "step", "mean_fraction_smoothed", "sample_std_smoothed", "seed_count"])
        writer.writerows(rows)

    # Show individual seeds separately so early truncation and seed variance
    # are visible instead of disappearing behind the shared-interval average.
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True, sharey=True)
    seed_ends = [x[-1] for seeds in entries.values() for _, (x, _) in seeds]
    seed_horizon = min(TARGET_STEPS, max(100000, np.ceil(max(seed_ends, default=TARGET_STEPS) / 100000) * 100000))
    for axis, (model, (label, _)) in zip(axes, LABELS.items()):
        for seed, (x, y) in entries[model]:
            axis.plot(x, y * 100, alpha=0.2, linewidth=0.6, color="C" + str(seed - 1))
            axis.plot(x, plotter.centered_rolling_mean(y, window) * 100,
                      label="seed {} (to {:.2f}M)".format(seed, x[-1] / 1e6),
                      color="C" + str(seed - 1), linewidth=1.5)
        axis.set(title=label, xlim=(0, seed_horizon), ylim=(0, 100), xlabel="Steps")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}M".format(v / 1e6)))
        axis.grid(alpha=0.3, linestyle="--")
        if entries[model]:
            axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Test Win Rate (%)")
    fig.suptitle(title + " — individual seeds (faint: raw; solid: smoothed)")
    path = output_dir / (scene + "_individual_seeds.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def upload(project, output_dir, outputs, inventory, window):
    import wandb
    entity, project_name = project.split("/", 1)
    with wandb.init(entity=entity, project=project_name,
                    id=hashlib.sha1((GROUP + "_plots").encode()).hexdigest()[:8], resume="allow",
                    name=GROUP + "_figures", group=GROUP, job_type="analysis", mode="online",
                    dir=str(output_dir), config={"target_steps": TARGET_STEPS, "mean_window": window,
                                                "selection": "latest attempt per seed", "seeds": [1, 2, 3]}) as run:
        payload = {"seed_inventory": wandb.Table(columns=list(FIELDS),
                    data=[[row[field] for field in FIELDS] for row in inventory])}
        for scene, paths in outputs.items():
            payload[scene + "_three_seed"] = wandb.Image(str(paths[0]))
            payload[scene + "_individual_seeds"] = wandb.Image(str(paths[-1]))
        run.log(payload)
        artifact = wandb.Artifact(GROUP + "_available_seed_data", type="dataset")
        for path in sorted(output_dir.glob("*.csv")):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        print("Uploaded figures: " + run.url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="hjh331-sjtu/gomarl")
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--mean-window", type=int, default=100)
    args = parser.parse_args()
    if args.mean_window < 1 or "/" not in args.project:
        parser.error("positive --mean-window and ENTITY/PROJECT are required")
    output_dir = args.output_dir or args.runtime_root / "figures/linear_threeway_10m"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLOT_CACHE_DIR", str(output_dir / "curve_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))
    import plot_wandb_seed_aggregate as plotter
    plans = build_plans(ROOT)
    wanted = {p["job_name"] for p in plans}
    wandb_root = args.runtime_root / "wandb"
    local = local_run_index(wandb_root, wanted) if wandb_root.is_dir() else {}
    cloud = {} if args.local_only else cloud_run_index(args.project, wanted)
    sacred = discover_sacred(args.runtime_root / "results/sacred", wanted)
    entries = {scene: {model: [] for model in LABELS} for scene in SCENES}
    inventory, raw_rows = [], []
    for plan in plans:
        scene, model, seed = plan["scene"], plan["label"], plan["seed"]
        curve, details, note = select_latest(plan, SCENES[scene][1], local, cloud,
                                             sacred, wandb_root, plotter)
        row = dict.fromkeys(FIELDS, "")
        row.update(scene=scene, model=model, seed=seed, run_name=plan["job_name"],
                   points=0, coverage="missing", note=note, **details)
        if curve is not None:
            x, y = curve
            entries[scene][model].append((seed, curve))
            row.update(points=int(x.size), start_step=float(x[0]), end_step=float(x[-1]),
                       coverage="complete" if x[-1] >= TARGET_STEPS else "partial")
            raw_rows.extend((scene, model, seed, row["run_id"], step, value) for step, value in zip(x, y))
        inventory.append(row)
        print("{}: {} points={} end={} {}".format(plan["job_name"], row["coverage"],
                                                  row["points"], row["end_step"], note))
    with (output_dir / "seed_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(inventory)
    if not raw_rows:
        raise SystemExit("No test-win histories for this batch yet. Inventory: " + str(output_dir / "seed_inventory.csv"))
    with (output_dir / "seed_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "model", "seed", "run_id", "step", "win_fraction"])
        writer.writerows(raw_rows)
    outputs = {scene: render(scene, entries[scene], output_dir, plotter, args.mean_window)
               for scene in SCENES}
    print("Figures and seed inventory: " + str(output_dir))
    if not args.no_upload:
        upload(args.project, output_dir, outputs, inventory, args.mean_window)


if __name__ == "__main__":
    main()
