#!/usr/bin/env python3
"""Plot available four-map Linear study seeds and upload four W&B images.

The curves use the same CASH-style mean/sample-std, centered-100 smoothing,
and 10x downsampling as the earlier paper plots. Partial seed coverage is
reported explicitly; a one-seed line has no uncertainty band.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_linear_directkl_four_model_3seeds import (  # noqa: E402
    GROUP, MODELS, build_plans,
)


TITLES = {
    "grf_counter": ("Counterattack Easy", "counter", "test_game_win_mean"),
    "grf_pass": ("Pass and Shoot", "pass", "test_game_win_mean"),
    "smac_5m6m": ("5m vs. 6m", "5m6m", "test_battle_won_mean"),
    "smac_mmm2": ("MMM2", "mmm2", "test_battle_won_mean"),
}
LABELS = {
    "linear_baseline": ("Linear (no gate)", "#1f77b4"),
    "linear_bayesg_kl80_keep": ("Linear + KL80", "#ff7f0e"),
    "linear_directkl_qme_action_q_episode_mean": (
        "+ KL80 + QME epmean", "#d62728"
    ),
    "linear_directkl_qme_td_quality": (
        "+ KL80 + QME td_quality", "#9467bd"
    ),
}
INVENTORY_FIELDS = (
    "scene", "model", "seed", "run_name", "run_id", "source",
    "points", "end_step", "coverage",
)


def local_run_index(wandb_root: Path, wanted: set[str]) -> dict[str, list[Path]]:
    """Index exact run names without reading scalar history of unrelated jobs."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    index: dict[str, list[Path]] = {}
    exact_name = re.compile(
        r"(?<![A-Za-z0-9_])(?:{})(?![A-Za-z0-9_])".format(
            "|".join(re.escape(name) for name in sorted(wanted, key=len, reverse=True))
        )
    )
    for directory in wandb_root.glob("offline-run-*"):
        files = list(directory.glob("run-*.wandb"))
        if len(files) != 1:
            continue
        path = files[0]
        names = set()
        config = directory / "files" / "config.yaml"
        if config.is_file():
            try:
                contents = config.read_text(errors="ignore")
                names.update(match.group(0) for match in exact_name.finditer(contents))
            except OSError:
                pass
        if not names:
            scanner = DataStore()
            try:
                scanner.open_for_scan(str(path))
                for _ in range(128):
                    data = scanner.scan_data()
                    if data is None:
                        break
                    record = wandb_internal_pb2.Record()
                    record.ParseFromString(data)
                    if record.HasField("run"):
                        if record.run.display_name in wanted:
                            names.add(record.run.display_name)
                        break
            except (OSError, ValueError) as exc:
                print("WARNING: cannot index {}: {}".format(path, exc), file=sys.stderr)
            finally:
                if hasattr(scanner, "close"):
                    scanner.close()
        for name in names:
            index.setdefault(name, []).append(path)
    return index


def cloud_run_index(project: str, wanted: set[str]) -> dict[str, list[Any]]:
    """Fetch only matching run metadata; histories load on demand."""
    import wandb

    try:
        api = wandb.Api(timeout=90)
        runs = api.runs(
            project, filters={"display_name": {"$in": sorted(wanted)}},
            per_page=500,
        )
        found = list(runs)
        if not found:
            found = [run for run in api.runs(project, per_page=500)
                     if run.name in wanted]
    except Exception as exc:
        print("WARNING: filtered W&B discovery failed: {}".format(exc), file=sys.stderr)
        try:
            api = wandb.Api(timeout=90)
            found = [run for run in api.runs(project, per_page=500)
                     if run.name in wanted]
        except Exception as fallback_exc:
            print("WARNING: cloud discovery unavailable: {}".format(
                fallback_exc), file=sys.stderr)
            return {}
    result: dict[str, list[Any]] = {}
    for run in found:
        if run.name in wanted:
            result.setdefault(run.name, []).append(run)
    return result


def cloud_curve(run: Any, metric: str, plotter: Any):
    rows = run.scan_history(keys=["_step", metric], page_size=10000)
    points = [
        (row.get("_step"), row.get(metric)) for row in rows
        if row.get("_step") is not None and row.get(metric) is not None
    ]
    if not points:
        raise RuntimeError("No {} points in cloud run {}".format(metric, run.id))
    import numpy as np

    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    return plotter.collapse_duplicate_steps(x, y)


def choose_curve(
    plan: dict[str, Any], metric: str, wandb_root: Path,
    local: dict[str, list[Path]], cloud: dict[str, list[Any]], plotter: Any,
):
    best = None
    for name in [plan["job_name"]] + plan["historical_candidates"]:
        paths = local.get(name, [])
        if paths:
            try:
                x, y, source = plotter.fetch_local_run_curve_by_paths(
                    wandb_root, paths, metric
                )
                run_id = Path(source).stem.removeprefix("run-")
                candidate = (x, y, name, run_id, "local")
                if best is None or (x[-1], x.size) > (best[0][-1], best[0].size):
                    best = candidate
                continue
            except Exception as exc:
                print("WARNING: {}: {}".format(name, exc), file=sys.stderr)
        for run in cloud.get(name, []):
            try:
                x, y = cloud_curve(run, metric, plotter)
            except Exception as exc:
                print("WARNING: {}: {}".format(name, exc), file=sys.stderr)
                continue
            candidate = (x, y, name, run.id, "cloud")
            if best is None or (x[-1], x.size) > (best[0][-1], best[0].size):
                best = candidate
    return best


def render_scene(scene: str, entries: dict[str, list[Any]], output_dir: Path, plotter: Any):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    title, slug, _ = TITLES[scene]
    all_curves = [curve for curves in entries.values() for curve in curves]
    output = output_dir / (slug + "_linear_four_model_available_seeds.png")
    csv_path = output.with_suffix(".csv")
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    summaries = {}
    if all_curves:
        grid = plotter.common_grid(all_curves, 10000.0)
        grid = grid[grid <= 5000000]
        for label, _ in MODELS:
            curves = entries[label]
            if not curves:
                continue
            aligned = plotter.align_curves(curves, grid)
            center, lower, upper, counts = plotter.summarize(aligned, "std")
            required = len(curves)
            center[counts < required] = np.nan
            lower[counts < required] = np.nan
            upper[counts < required] = np.nan
            smooth_center = plotter.centered_rolling_mean(center, 100)
            smooth_radius = plotter.centered_rolling_mean((upper - lower) / 2.0, 100)
            smooth_center[counts < required] = np.nan
            smooth_lower = smooth_center - smooth_radius
            smooth_upper = smooth_center + smooth_radius
            summaries[label] = (smooth_center, smooth_lower, smooth_upper, counts)
        keep = np.arange(0, grid.size, 10)
        grid = grid[keep]
        summaries = {
            label: tuple(array[keep] for array in summary)
            for label, summary in summaries.items()
        }
        for label, _ in MODELS:
            if label not in summaries:
                continue
            center, lower, upper, _ = summaries[label]
            base_label, color = LABELS[label]
            n = len(entries[label])
            axis.plot(grid, center * 100, color=color, linewidth=2,
                      label="{} ({}/3 seeds)".format(base_label, n))
            if n >= 2:
                axis.fill_between(grid, lower * 100, upper * 100,
                                  color=color, alpha=0.18, linewidth=0)
        plotter.write_csv(csv_path, grid, summaries)
    else:
        axis.text(0.5, 0.5, "No available test-win histories yet",
                  ha="center", va="center", transform=axis.transAxes)
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["step"])

    axis.set_title(title + " — current 5M seed coverage")
    axis.set_xlabel("Environment Steps")
    axis.set_ylabel("Test Win Rate (%)")
    axis.set_xlim(0, 5000000)
    axis.set_ylim(0, 100)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "{:g}M".format(value / 1e6))
    )
    if summaries:
        axis.legend(frameon=False, fontsize=9)
    fig.text(0.02, 0.01,
             "Mean ± sample std; CASH centered-100 smoothing; only shared seed steps shown."
             " One seed: line only.", fontsize=8)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output, csv_path


def upload_outputs(project: str, outputs: dict[str, Path], inventory: list[dict[str, Any]]):
    import wandb

    entity, project_name = project.split("/", 1)
    run_id = hashlib.sha1(GROUP.encode("utf-8")).hexdigest()[:8]
    upload_root = next(iter(outputs.values())).parent / "wandb_upload"
    upload_root.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        entity=entity, project=project_name, id=run_id, resume="allow",
        name=GROUP + "_figures", group=GROUP, job_type="analysis",
        mode="online", dir=str(upload_root),
        config={"target_steps": 5000000, "seeds": [1, 2, 3]},
    ) as run:
        table = wandb.Table(
            columns=list(INVENTORY_FIELDS),
            data=[[row[field] for field in INVENTORY_FIELDS] for row in inventory],
        )
        payload = {"seed_inventory": table}
        for scene, path in outputs.items():
            slug = TITLES[scene][1]
            payload[slug + "_three_seed"] = wandb.Image(str(path))
        run.log(payload)
        artifact = wandb.Artifact(GROUP + "_aggregate", type="dataset")
        artifact.add_file(str(next(iter(outputs.values())).parent / "seed_inventory.csv"))
        for path in outputs.values():
            artifact.add_file(str(path.with_suffix(".csv")))
        run.log_artifact(artifact)
        print("Uploaded {} figures and {} CSV files to {}".format(
            len(outputs), len(outputs) + 1, run.url
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="hjh331-sjtu/gomarl")
    parser.add_argument("--wandb-root", type=Path, default=Path(os.environ.get(
        "WANDB_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch/wandb"
    )))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get(
        "PLOT_OUTPUT_DIR",
        "/home/kyang/gomarl-runtime/gomarl-dual-branch/figures/linear_four_model"
    )))
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if "/" not in args.project:
        parser.error("--project must be ENTITY/PROJECT")

    import plot_wandb_seed_aggregate as plotter

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = build_plans(ROOT)
    wanted = {
        name for plan in plans
        for name in [plan["job_name"]] + plan["historical_candidates"]
    }
    local = local_run_index(args.wandb_root.resolve(), wanted)
    print("Local indexed run names: {}".format(len(local)))
    cloud = {}
    if not args.local_only:
        missing = wanted - set(local)
        if missing:
            cloud = cloud_run_index(args.project, missing)
            print("Cloud fallback run names: {}".format(len(cloud)))

    inventory = []
    entries: dict[str, dict[str, list[Any]]] = {
        scene: {label: [] for label, _ in MODELS} for scene in TITLES
    }
    for plan in plans:
        scene, label, seed = plan["scene"], plan["label"], plan["seed"]
        metric = TITLES[scene][2]
        selected = choose_curve(plan, metric, args.wandb_root.resolve(),
                                local, cloud, plotter)
        row = {
            "scene": scene, "model": label, "seed": seed, "run_name": "",
            "run_id": "", "source": "", "points": 0, "end_step": 0,
            "coverage": "missing",
        }
        if selected is not None:
            x, y, name, run_id, source = selected
            row.update(run_name=name, run_id=run_id, source=source,
                       points=int(x.size), end_step=float(x[-1]),
                       coverage="complete" if x[-1] >= 5000000 else "partial")
            entries[scene][label].append((x, y))
        inventory.append(row)

    inventory_csv = output_dir / "seed_inventory.csv"
    with inventory_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(inventory)
    counts = {status: sum(row["coverage"] == status for row in inventory)
              for status in ("complete", "partial", "missing")}
    print("Seed inventory: {} ({})".format(inventory_csv, counts))

    outputs = {}
    for scene in TITLES:
        output, csv_path = render_scene(scene, entries[scene], output_dir, plotter)
        outputs[scene] = output
        print("Rendered {} and {}".format(output, csv_path))
    if not args.no_upload:
        if counts["complete"] + counts["partial"] == 0:
            raise RuntimeError("No available curves; figures retained locally, nothing uploaded")
        upload_outputs(args.project, outputs, inventory)


if __name__ == "__main__":
    main()
