#!/usr/bin/env python3
"""Patch a saved 5m6m snapshot by fetching only six historical 5M controls.

Other curves are reused unchanged from the previous CSV snapshot. No training
jobs, original W&B runs, or other maps are refreshed. There is no full-read
fallback if the snapshot is absent or incompatible.
"""
import argparse
import csv
import os
from pathlib import Path

import plot_5m6m_head_condition_3seeds as study
from plot_linear_directkl_four_model_3seeds import cloud_run_index

OUTPUT_SUBDIR = "5m6m_5m_mixer_patch"
SCENE = "smac_5m6m"


def load_snapshot(directory, plans):
    import numpy as np
    inventory_path = directory / "seed_inventory.csv"
    curves_path = directory / "seed_curves.csv"
    if not inventory_path.is_file() or not curves_path.is_file():
        raise RuntimeError("Previous seed_inventory.csv and seed_curves.csv required in {}. "
                           "No full read started.".format(directory))
    with inventory_path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["scene"] == SCENE]
    by_name = {r["run_name"]: r for r in rows}
    if len(by_name) != len(rows):
        raise RuntimeError("Duplicate run names in saved inventory")
    wanted = {p["job_name"]: p for p in plans}
    retained = {name for name, p in wanted.items() if not p["label"].startswith("historical_")}
    if retained - by_name.keys():
        raise RuntimeError("Snapshot lacks current 5m6m inventory rows: {}. No full read started.".format(
            ", ".join(sorted(retained - by_name.keys()))))
    points = {}
    with curves_path.open() as handle:
        for r in csv.DictReader(handle):
            if r["scene"] == SCENE:
                key = (r["model"], int(r["seed"]), r["run_id"])
                points.setdefault(key, []).append((float(r["step"]), float(r["win_fraction"])))
    curves, inventory = {}, {}
    for name, p in wanted.items():
        if name not in by_name:
            continue
        r = dict(by_name[name])
        if r["model"] != p["label"] or int(r["seed"]) != p["seed"]:
            raise RuntimeError("Snapshot cohort mismatch: " + name)
        for field in ("seed", "points"):
            r[field] = int(r[field])
        for field in ("start_step", "end_step"):
            r[field] = float(r[field]) if r.get(field) not in (None, "") else None
        values = points.get((p["label"], p["seed"], r["run_id"]), [])
        if len(values) != r["points"]:
            raise RuntimeError("Snapshot point count mismatch: " + name)
        if values:
            data = np.asarray(values, dtype=float)
            if (not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0)
                    or np.any(data[:, 0] < 0) or np.any(data[:, 0] > 10050000)
                    or np.any(data[:, 1] < 0) or np.any(data[:, 1] > 1)
                    or data[0, 0] != r["start_step"] or data[-1, 0] != r["end_step"]):
                raise RuntimeError("Invalid saved curve: " + name)
            curves[name] = (data[:, 0], data[:, 1])
        inventory[name] = r
    return curves, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="hjh331-sjtu/gomarl")
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch")))
    parser.add_argument("--snapshot-dir", type=Path,
                        help="Previous figure directory containing both seed CSVs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mean-window", type=int, default=100)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if args.mean_window < 1 or "/" not in args.project:
        parser.error("positive mean window and ENTITY/PROJECT required")
    snapshot = args.snapshot_dir or args.runtime_root / "figures/smac_seven_maps_latest_3seeds"
    output = args.output_dir or args.runtime_root / "figures" / OUTPUT_SUBDIR
    if snapshot.resolve() == output.resolve():
        parser.error("Use a separate output directory to preserve the source snapshot")
    study.configure()
    plans = study.build_plans(study.charts.ROOT)
    targets = [p for p in plans if p["label"] in {"historical_vdn", "historical_qmix"}]
    curves, inventory = load_snapshot(snapshot, plans)
    wanted = {p["job_name"] for p in targets}
    print("Reuse saved 5m6m data: {}; query ONLY six 5M VDN/QMIX runs; no other histories refreshed".format(
        snapshot), flush=True)
    # Exact filter only: unlike full snapshots, never scan the entire project.
    cloud = cloud_run_index(args.project, wanted, allow_full_scan=False)
    import plot_wandb_seed_aggregate as plotter
    for index, p in enumerate(targets, 1):
        name = p["job_name"]
        print("READ {}/6: {}".format(index, name), flush=True)
        curve, details, note = study.charts.select_latest(
            p, "test_battle_won_mean", {}, cloud, {}, args.runtime_root / "wandb", plotter)
        if (curve is None and name in curves
                and (not details.get("run_id") or details["run_id"] == inventory[name]["run_id"])):
            print("RETAIN previously saved control; fresh read unavailable: " + note, flush=True)
            continue
        r = dict.fromkeys(study.charts.FIELDS, "")
        r.update(scene=SCENE, model=p["label"], seed=p["seed"], run_name=name,
                 points=0, start_step=None, end_step=None, coverage="missing",
                 note="; ".join(filter(None, (p["inventory_note"], note))), **details)
        if curve is not None:
            x, y = curve
            curves[name] = curve
            r.update(points=int(x.size), start_step=float(x[0]), end_step=float(x[-1]),
                     coverage="complete" if x[-1] >= p["target_steps"] else "partial")
        else:
            # A genuinely newer attempt with no points must not inherit the
            # cached older run's curve under the newer run ID.
            curves.pop(name, None)
        inventory[name] = r
        print("{}: {} points={} end={}".format(name, r["coverage"], r["points"], r["end_step"]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    ordered_inventory = [inventory[p["job_name"]] for p in plans]
    entries = {model: [] for model in study.LABELS}
    with (output / "seed_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "model", "seed", "run_id", "step", "win_fraction"])
        for p in plans:
            name = p["job_name"]
            if name in curves:
                x, y = curves[name]
                entries[p["label"]].append((p["seed"], (x, y)))
                writer.writerows((SCENE, p["label"], p["seed"], inventory[name]["run_id"], step, value)
                                 for step, value in zip(x, y))
    with (output / "seed_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=study.charts.FIELDS)
        writer.writeheader()
        writer.writerows(ordered_inventory)
    with (output / "source_snapshot.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snapshot_directory", "snapshot_inventory_mtime", "refresh_scope"])
        writer.writerow([str(snapshot), (snapshot / "seed_inventory.csv").stat().st_mtime,
                         "Only six historical 5M VDN/QMIX controls; all other cohorts reused unchanged"])
    study.charts.RESULTS_TITLE = "5M mixer patch; other cohorts reused from saved snapshot"
    paths = study.charts.render(SCENE, entries, output, plotter, args.mean_window)
    print("Patched 5m6m figures: " + str(output), flush=True)
    if not args.no_upload:
        study.upload_if_changed(args.project, output, {SCENE: paths}, ordered_inventory, args.mean_window)


if __name__ == "__main__":
    main()
