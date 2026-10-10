#!/usr/bin/env python3
"""Upload nine three-seed 5m6m groups, including historical VDN/QMIX controls.

Uses exact run names and the latest attempt per seed. Periodic invocations
upload only when curve data, seed inventory or plotting settings change.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import plot_linear_counter_mmm2_threeway_3seeds as charts
from ozstar_submit_six_model_visualization_18jobs import build_plans as visualization_plans
from ozstar_submit_linear_id_5m6m_8m9m_10m_3seeds import build_plans as id_plans
from ozstar_submit_5m6m_id_kl80_3seeds import build_plans as id_kl_plans
from ozstar_submit_5m6m_global_state_3seeds import build_plans as state_plans
from ozstar_submit_5m6m_ones_timestep_10m_3seeds import build_plans as signal_plans
from ozstar_submit_5m6m_obs_entity_id_10m_3seeds import build_plans as entity_id_plans

GROUP = "smac_5m6m_head_condition_comparison_3seeds"
OUTPUT_SUBDIR = "5m6m_head_condition_comparison_3seeds"
LABELS = {
    "linear_baseline": ("Linear obs-based (10M)", "#1f77b4"),
    "linear_id_baseline": ("Linear ID-based (10M)", "#ff7f0e"),
    "linear_bayesg_kl80_keep": ("Linear obs + direct KL80 (5M)", "#9467bd"),
    "linear_global_state_baseline": ("Linear global-state (10M)", "#2ca02c"),
    "linear_ones_baseline": ("Linear all-ones hyper-obs (10M)", "#8c564b"),
    "linear_timestep_baseline": ("Linear episode-timestep hyper-obs (10M)", "#e377c2"),
    "linear_obs_entity_id_baseline": ("Linear obs + absolute entity IDs (10M)", "#17becf"),
    "historical_vdn": ("VDN (historical 5M paper run)", "#d62728"),
    "historical_qmix": ("QMIX (historical 5M paper run)", "#7f7f7f"),
}
_BASE_UPLOAD = charts.upload


def build_plans(repo):
    plans = [plan for plan in visualization_plans(repo) if plan["label"] == "linear_baseline"]
    plans.extend(plan for plan in id_plans(repo) if plan["scene"] == "smac_5m6m")
    plans.extend(plan for plan in id_kl_plans(repo) if plan["label"] == "linear_bayesg_kl80_keep")
    plans.extend(state_plans(repo))
    plans.extend(signal_plans(repo))
    plans.extend(entity_id_plans(repo))
    for method in ("vdn", "qmix"):
        for seed in (1, 2, 3):
            plans.append({
                "scene": "smac_5m6m", "map_name": "5m_vs_6m",
                "label": "historical_" + method, "seed": seed,
                "job_name": "smac_5m6m_paper_{}_5m_s{}".format(method, seed),
                "inventory_note": (
                    "Historical 5M paper control; separate training vintage, "
                    "not a new 10M replica"
                ),
            })
    for plan in plans:
        plan["target_steps"] = (
            5000000 if plan["label"] in {
                "linear_bayesg_kl80_keep",
                "historical_vdn", "historical_qmix",
            } else 10000000
        )
    if len(plans) != 27 or len({p["job_name"] for p in plans}) != 27:
        raise RuntimeError("Expected twenty-seven distinct runs: nine models, three seeds each")
    for label in LABELS:
        if {p["seed"] for p in plans if p["label"] == label} != {1, 2, 3}:
            raise RuntimeError("Incorrect seed inventory: " + label)
    return plans


def upload_if_changed(project, output_dir, outputs, inventory, window):
    signature = hashlib.sha256()
    settings = dict(project=project, group=GROUP, labels=LABELS, window=window,
                    selection="latest attempt per seed; shared seed interval only")
    signature.update(json.dumps(settings, sort_keys=True).encode())
    for filename in ("seed_curves.csv", "seed_inventory.csv"):
        signature.update(filename.encode())
        signature.update((output_dir / filename).read_bytes())
    digest = signature.hexdigest()
    checkpoint = output_dir / "last_uploaded_sha256.txt"
    if checkpoint.is_file() and checkpoint.read_text().strip() == digest:
        print("Three-seed data unchanged; skipping W&B figure upload", flush=True)
        return
    _BASE_UPLOAD(project, output_dir, outputs, inventory, window)
    # Record only successful uploads, so failures retry on the next round.
    pending = checkpoint.with_suffix(".tmp")
    pending.write_text(digest + "\n")
    pending.replace(checkpoint)


def configure():
    charts.GROUP = GROUP
    charts.DEFAULT_OUTPUT_SUBDIR = OUTPUT_SUBDIR
    charts.RESULTS_TITLE = "head conditions / historical 5M VDN and QMIX controls"
    charts.REPORT_SEED_COVERAGE = True
    charts.TARGET_STEPS = 10000000
    charts.SCENES = {"smac_5m6m": ("5 Marines vs. 6 Marines", "test_battle_won_mean")}
    charts.LABELS = LABELS
    charts.build_plans = build_plans
    charts.upload = upload_if_changed


def main():
    configure()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get(
        "RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch")))
    parser.add_argument("--output-dir", type=Path)
    args, _ = parser.parse_known_args()
    output_dir = args.output_dir or args.runtime_root / "figures" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".figure-update.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another 5m6m figure update is running; skipping overlap", flush=True)
            return
        try:
            charts.main()
        except SystemExit as exc:
            if str(exc).startswith("No test-win histories for this batch yet."):
                print("Awaiting 5m6m test-win data; no figures uploaded", flush=True)
            else:
                raise


if __name__ == "__main__":
    main()
