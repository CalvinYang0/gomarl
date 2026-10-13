#!/usr/bin/env python3
"""Synthetic figure QA; no experimental curves or W&B requests."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_5m6m_mixer_input_controls_3seeds as study


def main():
    study.configure()
    c = study.charts
    with tempfile.TemporaryDirectory(prefix="gomarl-mixer-figure-smoke-") as temp:
        runtime = Path(temp)
        for p in c.build_plans(c.ROOT):
            if p["label"] in {"qmix", "vdn"}:
                continue
            path = runtime / "results/sacred/5m_vs_6m" / p["label"] / str(p["seed"])
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps({"wandb_run_name": p["job_name"]}))
            (path / "run.json").write_text(json.dumps({"start_time": "2026-10-13T00:00:00Z"}))
            end = 5000000 if p["label"].startswith("historical_") else (8000000 if p["seed"] == 3 else 10000000)
            (path / "info.json").write_text(json.dumps({
                "test_battle_won_mean_T": [10000, 1000000, 3000000, end],
                "test_battle_won_mean": [.1, .4, .6, .7],
            }))
        with patch.object(sys, "argv", ["plot", "--runtime-root", str(runtime),
                                         "--local-only", "--no-upload", "--mean-window", "10"]):
            c.main()
        output = runtime / "figures" / c.DEFAULT_OUTPUT_SUBDIR
        for name in ("smac_5m6m_three_seed.png", "smac_5m6m_three_seed.pdf", "smac_5m6m_individual_seeds.png"):
            assert (output / name).stat().st_size > 1000
        with (output / "seed_inventory.csv").open() as f:
            inventory = list(csv.DictReader(f))
        assert len(inventory) == 21
        with (output / "smac_5m6m_aggregate.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert {r["model"] for r in rows} == set(study.PLOT_LABELS) | {"historical_qmix", "historical_vdn"}
        assert all(r["seed_count"] == "3" for r in rows)
        assert all(float(r["step"]) <= (5000000 if r["model"].startswith("historical_") else 8000000) for r in rows)
    print("PASS: synthetic PNG/PDF; all nine new seeds; explicit whole-cohort 5M QMIX fallback; shared interval; no cloud/uploads")


if __name__ == "__main__":
    main()
