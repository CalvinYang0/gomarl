#!/usr/bin/env python3
"""Verify exact seeds, Sacred decoding, figures and retryable upload dedup."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_5m6m_head_condition_3seeds as study


def main():
    plans = study.build_plans(study.charts.ROOT)
    assert len(plans) == 27
    for method in ("vdn", "qmix"):
        controls = [p for p in plans if p["label"] == "historical_" + method]
        assert {p["job_name"] for p in controls} == {
            "smac_5m6m_paper_{}_5m_s{}".format(method, seed) for seed in (1, 2, 3)
        }
        assert all(p["target_steps"] == 5000000 for p in controls)
        assert all("Historical 5M" in p["inventory_note"] for p in controls)
        assert "historical 5M" in study.LABELS["historical_" + method][0]
    assert sum("entityidcond" in p["job_name"] for p in plans) == 3
    assert not any(p["label"] == "hyper_hypermarl_id" for p in plans)
    assert sum("statecond" in p["job_name"] for p in plans) == 3
    assert sum("idkl80fix" in p["job_name"] for p in plans) == 3
    assert sum("_recheck" in p["job_name"] for p in plans) == 3
    assert {p["job_name"] for p in plans if p["label"] == "linear_id_baseline"} == {
        "smac_5m6m_linear_id_baseline_10m_s{}_valuediag".format(seed) for seed in (1, 2, 3)
    }
    assert sum("signalcond" in p["job_name"] for p in plans) == 6
    for plan in plans:
        assert plan["target_steps"] in (5000000, 10000000)
        if plan["label"] == "linear_id_baseline":
            assert plan["target_steps"] == 10000000
    with tempfile.TemporaryDirectory(prefix="gomarl-head-figures-") as tmp:
        runtime = Path(tmp)
        empty_runtime = runtime / "empty"
        with patch.object(sys, "argv", ["plot", "--local-only", "--no-upload",
                                         "--runtime-root", str(empty_runtime)]):
            study.main()
        assert not list(empty_runtime.rglob("*.png"))
        for plan in plans:
            path = runtime / "results/sacred/5m_vs_6m" / plan["label"] / str(plan["seed"])
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps({"wandb_run_name": plan["job_name"]}))
            (path / "run.json").write_text(json.dumps({"start_time": "2026-10-09T00:00:00Z"}))
            (path / "info.json").write_text(json.dumps({
                "test_battle_won_mean_T": [10000, 500000, 1000000, 2000000],
                "test_battle_won_mean": [
                    {"dtype": "float64", "py/object": "numpy.float64", "value": v}
                    for v in (0.1, 0.5, 0.8, 0.6 + plan["seed"] * 0.01)
                ],
            }))
        with patch.object(sys, "argv", ["plot", "--local-only", "--no-upload",
                                         "--runtime-root", str(runtime)]):
            study.main()
        output = runtime / "figures" / study.OUTPUT_SUBDIR
        with (output / "seed_inventory.csv").open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 27
        assert all("Historical 5M" in row["note"] for row in inventory
                   if row["model"] in {"historical_vdn", "historical_qmix"})
        assert all(row["source"] == "sacred" and row["points"] == "4" for row in inventory)
        with (output / "smac_5m6m_aggregate.csv").open() as handle:
            aggregate = list(csv.DictReader(handle))
        assert {row["model"] for row in aggregate} == set(study.LABELS)
        assert all(row["seed_count"] == "3" for row in aggregate)
        for name in ("smac_5m6m_three_seed.png", "smac_5m6m_three_seed.pdf",
                     "smac_5m6m_individual_seeds.png"):
            assert (output / name).stat().st_size > 1000
        with patch.object(study, "_BASE_UPLOAD") as upload:
            args = ("hjh331-sjtu/gomarl", output, {}, inventory, 100)
            study.upload_if_changed(*args)
            study.upload_if_changed(*args)
            assert upload.call_count == 1
            checkpoint = output / "last_uploaded_sha256.txt"
            before = checkpoint.read_text()
            with (output / "seed_curves.csv").open("a") as handle:
                handle.write("new data\n")
            upload.side_effect = RuntimeError("mock upload failure")
            try:
                study.upload_if_changed(*args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Upload failure must propagate for retry")
            assert checkpoint.read_text() == before
            upload.side_effect = None
            study.upload_if_changed(*args)
            assert checkpoint.read_text() != before
    print("PASS: nine groups with separately labelled historical 5M VDN/QMIX, exact seeds, Sacred NumPy scalars, "
          "PNG/PDF, upload dedup and failure retry")


if __name__ == "__main__":
    main()
