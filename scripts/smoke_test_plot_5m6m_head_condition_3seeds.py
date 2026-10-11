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
    assert len(plans) == 36
    assert len(study.LABELS) == 12
    assert {p["label"] for p in plans} == set(study.LABELS)
    health = [p for p in plans if p["label"] == "linear_health_baseline"]
    assert len(health) == 3 and all("healthcond" in p["job_name"] for p in health)
    fresh_ids = [p for p in plans if p["label"] == "linear_id_baseline_vizcoverage"]
    assert len(fresh_ids) == 3 and all("vizcoverage" in p["job_name"] for p in fresh_ids)
    assert all(p["exports"]["MODEL_TYPE"] == "smac_single_linear_suite_id_baseline_hypercond"
               for p in fresh_ids)
    excluded_templates = (
        "smac_5m6m_linear_id_baseline_10m_s{}_valuediag",
        "smac_5m6m_linear_id_5m_s{}_idkl80fix",
        "smac_5m6m_linear_obs_baseline_10m_s{}_valuediag",
        "smac_5m6m_id_baseline_10m_s{}_valuediag",
    )
    excluded_names = {template.format(seed) for template in excluded_templates
                      for seed in (1, 2, 3)}
    assert excluded_names.isdisjoint(p["job_name"] for p in plans)
    for method in ("vdn", "qmix"):
        new_controls = [p for p in plans if p["label"] == method]
        assert {p["job_name"] for p in new_controls} == {
            "smac_5m6m_{}_baseline_10m_s{}_valuediag".format(method, seed)
            for seed in (1, 2, 3)
        }
        assert all(p["target_steps"] == 10000000 for p in new_controls)
        old_controls = [p for p in plans if p["label"] == "historical_" + method]
        assert {p["job_name"] for p in old_controls} == {
            "smac_5m6m_paper_{}_5m_s{}".format(method, seed) for seed in (1, 2, 3)
        }
        assert all(p["target_steps"] == 5000000 for p in old_controls)
    assert sum("entityidcond" in p["job_name"] for p in plans) == 3
    assert sum("statecond" in p["job_name"] for p in plans) == 3
    assert sum("idkl80fix" in p["job_name"] for p in plans) == 3
    assert sum("_recheck" in p["job_name"] for p in plans) == 3
    assert sum("signalcond" in p["job_name"] for p in plans) == 6
    for plan in plans:
        assert plan["target_steps"] == (5000000 if plan["label"] in {
            "linear_bayesg_kl80_keep", "historical_vdn", "historical_qmix"}
                                         else 10000000)
    with tempfile.TemporaryDirectory(prefix="gomarl-head-figures-") as tmp:
        runtime = Path(tmp)
        empty_runtime = runtime / "empty"
        with patch.object(sys, "argv", ["plot", "--local-only", "--no-upload",
                                         "--runtime-root", str(empty_runtime)]):
            study.main()
        assert not list(empty_runtime.rglob("*.png"))
        # Even if complete obsolete histories exist, do not read or fall back to them.
        for index, name in enumerate(sorted(excluded_names)):
            path = runtime / "results/sacred/5m_vs_6m/excluded" / str(index)
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps({"wandb_run_name": name}))
            (path / "run.json").write_text(json.dumps({"start_time": "2026-10-08T00:00:00Z"}))
            (path / "info.json").write_text(json.dumps({
                "test_battle_won_mean_T": [10000, 10000000],
                "test_battle_won_mean": [1.0, 1.0],
            }))
        for plan in plans:
            # VDN has no 10M data: use its separate 5M cohort. QMIX has one
            # 10M seed: do NOT fill its other seeds from historical QMIX.
            if plan["label"] in {"linear_id_baseline_vizcoverage", "vdn"}:
                continue
            if plan["label"] == "qmix" and plan["seed"] != 1:
                continue
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
        assert len(inventory) == 36
        assert excluded_names.isdisjoint(row["run_name"] for row in inventory)
        assert all(row["source"] == "sacred" and row["points"] == "4"
                   for row in inventory if row["coverage"] != "missing")
        missing_ids = [row for row in inventory if row["model"] == "linear_id_baseline_vizcoverage"]
        assert len(missing_ids) == 3 and all(row["coverage"] == "missing" for row in missing_ids)
        with (output / "smac_5m6m_aggregate.csv").open() as handle:
            aggregate = list(csv.DictReader(handle))
        assert {row["model"] for row in aggregate} == set(study.LABELS) - {
            "linear_id_baseline_vizcoverage", "vdn", "historical_qmix"}
        assert all(row["seed_count"] == ("1" if row["model"] == "qmix" else "3")
                   for row in aggregate)
        assert all("Historical 5M" in row["note"] for row in inventory
                   if row["model"].startswith("historical_"))
        for name in ("smac_5m6m_three_seed.png", "smac_5m6m_three_seed.pdf"):
            assert (output / name).stat().st_size > 1000
        assert (output / "smac_5m6m_individual_seeds.png").stat().st_size > 1000
        assert not list(output.glob("*__seed_*.png"))
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
    print("PASS: ten current cohorts plus two labelled 5M mixer fallbacks, 5M VDN visible when 10M absent, one 10M QMIX seed never filled from 5M, obsolete IDs excluded, exact seeds, Sacred NumPy scalars, "
          "PNG/PDF, upload dedup and failure retry")


if __name__ == "__main__":
    main()
