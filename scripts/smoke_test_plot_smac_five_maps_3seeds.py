#!/usr/bin/env python3
"""Test one-shot six-map plotting with synthetic Sacred data; no W&B calls."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_smac_five_maps_3seeds as study


def main():
    plans = study.build_plans(study.charts.ROOT)
    assert len(plans) == 36
    assert {p["scene"] for p in plans} == set(study.SCENES)
    assert not any(p["label"] == "hyper_hypermarl_id" for p in plans)
    # No queue filter: completed-map data is just as eligible as running data.
    marine_names = {}
    for scene, map_name in (("smac_8m", "8m"), ("smac_8m9m", "8m_vs_9m")):
        selected = [p for p in plans if p["scene"] == scene]
        assert len(selected) == 3
        assert {p["map_name"] for p in selected} == {map_name}
        marine_names[scene] = {p["job_name"] for p in selected}
        assert marine_names[scene] == {
            f"{scene}_linear_obs_baseline_10m_s{seed}_valuediag"
            for seed in (1, 2, 3)
        }
    assert marine_names["smac_8m"].isdisjoint(marine_names["smac_8m9m"])
    with tempfile.TemporaryDirectory(prefix="gomarl-six-map-test-") as tmp:
        runtime = Path(tmp)
        for plan in plans:
            # Simulate pending global-state jobs and a missing seed on 3m.
            if plan["label"] == "linear_global_state_baseline":
                continue
            if plan["scene"] == "smac_3m" and plan["seed"] == 3:
                continue
            path = runtime / "results/sacred" / plan["scene"] / plan["label"] / str(plan["seed"])
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps({"wandb_run_name": plan["job_name"]}))
            (path / "run.json").write_text(json.dumps({
                "start_time": "2026-10-09T00:00:00Z", "status": "COMPLETED",
            }))
            # Deliberately different histories catch accidental 8v8/8v9 mixing.
            wins = (0.1, 0.5, 0.8, 0.6 + plan["seed"] * 0.01)
            if plan["scene"] == "smac_8m":
                wins = (0.1, 0.9, 0.98, 0.98)
            elif plan["scene"] == "smac_8m9m":
                wins = (0.1, 0.5, 0.8, 0.8)
            (path / "info.json").write_text(json.dumps({
                "test_battle_won_mean_T": [10000, 500000, 1000000, 2000000],
                "test_battle_won_mean": [
                    {"dtype": "float64", "py/object": "numpy.float64", "value": v}
                    for v in wins
                ],
            }))
        # Default upload called exactly once for all six maps in one analysis run.
        with patch.object(study.charts, "upload") as upload:
            with patch.object(sys, "argv", ["plot", "--local-only",
                                           "--runtime-root", str(runtime)]):
                study.main()
            assert upload.call_count == 1
            assert set(upload.call_args.args[2]) == set(study.SCENES)
        output = runtime / "figures" / study.OUTPUT_SUBDIR
        with (output / "seed_inventory.csv").open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 36
        missing = [row for row in inventory if row["coverage"] == "missing"]
        assert len(missing) == 4
        assert all(row["source"] == "sacred" for row in inventory if row["points"] != "0")
        for scene in study.SCENES:
            with (output / (scene + "_aggregate.csv")).open() as handle:
                aggregate = list(csv.DictReader(handle))
            expected = set(study.LABELS) - {"linear_global_state_baseline"}
            if scene != "smac_5m6m":
                expected = {"linear_baseline"}
            assert {row["model"] for row in aggregate} == expected
            assert {row["seed_count"] for row in aggregate} == ({"2"} if scene == "smac_3m" else {"3"})
            for suffix in ("_three_seed.png", "_three_seed.pdf", "_individual_seeds.png"):
                assert (output / (scene + suffix)).stat().st_size > 1000
        assert study.charts.SCENE_MODELS["smac_3m"] == ("linear_baseline",)
        assert study.charts.SCENE_MODELS["smac_8m9m"] == ("linear_baseline",)
        assert len(study.charts.SCENE_MODELS["smac_5m6m"]) == 7
        with (output / "seed_curves.csv").open() as handle:
            raw_rows = list(csv.DictReader(handle))
        for scene, expected in (("smac_8m", 0.98), ("smac_8m9m", 0.8)):
            terminal = [row for row in raw_rows
                        if row["scene"] == scene and float(row["step"]) == 2000000]
            assert len(terminal) == 3
            assert all(float(row["win_fraction"]) == expected for row in terminal)
        import plot_smac_six_maps_3seeds as explicit_entry
        assert explicit_entry.main is study.main
        # Without upload flag no external write happens.
        with patch.object(study.charts, "upload") as upload:
            with patch.object(sys, "argv", ["plot", "--local-only", "--no-upload",
                                           "--runtime-root", str(runtime)]):
                study.main()
            upload.assert_not_called()
    print("PASS: six maps, 36 exact runs, separate 8v8/8v9 histories, missing seeds/pending state retained, "
          "completed histories eligible, per-map models, PNG/PDF and one upload")


if __name__ == "__main__":
    main()
