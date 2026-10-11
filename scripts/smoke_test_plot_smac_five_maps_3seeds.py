#!/usr/bin/env python3
"""Test one-shot seven-map plotting with synthetic Sacred data; no W&B calls."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import plot_smac_five_maps_3seeds as study


def main():
    plans = study.build_plans(study.charts.ROOT)
    assert len(plans) == 75
    assert {p["scene"] for p in plans} == set(study.SCENES)
    assert not any(p["label"] in {
        "hyper_hypermarl_id", "linear_id_baseline_5m", "original_linear_baseline",
        "historical_vdn", "historical_qmix",
    } for p in plans)
    assert sum(len(models) for models in study.SCENE_MODELS.values()) == 25
    # No queue filter: completed-map data is just as eligible as running data.
    marine_names = {}
    for scene, map_name in (("smac_8m", "8m"), ("smac_8m9m", "8m_vs_9m")):
        selected = [p for p in plans if p["scene"] == scene]
        assert len(selected) == (3 if scene == "smac_8m" else 15)
        assert {p["map_name"] for p in selected} == {map_name}
        marine_names[scene] = {p["job_name"] for p in selected if p["label"] == "linear_baseline"}
        assert marine_names[scene] == {
            f"{scene}_linear_obs_baseline_10m_s{seed}_valuediag"
            for seed in (1, 2, 3)
        }
    assert marine_names["smac_8m"].isdisjoint(marine_names["smac_8m9m"])
    with tempfile.TemporaryDirectory(prefix="gomarl-seven-map-test-") as tmp:
        runtime = Path(tmp)
        # Old 3s5z Obs and legacy ID records must not replace new missing histories.
        for scene in ("smac_5m6m", "smac_8m9m", "smac_3svs5z", "smac_6h8z"):
            for seed in (1, 2, 3):
                path = runtime / "results/sacred" / scene / "legacy" / str(seed)
                path.mkdir(parents=True)
                (path / "config.json").write_text(json.dumps({
                    "wandb_run_name": f"{scene}_id_baseline_10m_s{seed}_valuediag",
                }))
                (path / "run.json").write_text(json.dumps({"start_time": "2026-10-08T00:00:00Z"}))
                (path / "info.json").write_text(json.dumps({
                    "test_battle_won_mean_T": [10000, 10000000],
                    "test_battle_won_mean": [1.0, 1.0],
                }))
        old_obs = runtime / "results/sacred/smac_3svs5z/old_obs/3"
        old_obs.mkdir(parents=True)
        (old_obs / "config.json").write_text(json.dumps({
            "wandb_run_name": "smac_3svs5z_linear_obs_baseline_10m_s3_valuediag",
        }))
        (old_obs / "run.json").write_text(json.dumps({"start_time": "2026-10-08T00:00:00Z"}))
        (old_obs / "info.json").write_text(json.dumps({
            "test_battle_won_mean_T": [10000, 10000000], "test_battle_won_mean": [1.0, 1.0],
        }))
        for plan in plans:
            # Simulate pending global-state jobs and a missing seed on 3m.
            if plan["label"] == "linear_global_state_baseline":
                continue
            if plan["scene"] == "smac_3m" and plan["seed"] == 3:
                continue
            if (plan["scene"], plan["label"], plan["seed"]) == (
                "smac_3svs5z", "linear_baseline_vizcoverage", 3,
            ):
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
        # Default upload called exactly once for all seven maps in one analysis run.
        with patch.object(study.charts, "upload") as upload:
            with patch.object(sys, "argv", ["plot", "--local-only",
                                           "--runtime-root", str(runtime)]):
                study.main()
            assert upload.call_count == 1
            assert set(upload.call_args.args[2]) == set(study.SCENES)
            uploaded_inventory = upload.call_args.args[3]
            assert {row["run_name"] for row in uploaded_inventory} == {p["job_name"] for p in plans}
        output = runtime / "figures" / study.OUTPUT_SUBDIR
        with (output / "seed_inventory.csv").open() as handle:
            inventory = list(csv.DictReader(handle))
        assert len(inventory) == 75
        missing = [row for row in inventory if row["coverage"] == "missing"]
        assert len(missing) == 5
        missing_new_obs = [row for row in missing if row["scene"] == "smac_3svs5z"]
        assert len(missing_new_obs) == 1 and missing_new_obs[0]["seed"] == "3"
        assert all(row["source"] == "sacred" for row in inventory if row["points"] != "0")
        for scene in study.SCENES:
            with (output / (scene + "_aggregate.csv")).open() as handle:
                aggregate = list(csv.DictReader(handle))
            expected = set(study.SCENE_MODELS[scene]) - {"linear_global_state_baseline"}
            assert {row["model"] for row in aggregate} == expected
            counts = {"2"} if scene == "smac_3m" else {"2", "3"} if scene == "smac_3svs5z" else {"3"}
            assert {row["seed_count"] for row in aggregate} == counts
            for suffix in ("_three_seed.png", "_three_seed.pdf"):
                assert (output / (scene + suffix)).stat().st_size > 1000
            for model in study.SCENE_MODELS[scene]:
                for seed in (1, 2, 3):
                    assert (output / "{}__{}__seed_{}.png".format(scene, model, seed)).stat().st_size > 1000
        assert len(list(output.glob("*__seed_*.png"))) == 75
        assert not list(output.glob("*_individual_seeds.png"))
        assert study.charts.SCENE_MODELS["smac_3m"] == ("linear_baseline",)
        assert set(study.charts.SCENE_MODELS["smac_8m9m"]) == {
            "linear_baseline", "linear_id_baseline", "vdn", "qmix",
            "linear_ones_baseline_vizcoverage"}
        assert set(study.charts.SCENE_MODELS["smac_6h8z"]) == {
            "linear_baseline", "linear_id_baseline"}
        assert len(study.charts.SCENE_MODELS["smac_5m6m"]) == 10
        assert set(study.charts.SCENE_MODELS["smac_3svs5z"]) == set(study.PLOT_LABELS)
        assert set(study.charts.SCENE_MODELS["smac_mmm2"]) == set(study.PLOT_LABELS)
        fresh = [row for row in inventory if row["model"].endswith("_vizcoverage")]
        assert len(fresh) == 24 and all("vizcoverage" in row["run_name"] for row in fresh)
        assert "linear_id_baseline_vizcoverage" in study.charts.SCENE_MODELS["smac_5m6m"]
        assert "linear_id_baseline" not in study.charts.SCENE_MODELS["smac_5m6m"]
        with (output / "seed_curves.csv").open() as handle:
            raw_rows = list(csv.DictReader(handle))
        for scene, expected in (("smac_8m", 0.98), ("smac_8m9m", 0.8)):
            terminal = [row for row in raw_rows
                        if row["scene"] == scene and float(row["step"]) == 2000000]
            assert len(terminal) == len(study.SCENE_MODELS[scene]) * 3
            assert all(float(row["win_fraction"]) == expected for row in terminal)
        import plot_smac_six_maps_3seeds as explicit_entry
        assert explicit_entry.main is study.main
        import plot_smac_seven_maps_3seeds as new_entry
        assert new_entry.main is study.main
        # Without upload flag no external write happens.
        with patch.object(study.charts, "upload") as upload:
            with patch.object(sys, "argv", ["plot", "--local-only", "--no-upload",
                                           "--runtime-root", str(runtime)]):
                study.main()
            upload.assert_not_called()
    print("PASS: seven maps, 25 selected cohorts / 75 exact runs including health-only and 24 fresh replicas, no old-reference fallback, separate 8v8/8v9 histories, missing seeds retained, "
          "completed histories eligible, per-map models, PNG/PDF and one upload")


if __name__ == "__main__":
    main()
