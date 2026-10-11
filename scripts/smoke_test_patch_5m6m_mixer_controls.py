#!/usr/bin/env python3
"""Synthetic checks: six exact cloud reads, unchanged cached groups, no full scan."""
import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import plot_5m6m_add_5m_mixer_controls as app


def main():
    app.study.configure()
    charts = app.study.charts
    plans = app.study.build_plans(charts.ROOT)
    targets = {p["job_name"] for p in plans if p["label"].startswith("historical_")}
    assert len(targets) == 6
    with tempfile.TemporaryDirectory(prefix="gomarl-mixer-patch-") as tmp:
        root = Path(tmp)
        snapshot, output = root / "saved", root / "patched"
        snapshot.mkdir()
        rows, raw = [], []
        for p in plans:
            if p["job_name"] in targets:
                continue
            r = dict.fromkeys(charts.FIELDS, "")
            missing = p["label"] in {"vdn", "qmix"}
            r.update(scene=app.SCENE, model=p["label"], seed=p["seed"], run_name=p["job_name"],
                     run_id="saved_" + p["job_name"], source="cloud", points=0 if missing else 3,
                     start_step=None if missing else 10000, end_step=None if missing else 5000000,
                     coverage="missing" if missing else "partial", note="cached cohort")
            rows.append(r)
            if not missing:
                raw.extend((app.SCENE, p["label"], p["seed"], r["run_id"], x, y)
                           for x, y in ((10000, .1), (1000000, .6), (5000000, .7)))
        for name, fields, data in (
            ("seed_inventory.csv", charts.FIELDS, rows),
            ("seed_curves.csv", ("scene", "model", "seed", "run_id", "step", "win_fraction"), raw),
        ):
            with (snapshot / name).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(([r[k] for k in fields] for r in data) if name.startswith("seed_inventory") else data)
        source_bytes = [(snapshot / name).read_bytes() for name in ("seed_inventory.csv", "seed_curves.csv")]
        runs = {}
        for name in targets:
            run = SimpleNamespace(id="fresh_" + name, created_at="2026-10-01T00:00:00Z")
            run.scan_history = lambda **kwargs: iter([
                {"_step": 10000, "test_battle_won_mean": .1},
                {"_step": 1000000, "test_battle_won_mean": .5},
                {"_step": 5000000, "test_battle_won_mean": .75},
            ])
            runs[name] = [run]
        argv = ["patch", "--snapshot-dir", str(snapshot), "--output-dir", str(output)]
        with patch.object(app, "cloud_run_index", return_value=runs) as query, \
                patch.object(app.study, "_BASE_UPLOAD") as upload, \
                patch.object(sys, "argv", argv):
            app.main()
            assert query.call_args.args[1] == targets
            assert query.call_args.kwargs == {"allow_full_scan": False}
            assert upload.call_count == 1 and set(upload.call_args.args[2]) == {app.SCENE}
        with (output / "seed_inventory.csv").open() as handle:
            new_rows = list(csv.DictReader(handle))
        with (snapshot / "seed_inventory.csv").open() as handle:
            before = {r["run_name"]: r for r in csv.DictReader(handle)}
        assert len(new_rows) == 36
        def normalized(r):
            return {k: float(v) if k in {"seed", "points", "start_step", "end_step"} and v != ""
                    else v for k, v in r.items()}
        assert all(normalized(r) == normalized(before[r["run_name"]])
                   for r in new_rows if r["run_name"] not in targets)
        with (output / "seed_curves.csv").open() as handle:
            new_raw = list(csv.reader(handle))[1:]
        def curve_values(r):
            return tuple(r[:4]) + (float(r[4]), float(r[5]))
        assert [curve_values(r) for r in new_raw if not r[1].startswith("historical_")] == [
            curve_values([str(v) for v in r]) for r in raw]
        assert all((output / (app.SCENE + suffix)).stat().st_size > 1000
                   for suffix in ("_three_seed.png", "_three_seed.pdf", "_individual_seeds.png"))
        assert source_bytes == [(snapshot / name).read_bytes() for name in ("seed_inventory.csv", "seed_curves.csv")]
        # Fix one missing entity-ID seed, not all six controls or all maps.
        one_name = "smac_5m6m_linear_obs_entity_id_10m_s1_entityidcond"
        with (output / "seed_inventory.csv").open() as handle:
            one_inventory = list(csv.DictReader(handle))
        for r in one_inventory:
            if r["run_name"] == one_name:
                r.update(points="0", start_step="", end_step="", coverage="missing", run_id="", note="old miss")
        with (output / "seed_inventory.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=charts.FIELDS)
            writer.writeheader()
            writer.writerows(one_inventory)
        with (output / "seed_curves.csv").open() as handle:
            one_raw = list(csv.DictReader(handle))
        with (output / "seed_curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=one_raw[0].keys())
            writer.writeheader()
            writer.writerows(r for r in one_raw if not (
                r["model"] == "linear_obs_entity_id_baseline" and r["seed"] == "1"))
        one_run = SimpleNamespace(id="recovered_entity_seed1", created_at="2026-10-01T00:00:00Z",
                                  scan_history=runs[next(iter(targets))][0].scan_history)
        one_output = root / "one_patch"
        with patch.object(app, "cloud_run_index", return_value={one_name: [one_run]}) as query, \
                patch.object(sys, "argv", ["patch", "--snapshot-dir", str(output),
                                          "--output-dir", str(one_output), "--refresh-run", one_name,
                                          "--no-upload"]):
            app.main()
            assert query.call_args.args[1] == {one_name}
        with (one_output / "seed_inventory.csv").open() as handle:
            recovered = {r["run_name"]: r for r in csv.DictReader(handle)}
        assert recovered[one_name]["points"] == "3"
        assert recovered[one_name]["run_id"] == "recovered_entity_seed1"
        assert all(normalized(r) == normalized(recovered[r["run_name"]])
                   for r in one_inventory if r["run_name"] != one_name)
        with (one_output / (app.SCENE + "_aggregate.csv")).open() as handle:
            entity_rows = [r for r in csv.DictReader(handle)
                           if r["model"] == "linear_obs_entity_id_baseline"]
        assert entity_rows and all(r["seed_count"] == "3" for r in entity_rows)
        newer_name = "smac_5m6m_paper_vdn_5m_s1"
        newer = SimpleNamespace(id="new_empty_attempt", created_at="2026-10-11T00:00:00Z",
                                scan_history=lambda **kwargs: iter([]))
        second_output = root / "newer_attempt"
        with patch.object(app, "cloud_run_index", return_value={newer_name: [newer]}), \
                patch.object(sys, "argv", ["patch", "--snapshot-dir", str(output),
                                          "--output-dir", str(second_output), "--no-upload"]), \
                patch.object(app.study, "_BASE_UPLOAD") as upload:
            app.main()
            upload.assert_not_called()
        with (second_output / "seed_inventory.csv").open() as handle:
            retained = {r["run_name"]: r for r in csv.DictReader(handle)}
        assert retained[newer_name]["coverage"] == "missing"
        assert retained[newer_name]["run_id"] == "new_empty_attempt"
        assert all(retained[name]["coverage"] == "complete" for name in targets - {newer_name})
        with (second_output / "seed_curves.csv").open() as handle:
            assert not any(r["run_id"] in {"new_empty_attempt", "fresh_" + newer_name}
                           for r in csv.DictReader(handle))
        # Missing snapshot must fail before any network request.
        with patch.object(app, "cloud_run_index") as query, patch.object(sys, "argv", [
                "patch", "--snapshot-dir", str(root / "absent"), "--no-upload"]):
            try:
                app.main()
                raise AssertionError("Missing cache was accepted")
            except RuntimeError as exc:
                assert "No full read started" in str(exc)
            query.assert_not_called()
        # Exact cloud query never enumerates unrelated runs, including on error.
        fake_api = SimpleNamespace(runs=lambda *args, **kwargs: [])
        fake_wandb = SimpleNamespace(Api=lambda **kwargs: fake_api)
        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.object(fake_api, "runs", return_value=[]) as calls:
            assert app.cloud_run_index("entity/project", targets, allow_full_scan=False) == {}
            assert calls.call_count == 1
            assert calls.call_args.kwargs["filters"]["display_name"]["$in"] == sorted(targets)
        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.object(fake_api, "runs", side_effect=OSError("network")) as calls:
            try:
                app.cloud_run_index("entity/project", targets, allow_full_scan=False)
                raise AssertionError("Strict query swallowed error")
            except RuntimeError:
                pass
            assert calls.call_count == 1
    print("PASS: six default controls or one exact missing entity-ID seed; retained curves and source snapshot unchanged; "
          "5m6m-only render/upload; absent snapshot and cloud failures never trigger full scans")


if __name__ == "__main__":
    main()
