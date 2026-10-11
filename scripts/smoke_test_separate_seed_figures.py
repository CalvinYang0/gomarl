#!/usr/bin/env python3
"""Restored combined seed panels (legacy test filename); no actual uploads."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import plot_linear_counter_mmm2_threeway_3seeds as charts
import plot_wandb_seed_aggregate as plotter


def main():
    x = np.array([10000., 20000., 30000.])
    labels = {"model_a": ("MODEL A", "#1f77b4"), "model_b": ("MODEL B", "#ff7f0e"),
              "model_c": ("MODEL C", "#2ca02c")}
    entries = {model: [(seed, (x, np.array([seed, seed + 1, seed + 2]) * .1 + offset))
                      for seed in (1, 2, 3) if not (model == "model_b" and seed == 3)]
               for offset, model in ((0, "model_a"), (.2, "model_b"))}
    entries["model_c"] = []
    inspected = set()
    savefig = Figure.savefig
    def inspect(fig, filename, *args, **kwargs):
        name = Path(filename).stem
        if name == "test_scene_individual_seeds":
            assert len(fig.axes) == 3
            for axis, model in zip(fig.axes, labels):
                assert axis.get_title() == labels[model][0]
                assert len(axis.lines) == 2 * len(entries[model])
                for index, (seed, curve) in enumerate(entries[model]):
                    for line in axis.lines[index * 2:index * 2 + 2]:
                        assert np.array_equal(line.get_xdata(), x)
                        assert np.allclose(line.get_ydata(), curve[1] * 100)
                        assert line.get_color() == "C" + str(seed - 1)
                    inspected.add((model, seed))
                legend_labels = axis.get_legend_handles_labels()[1]
                assert len(legend_labels) == len(entries[model])
                if model == "model_b":
                    assert "Missing test history: seed 3" in axis.texts[0].get_text()
                elif model == "model_c":
                    assert "job status not checked" in axis.texts[0].get_text()
        return savefig(fig, filename, *args, **kwargs)
    with tempfile.TemporaryDirectory(prefix="gomarl-combined-seed-qa-") as temp, \
         patch.object(charts, "SCENES", {"test_scene": ("SYNTHETIC QA ONLY", "win")}), \
         patch.object(charts, "LABELS", labels), patch.object(charts, "REPORT_SEED_COVERAGE", True), \
         patch.object(Figure, "savefig", new=inspect):
        directory = Path(temp)
        paths = charts.render("test_scene", entries, directory, plotter, 1)
        assert len(paths) == 3 and len(inspected) == 5
        assert not list(directory.glob("*__seed_*.png"))
        assert len(list(directory.glob("*_individual_seeds.png"))) == 1
        assert all(path.stat().st_size > 1000 for path in paths)
        class Run:
            url = "synthetic://NO_UPLOAD"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def log(self, payload):
                self.payload = payload
            def log_artifact(self, artifact):
                pass
        run = Run()
        artifact = SimpleNamespace(add_file=lambda path: None)
        fake = SimpleNamespace(init=lambda **kwargs: run,
                               util=SimpleNamespace(generate_id=lambda: "synthetic"),
                               Image=lambda path: path, Table=lambda **kwargs: kwargs,
                               Artifact=lambda *args, **kwargs: artifact)
        with patch.dict(sys.modules, wandb=fake):
            charts.upload("synthetic/QA_ONLY", directory, {"test_scene": paths}, [], 1)
        assert "test_scene_three_seed" in run.payload
        assert run.payload["test_scene_individual_seeds"] == str(paths[-1])
        assert len(run.payload) == 3  # inventory + aggregate + model/seed composite
    print("PASS: three seeds per model panel; correct unmixed raw/smoothed curves and seed colors; explicit missing-history placeholders; one composite W&B image key per map")


if __name__ == "__main__":
    main()
