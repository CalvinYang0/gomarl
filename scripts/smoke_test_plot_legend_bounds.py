#!/usr/bin/env python3
"""Synthetic layout QA: full model legends fit the actual PNG/PDF canvas."""
import csv
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

import plot_5m6m_8m9m_qmix_variants_3seeds as study


def main():
    study.configure()
    c = study.charts
    with tempfile.TemporaryDirectory(prefix='synthetic-legend-bounds-') as temp:
        os.environ.setdefault('MPLCONFIGDIR', str(Path(temp) / 'matplotlib'))
        from matplotlib.figure import Figure
        import plot_wandb_seed_aggregate as plotter
        save = Figure.savefig
        cases = []
        for count in (4, 10):
            models = study.SCENE_MODELS['smac_5m6m'][:count]
            x = np.arange(10000, 10000001, 10000)
            entries = {model: [(seed, (x, np.linspace(.05, .8, len(x)) + seed * .01))
                               for seed in (1, 2, 3)] for model in models}
            checked = []

            def inspect(fig, path, **kwargs):
                if str(path).endswith('_three_seed.png'):
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    legends = [axis.get_legend() for axis in fig.axes if axis.get_legend()]
                    assert len(legends) == 1
                    legend = legends[0]
                    assert len(legend.get_texts()) == count
                    for artist in [legend] + list(legend.get_texts()) + list(legend.get_lines()):
                        box = artist.get_window_extent(renderer)
                        assert box.x0 >= 0 and box.y0 >= 0, ('left/bottom clipped', box)
                        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, ('right/top clipped', box)
                    for model, text in zip(models, legend.get_texts()):
                        normalized = ' '.join(text.get_text().split())
                        assert c.LABELS[model][0] in normalized
                        assert '(3/3 seeds; to 10.00M)' in normalized
                    assert kwargs['bbox_inches'] == 'tight'
                    # Layout changes do not add/remove/modify plotted data lines.
                    assert len(fig.axes[0].lines) == count
                    assert all(np.isfinite(line.get_ydata()).all() for line in fig.axes[0].lines)
                    checked.append(True)
                return save(fig, path, **kwargs)

            output = Path(temp) / str(count)
            output.mkdir()
            with patch.object(Figure, 'savefig', inspect):
                study._RENDER('smac_5m6m', entries, output, plotter, 10)
            assert checked == [True]
            with (output / 'smac_5m6m_aggregate.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            assert {row['model'] for row in rows} == set(models)
            first = [row for row in rows if row['model'] == models[0]]
            expected_mean = plotter.centered_rolling_mean(np.linspace(.05, .8, len(x)) + .02, 10)
            np.testing.assert_allclose([float(row['mean_fraction_smoothed']) for row in first], expected_mean)
            np.testing.assert_allclose([float(row['sample_std_smoothed']) for row in first], .01)
            cases.append(output)
        if os.environ.get('LEGEND_QA_OUTPUT'):
            import shutil
            shutil.copytree(cases[-1], os.environ['LEGEND_QA_OUTPUT'], dirs_exist_ok=True)
    print('PASS: four/ten full legends fit; PNG/PDF exported; colors/coverage preserved; mean/std unchanged')


if __name__ == '__main__':
    main()
