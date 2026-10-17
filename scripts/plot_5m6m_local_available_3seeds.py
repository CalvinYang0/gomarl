#!/usr/bin/env python3
"""Upload available SMAC QMIX/VDN curves from local logs and retained CSVs only."""
import argparse
import os
from pathlib import Path
import sys

import plot_5m6m_8m9m_qmix_variants_3seeds as study


def configure(scene='smac_5m6m'):
    study.configure()
    c = study.charts
    # Materialize before narrowing the wrapper's two-map manifest validation.
    if scene not in study.SCENE_MODELS:
        raise ValueError('Unsupported scene: ' + scene)
    plans = [p for p in c.build_plans(c.ROOT) if p['scene'] == scene]
    c.build_plans = lambda root: plans
    c.SCENES = {scene: c.SCENES[scene]}
    c.SCENE_MODELS = {scene: study.SCENE_MODELS[scene]}
    c.GROUP = scene + '_local_available_qmix_variants_3seeds'
    c.DEFAULT_OUTPUT_SUBDIR = scene.removeprefix('smac_') + '_local_available_qmix_variants_3seeds'
    return c


def main(scene='smac_5m6m'):
    c = configure(scene)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime-root', type=Path,
                        default=Path(os.environ.get('RUNTIME_ROOT',
                                     '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--retained-figure-root', type=Path)
    parser.add_argument('--retained-curves', type=Path)
    parser.add_argument('--mean-window', type=int)
    args, _ = parser.parse_known_args()
    if not args.retained_curves and not args.retained_figure_root:
        sys.argv.extend(['--retained-figure-root', str(args.runtime_root / 'figures')])
    if args.mean_window is None:
        sys.argv.extend(['--mean-window', '10'])
    for flag in ('--local-only', '--available-only'):
        if flag not in sys.argv:
            sys.argv.append(flag)
    print('LOCAL ONLY: {} logs + retained raw CSVs; no cloud history queries. '
          'Only models with readable curves are plotted.'.format(scene), flush=True)
    c.main()


if __name__ == '__main__':
    main()
