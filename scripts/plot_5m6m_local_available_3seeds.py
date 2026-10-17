#!/usr/bin/env python3
"""Upload available 5m6m QMIX/VDN curves from local logs and retained CSVs only."""
import argparse
import os
from pathlib import Path
import sys

import plot_5m6m_8m9m_qmix_variants_3seeds as study


def configure():
    study.configure()
    c = study.charts
    # Materialize before narrowing the wrapper's two-map manifest validation.
    plans = [p for p in c.build_plans(c.ROOT) if p['scene'] == 'smac_5m6m']
    c.build_plans = lambda root: plans
    c.SCENES = {'smac_5m6m': c.SCENES['smac_5m6m']}
    c.SCENE_MODELS = {'smac_5m6m': study.SCENE_MODELS['smac_5m6m']}
    c.GROUP = 'smac_5m6m_local_available_qmix_variants_3seeds'
    c.DEFAULT_OUTPUT_SUBDIR = '5m6m_local_available_qmix_variants_3seeds'
    return c


def main():
    c = configure()
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
    print('LOCAL ONLY: 5m6m logs + retained raw CSVs; no cloud history queries. '
          'Only models with readable curves are plotted.', flush=True)
    c.main()


if __name__ == '__main__':
    main()
