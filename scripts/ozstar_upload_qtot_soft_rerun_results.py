#!/usr/bin/env python3
"""ONLY successful October 22 fresh reruns: 18 originals / 6 map-model mean3.

Original failed October 21 attempts and their PARTIAL means are never merged.
"""
from types import SimpleNamespace

import ozstar_resubmit_qtot_soft_18jobs_20261022 as source
import ozstar_upload_5m6m_onefill_results as runner
import ozstar_calibration_suite_sync as sync

LABELS = {
    'qmix_qtot_sum_bothones': ('QMIX complete Qtot sum / both inputs ones (rerun 10M)', '#17becf'),
    'qmix_qtot_sum_p10': ('QMIX complete Qtot sum / one-fill secondary (rerun 10M)', '#e377c2'),
    'qmix_qtot_alpha10_p10': ('QMIX complete Qtot alpha sum / alpha0=0.1 (rerun 10M)', '#ff7f0e'),
    'qmix_soft_onefill_p10': ('QMIX deterministic soft one-fill / p0=0.1 (rerun 10M)', '#9467bd'),
    'qmix_onefill_p10': ('QMIX historical hard-ST one-fill / p0=0.1 (rerun 10M)', '#2ca02c'),
}
SCENES = {
    'smac_5m6m': ('5 Marines vs. 6 Marines', 'test_battle_won_mean'),
    'smac_mmm2': ('MMM2', 'test_battle_won_mean'),
    'smac_corridor': ('Corridor', 'test_battle_won_mean'),
}
cohort = SimpleNamespace(GROUP=source.GROUP, LABELS=tuple(LABELS), build_plans=source.build_plans)
SUBDIR = 'qtot_soft_rerun_20261022_3seeds'
TITLE = 'Qtot composition and soft one-fill (October 22 fresh reruns only)'


def main(argv=None):
    study = runner.study
    previous = (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
                sync.GROUP, sync.GROUPS, study.LABELS, study.SCENE_MODELS, study.configure)
    scope, configure = runner.original_scope, study.configure
    plans = cohort.build_plans(study.charts.ROOT)
    models = {scene: tuple(dict.fromkeys(p['label'] for p in plans if p['scene'] == scene)) for scene in SCENES}
    if len(plans) != 18 or sum(len(v) for v in models.values()) != 6:
        raise RuntimeError('Expected eighteen reruns and six map/model groups')

    def completed_scope(repo, runtime, expected):
        records = scope(repo, runtime, expected)
        if any(row['state'] != 'COMPLETED' or row['exit_code'] != '0:0' for row in records.values()):
            raise RuntimeError('Rerun update requires all eighteen jobs COMPLETED / 0:0')
        return records

    def configure_maps():
        configure()
        study.charts.SCENES = dict(SCENES)

    try:
        runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE = cohort, SUBDIR, TITLE
        runner.original_scope = completed_scope
        sync.GROUP, sync.GROUPS = source.GROUP, (source.GROUP,)
        study.LABELS, study.SCENE_MODELS, study.configure = LABELS, models, configure_maps
        return runner.main(argv)
    finally:
        (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
         sync.GROUP, sync.GROUPS, study.LABELS, study.SCENE_MODELS, study.configure) = previous


if __name__ == '__main__':
    main()
