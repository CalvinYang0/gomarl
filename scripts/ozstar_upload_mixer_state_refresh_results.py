#!/usr/bin/env python3
"""One-shot update of ONLY the October 23 refresh cohort: 9 originals + 3 mean3.

No training, deletion, tmux change, historical scan or cross-cohort substitution.
Use --no-upload for local-only plots/inventory with no scheduler/cloud calls.
"""
from types import SimpleNamespace

import ozstar_submit_5m6m_mixer_state_refresh_9jobs as source
import ozstar_upload_5m6m_onefill_results as runner
import ozstar_calibration_suite_sync as sync

LABELS = {
    'qmix_state_refresh_500k': ('QMIX mixer state refresh / stage=0.5M (10M)', '#1b9e77'),
    'qmix_state_refresh_1000k': ('QMIX mixer state refresh / stage=1M (10M)', '#d95f02'),
    'qmix_state_refresh_2000k': ('QMIX mixer state refresh / stage=2M (10M)', '#7570b3'),
}
cohort = SimpleNamespace(GROUP=source.GROUP, LABELS=tuple(LABELS), build_plans=source.build_plans)
SUBDIR = '5m6m_mixer_state_refresh_20261023_3seeds'
TITLE = 'Mixer state refresh curriculum (October 23 cohort only)'


def main(argv=None):
    study = runner.study
    previous = (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
                sync.GROUP, sync.GROUPS, study.LABELS, study.SCENE_MODELS)
    scope = runner.original_scope

    def completed_scope(repo, runtime, expected):
        records = scope(repo, runtime, expected)
        if any(row['state'] != 'COMPLETED' or row['exit_code'] != '0:0' for row in records.values()):
            raise RuntimeError('Refresh update requires all nine jobs COMPLETED / 0:0; inspect failed jobs first')
        return records

    try:
        runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE = cohort, SUBDIR, TITLE
        runner.original_scope = completed_scope
        # Both manifest discovery and sync_once read this shared module's scope.
        # Restrict only this process/call; do not expand the historical tmux loop.
        sync.GROUPS = (source.GROUP,)
        sync.GROUP = source.GROUP
        study.LABELS = LABELS
        study.SCENE_MODELS = {'smac_5m6m': tuple(LABELS)}
        return runner.main(argv)
    finally:
        (runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE, runner.original_scope,
         sync.GROUP, sync.GROUPS, study.LABELS, study.SCENE_MODELS) = previous


if __name__ == '__main__':
    main()
