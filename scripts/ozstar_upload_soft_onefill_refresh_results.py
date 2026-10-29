#!/usr/bin/env python3
"""Audit/backfill only the new soft one-fill / refresh 2M three-map cohort.

18 exact names, at most six complete mean3. No training/cleanup/tmux changes.
Run without flags to check state and upload missing ended originals/full mean3;
--status-only performs a read-only receipt/accounting audit without W&B calls.
"""
from pathlib import Path

import ozstar_submit_soft_onefill_refresh_three_maps_18jobs as source
import ozstar_upload_recent_21jobs_20261028 as uploader

GROUP = 'soft_onefill_refresh_three_maps_backfill_20261029'
LABELS = {
    'qmix_soft_onefill_p10': ('QMIX deterministic soft one-fill / p0=0.1 / live mixer state (10M)', '#17becf'),
    'qmix_soft_onefill_p10_refresh_2000k': ('QMIX deterministic soft one-fill / p0=0.1 / state refresh stage=2M (10M)', '#d627a5'),
}


def build_plans(repo):
    plans = source.build_plans(repo)
    expected = {(scene, label) for scene, _, _, _ in source.MAPS for label in source.LABELS}
    if (len(plans) != len({p['job_name'] for p in plans}) or len(plans) != 18 or
            {(p['scene'], p['label']) for p in plans} != expected):
        raise RuntimeError('Expected only this new three-map/two-model/eighteen-job cohort')
    if any({p['seed'] for p in plans if (p['scene'], p['label']) == key} != {1, 2, 3}
           for key in expected):
        raise RuntimeError('Expected seeds 1/2/3 for every exact model')
    return plans


def main(argv=None):
    build_plans(Path(__file__).resolve().parents[1])
    previous = uploader.GROUP, uploader.SOURCES, uploader.LABELS, uploader.build_plans
    try:
        uploader.GROUP, uploader.SOURCES = GROUP, (source,)
        uploader.LABELS, uploader.build_plans = LABELS, build_plans
        return uploader.main(argv)
    finally:
        uploader.GROUP, uploader.SOURCES, uploader.LABELS, uploader.build_plans = previous


if __name__ == '__main__':
    main()
