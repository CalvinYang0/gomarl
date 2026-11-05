#!/usr/bin/env python3
"""Audit Nov4 projected-init1 + faststages8M jobs; originals and full10M mean3.

Exact four-map/three-seed cohort only. Latest attempts are chosen by sacct Submit,
not outcome. No training, deletion, videos, tmux or project-wide cloud scan.
--status-only reads receipts/accounting without uploads or local output writes.
"""
from pathlib import Path
from unittest.mock import patch

import ozstar_submit_projected_fastrefresh_four_maps_12jobs as cohort
import ozstar_upload_3s5z_refresh_warmup_results as integrity

uploader = integrity.uploader
GROUP = 'projected_fastrefresh_four_maps_backfill_20261105'
LABELS = {cohort.LABEL: (
    'QMIX projected exact p0=1 + faststages8M / always-live evaluation', '#9467bd')}


def build_plans(repo):
    plans = cohort.build_plans(repo)
    expected = {scene for scene, _, _, _ in cohort.MAPS}
    if (len(plans) != 12 or len({p['job_name'] for p in plans}) != 12 or
            {p['scene'] for p in plans} != expected or
            {p['label'] for p in plans} != set(LABELS) or
            any({p['seed'] for p in plans if p['scene'] == scene} != {1, 2, 3}
                for scene in expected)):
        raise RuntimeError('Expected exactly the Nov4 combined model / four maps x three seeds')
    return plans


def main(argv=None):
    build_plans(Path(__file__).resolve().parents[1])
    chosen = {}
    original_manifest = uploader.sync.manifest_jobs
    original_render = uploader.render_model
    # Reuse the audited retry/ownership logic, without admitting the older
    # Nov1 parent experiments into the upload or aggregation scope.
    with patch.multiple(uploader, GROUP=GROUP, SOURCES=(cohort,), LABELS=LABELS,
            build_plans=build_plans,
            audit_jobs=lambda repo, runtime, plans: integrity.audit_latest(
                repo, runtime, plans, original_manifest, chosen),
            render_model=lambda runtime, project, output, plans: integrity.strict_render(
                runtime, project, output, plans, chosen, original_render)), \
            patch.object(uploader.sync, 'manifest_jobs', side_effect=lambda logs: dict(chosen)):
        return uploader.main(argv)


if __name__ == '__main__':
    main()
