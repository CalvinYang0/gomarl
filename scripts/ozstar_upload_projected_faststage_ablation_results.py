#!/usr/bin/env python3
"""Audit and backfill the Nov8 projected-faststage ablation only (18 jobs).

Originals are synced for confirmed terminal jobs. A full mean3 requires all
three latest attempts to be successful with job-owned test histories to 10M.
--status-only checks receipts and Slurm without writing to W&B.
"""
from pathlib import Path
from unittest.mock import patch

import ozstar_submit_projected_faststage_ablation_18jobs as cohort
import ozstar_upload_3s5z_refresh_warmup_results as integrity


uploader = integrity.uploader
GROUP = 'projected_faststage_ablation_backfill_20261108'
LABELS = {
    cohort.VARIANTS['init01_8m'][0]: ('Projected one-fill p0=.1 / live by 8M', '#ff7f0e'),
    cohort.VARIANTS['init1_5m'][0]: ('Projected one-fill p0=1 / live by 5M', '#1f77b4'),
    cohort.VARIANTS['init1_5m_freeze90'][0]: ('Projected one-fill p0=1 / live and binary freeze p>.9 at 5M', '#9467bd'),
}


def build_plans(repo):
    plans = cohort.build_plans(repo)
    expected = {(scene, label, seed) for scene in cohort.SCENES
                for label in LABELS for seed in (1, 2, 3)}
    if (len(plans) != 18 or {(p['scene'], p['label'], p['seed']) for p in plans} != expected or
            any(p['job_name'] != '{}_{}_10m_s{}_fastablate20261108'.format(
                p['scene'], p['label'], p['seed']) for p in plans)):
        raise RuntimeError('Expected only Nov8 projected-faststage ablation: six groups x seeds 1/2/3')
    return plans


def main(argv=None):
    chosen = {}
    manifest = uploader.sync.manifest_jobs
    render = uploader.render_model

    def render_selected(runtime, project, output, plans):
        try:
            return integrity.strict_render(runtime, project, output, plans, chosen, render)
        except SystemExit as exc:
            # Missing history must not block otherwise complete map/model groups.
            raise RuntimeError(str(exc)) from exc

    with patch.multiple(uploader, GROUP=GROUP, SOURCES=(cohort,), LABELS=LABELS,
            build_plans=build_plans, render_model=render_selected,
            audit_jobs=lambda repo, runtime, plans: integrity.audit_latest(
                repo, runtime, plans, manifest, chosen)), \
            patch.object(uploader.sync, 'manifest_jobs', side_effect=lambda logs: dict(chosen)):
        return uploader.main(argv)


if __name__ == '__main__':
    main()
