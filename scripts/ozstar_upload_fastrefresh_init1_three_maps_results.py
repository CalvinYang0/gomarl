#!/usr/bin/env python3
"""Audit/upload Nov6's two independent parents, three maps and three seeds.

Only confirmed ended jobs' originals are synced (TIMEOUT included). Complete
mean3 requires three successful exact latest jobs and actual 10M test histories.
No training, cleanup, videos or broad W&B project scan.
"""
from pathlib import Path
from unittest.mock import patch

import ozstar_submit_fastrefresh_init1_three_maps_18jobs as cohort
import ozstar_upload_3s5z_refresh_warmup_results as integrity

uploader = integrity.uploader
GROUP = 'fastrefresh_init1_three_maps_backfill_20261108'
LABELS = {
    cohort.VARIANTS['faststages']: ('Sigmoid one-fill p0=.1 / faststages8M / live evaluation', '#ff7f0e'),
    cohort.VARIANTS['init1']: ('Projected one-fill / exact p0=1 / no refresh', '#9467bd'),
}


def build_plans(repo):
    plans = cohort.build_plans(repo)
    expected = {(scene, label, seed) for scene, _, _ in cohort.MAPS
        for label in LABELS for seed in (1, 2, 3)}
    if (len(plans) != 18 or len({p['job_name'] for p in plans}) != 18 or
            {(p['scene'], p['label'], p['seed']) for p in plans} != expected or
            any(not p['job_name'].endswith('_parents20261106') for p in plans)):
        raise RuntimeError('Expected only Nov6 parent transfers: six groups x seeds1/2/3')
    return plans


def main(argv=None):
    build_plans(Path(__file__).resolve().parents[1])
    chosen = {}
    manifest = uploader.sync.manifest_jobs
    render = uploader.render_model

    def render_selected(runtime, project, output, plans):
        try:
            return integrity.strict_render(runtime, project, output, plans, chosen, render)
        except SystemExit as exc:
            # The plot CLI exits when a group has no readable points. Keep that
            # group incomplete without stopping uploads for the other groups.
            raise RuntimeError(str(exc)) from exc

    # Narrow the reusable backfill workflow to this manifest and selected latest
    # job IDs. The similarly named Nov1 models and Nov4 combination are excluded.
    with patch.multiple(uploader, GROUP=GROUP, SOURCES=(cohort,), LABELS=LABELS,
            build_plans=build_plans,
            audit_jobs=lambda repo, runtime, plans: integrity.audit_latest(
                repo, runtime, plans, manifest, chosen),
            render_model=render_selected), \
            patch.object(uploader.sync, 'manifest_jobs', side_effect=lambda logs: dict(chosen)):
        return uploader.main(argv)


if __name__ == '__main__':
    main()
