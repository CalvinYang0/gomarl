#!/usr/bin/env python3
"""Exercise the exact Nov8 receipt scope without Slurm or W&B writes."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import ozstar_upload_projected_faststage_ablation_results as suite


def main():
    repo = Path(__file__).resolve().parents[1]
    plans = suite.build_plans(repo)
    assert len({p['job_name'] for p in plans}) == 18
    complete = {p['job_name']: str(80000+i) for i, p in enumerate(plans)}
    retried, active = (p['job_name'] for p in (plans[0], plans[-1]))
    other = 'smac_5m6m_qmix_projected_init1_faststages8m_10m_s2_projfast20261104'
    with TemporaryDirectory(prefix='SYNTHETIC-faststage-upload-') as temp:
        root = Path(temp).resolve()
        logs = root/'ozstar_logs'
        logs.mkdir()
        for suffix, ids in [('first', complete), ('retry', {retried: '90001', active: '90002'})]:
            (logs/(suite.cohort.GROUP+'_'+suffix+'.json')).write_text(json.dumps(dict(
                group=suite.cohort.GROUP, plans=plans, submitted=ids)))
        (logs/'unrelated_older.json').write_text(json.dumps(dict(group='unrelated',
            plans=[dict(job_name=other)], submitted={other: '90003'})))

        def account(jobs, checkout):
            assert checkout == repo
            assert jobs == {**{job: name for name, job in complete.items()},
                            '90001': retried, '90002': active}
            return {job: dict(state={'90001': 'TIMEOUT', '90002': 'RUNNING'}.get(job, 'COMPLETED'),
                exit_code='0:0', submitted='2026-11-10T00:00:00' if job.startswith('9')
                else '2026-11-09T00:00:00') for job in jobs}

        def sync(checkout, runtime, python, **kwargs):
            assert checkout == repo and runtime == root
            assert kwargs == dict(ended_only=True, verify_existing=True,
                                  wanted_names=set(complete)-{active})
            selected = suite.uploader.sync.manifest_jobs(logs)
            assert '90001' in selected and '90002' in selected
            assert complete[retried] not in selected and complete[active] not in selected
            return dict(checked=17, current=17, uploaded=0, failed=0, missing_local=[], busy=False)

        def render(runtime, project, output, selected, chosen, base_render):
            assert len(selected) == 3 and {p['seed'] for p in selected} == {1, 2, 3}
            assert not {retried, active} & {p['job_name'] for p in selected}
            p = selected[0]
            return dict(scene=p['scene'], model=p['label'], sources=[],
                        rows=[dict(env_steps=10000000)]), []

        fake_wandb = SimpleNamespace(Api=lambda **kw: object())
        with patch.dict(sys.modules, wandb=fake_wandb), \
                patch.object(suite.integrity, 'detailed_accounting', side_effect=account), \
                patch.object(suite.integrity, 'strict_render', side_effect=render) as make_mean, \
                patch.object(suite.uploader.runner, 'sync_once', side_effect=sync) as originals, \
                patch.object(suite.uploader, 'scalar_current', return_value=True):
            args = ['--runtime-root', str(root)]
            report = suite.main(args+['--status-only'])
            assert report['state_counts'] == dict(COMPLETED=16, TIMEOUT=1, RUNNING=1)
            originals.assert_not_called()
            make_mean.assert_not_called()
            assert not (root/'figures').exists()
            report = suite.main(args)
            originals.assert_called_once()
            assert make_mean.call_count == len(report['mean3']) == 4
            assert len(report['skipped_mean3']) == 2 and not report['errors']
    print('PASS: exact Nov8 scope; latest retries; status-only; terminal-only sync; complete groups only')


if __name__ == '__main__':
    main()
