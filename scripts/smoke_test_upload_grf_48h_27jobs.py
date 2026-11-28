#!/usr/bin/env python3
"""Synthetic exact-GRF uploader QA; no Slurm calls, network writes or real data."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch
import numpy as np
import ozstar_upload_grf_48h_27jobs as g


def main():
    plans = g.expected_plans()
    jobs = g.expected_jobs()
    assert len(plans) == len(jobs) == 27
    assert list(jobs)[0] == '21445821' and list(jobs)[-1] == '21445847'
    assert len({(p['scene'], p['label']) for p in plans}) == 9
    repo = Path('/home/kyang/code/gomarl-sixmap-9b8d3a2')
    rows = '\n'.join('|'.join((j,n,'COMPLETED','0:0',str(repo))) for j,n in jobs.items())
    with patch.object(g.subprocess, 'check_output', return_value=rows):
        states = g.accounting(repo)
        assert len(states) == 27 and all(r['state'] == 'COMPLETED' for r in states)
    with patch.object(g.subprocess, 'check_output', return_value=rows.replace('21445821|grf_counter', '21445821|wrong')):
        try:
            g.accounting(repo)
        except RuntimeError: pass
        else: raise AssertionError('Accepted wrong job identity')
    h = g.load_helpers()
    assert h.PROJECT == 'hjh331-sjtu/gomarl'
    assert hasattr(h, 'upload_ended')
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-GRF48-upload-') as tmp:
        root = Path(tmp)
        (root/'wandb').mkdir()
        (root/'ozstar_logs').mkdir()
        selected = plans[:3]
        curves = {p['job_name']: ((np.array([0,10000,10000001]), np.full(3,p['seed']/10)),
                   dict(run_id='synthetic'+str(p['seed']), source='local', attempt_start='2026-11-24T00:00:00Z'))
                  for p in selected}
        record, inv = g.grf_mean3(selected, curves, root/'aggregate', 'QMIX')
        last = record['rows'][-1]
        assert last['env_steps'] == 10000000 and last['seed_count'] == 3
        assert np.isclose(last[g.METRIC], .2) and np.isclose(last[g.METRIC+'_std'], .1)
        assert 'test_battle_won_mean' not in last
        curves[selected[0]['job_name']][0][0][-1] = 9990000
        try: g.grf_mean3(selected, curves, root/'partial', 'QMIX')
        except ValueError: pass
        else: raise AssertionError('Partial seed promoted to full mean3')
        # Exact job-owned Sacred wins before any cloud-history request.
        p = selected[0]
        sacred = root/'results/sacred/grf/synthetic/1'
        sacred.mkdir(parents=True)
        (sacred/'config.json').write_text(json.dumps(dict(wandb_run_name=p['job_name'])))
        (sacred/'run.json').write_text(json.dumps(dict(start_time='2026-11-24T00:00:00')))
        (sacred/'info.json').write_text(json.dumps({g.METRIC+'_T':[0,10000001],g.METRIC:[0,.5]}))
        owner = {p['job_name']:dict(ids={'synthetic1'}, sacred={str(sacred.resolve())})}
        with patch.object(h.integrity, 'run_ownership', return_value=owner), \
             patch.object(h, 'owned_directories', return_value={}), \
             patch.object(h, 'exact_local_index', return_value={}):
            def no_cloud(): raise AssertionError('Local data should be preferred')
            recovered = g.selected_curves(h, repo, root, {'21445821':p['job_name']}, [p], no_cloud)
            assert recovered[p['job_name']][1]['source'] == 'sacred'
        summary = dict(last)
        rid, digest = h.up.scalar_identity(record, h.PROJECT)
        remote = SN(config=dict(source_digest=digest), summary=summary)
        api = lambda: SN(run=lambda _: remote)
        item = dict(run_id=rid, digest=digest)
        assert g.current(api, h.PROJECT, item)
        del summary['seed_3/'+g.METRIC]
        assert not g.current(api, h.PROJECT, item)
        # Read-only mode never touches upload functions or creates output.
        with patch.object(g, 'accounting', return_value=states), \
             patch.object(h.cleanup, 'live_jobs', return_value=set()), \
             patch.object(h, 'upload_ended', side_effect=AssertionError('Read-only uploaded')):
            g.run(repo, root, False, h)
        assert not (root/'figures').exists()
        uploaded = {}
        def fake_curves(h, repo, runtime, chosen, selected, api):
            assert set(chosen).issubset(jobs)
            return {p['job_name']: ((np.array([0,10000001]), np.full(2,p['seed']/10)),
                dict(run_id='synthetic'+str(p['seed']), source='local', attempt_start='2026-11-24T00:00:00Z'))
                for p in selected}
        def fake_originals(repo, runtime, ready, output, report, path, api):
            assert len(ready) == 27
            report['originals'] = [dict(status='verified') for r in ready]
        def fake_upload(record, project, output, wandb):
            rid, digest = h.up.scalar_identity(record, project)
            uploaded[project+'/'+rid] = SN(config=dict(source_digest=digest), summary=record['rows'][-1])
        def fake_remote(path):
            if path not in uploaded: raise RuntimeError('SYNTHETIC not yet uploaded')
            return uploaded[path]
        fake_wandb = SN(Api=lambda **kw: SN(run=fake_remote))
        with patch.dict('sys.modules', {'wandb':fake_wandb}), \
             patch.object(g, 'accounting', return_value=states), \
             patch.object(h.cleanup, 'live_jobs', return_value=set()), \
             patch.object(h, 'upload_ended', side_effect=fake_originals), \
             patch.object(g, 'selected_curves', side_effect=fake_curves), \
             patch.object(h.up, 'upload_model', side_effect=fake_upload):
            g.run(repo, root, True, h)
        report = json.loads((root/'figures'/(g.GROUP+'_upload')/'upload_report.json').read_text())
        assert len(uploaded) == len(report['groups']) == 9 and not report['errors']
        assert all(r['status'] == 'verified' for r in report['groups'])
    print('PASS exact 27 identities, GRF metric, strict 10M/3 seeds, local-first, cloud verification, status-only')


if __name__ == '__main__': main()
