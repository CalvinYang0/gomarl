#!/usr/bin/env python3
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import ozstar_calibration_suite_sync as c


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        logs, wandb = root / 'ozstar_logs', root / 'wandb'
        logs.mkdir()
        wandb.mkdir()
        name = 'smac_5m6m_qmix_baseline_10m_s1_fresh20261014'
        (logs / (c.GROUP + '_20261014_1.json')).write_text(json.dumps(dict(
            group=c.GROUP, plans=[dict(job_name=name)], submitted={name: '123'}, retained={})))
        jobs = c.manifest_jobs(logs)
        assert jobs == {'123': name}
        ablation = 'smac_5m6m_qmix_state_group_qchange_isolated_10m_s1_ablate20261014'
        extra_manifest = logs / (c.GROUPS[1] + '_20261014_1.json')
        extra_manifest.write_text(json.dumps(dict(group=c.GROUPS[1], plans=[dict(job_name=ablation)],
                                                 submitted={ablation: '456'}, retained={})))
        assert c.manifest_jobs(logs) == {'123': name, '456': ablation}
        extra_manifest.unlink()  # Only this test fixture; remaining tests use one run.
        run = wandb / 'offline-run-20261014_120000-ab12'
        (run / 'files').mkdir(parents=True)
        (run / 'run-ab12.wandb').write_bytes(b'synthetic')
        (run / 'files' / 'config.yaml').write_text('wandb_run_name:\n  value: ' + name + '\n')
        (logs / (name + '_123.out')).write_text(str(run) + '\n')
        other = wandb / 'offline-run-20261014_120001-cd34'
        other.mkdir()
        (other / 'run-cd34.wandb').write_bytes(b'unrelated')
        assert c.discover(wandb, logs, root, jobs) == [(run, {'123'})]
        state = {'123': dict(state='RUNNING', exit_code='0:0', elapsed='1:00')}
        calls = []
        def upload(args, **kwargs):
            assert args[-1] == str(run)
            assert '--append' in args and '--no-mark-synced' in args
            calls.append(args)
            return SimpleNamespace(returncode=0)
        with redirect_stdout(io.StringIO()), patch.object(c, 'accounting', return_value=state), \
                patch.object(c.subprocess, 'run', side_effect=upload):
            c.sync_once(root, root, 'synthetic_python', ended_only=True)
            assert not calls
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 1
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 1  # Unchanged successful runs skipped.
            state['123']['state'] = 'COMPLETED'
            c.sync_once(root, root, 'synthetic_python', ended_only=True)
            assert len(calls) == 2  # Final state forces final upload.
            (run / 'run-ab12.wandb').write_bytes(b'new tail')
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 3
            (run / 'run-ab12.wandb').write_bytes(b'failed upload tail')
            with patch.object(c.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                c.sync_once(root, root, 'synthetic_python')
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 4  # Failure never advances ledger.
        with patch.object(c, 'command', return_value='123|FAILED|1:0|00:20:00|{}|\n124|COMPLETED|0:0|01:00:00|/other|'.format(root)):
            result = c.accounting({'123': name, '124': 'other'}, root)
            assert result['123']['state'] == 'FAILED' and '124' not in result
    print('PASS: exact cohort IDs; terminal states; ended backfill; live/final sync; unchanged skip; failure retry; no unrelated uploads')


if __name__ == '__main__':
    main()
