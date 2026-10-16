#!/usr/bin/env python3
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import shlex
from unittest.mock import patch
from types import SimpleNamespace

import ozstar_calibration_suite_sync as c


def running_scope_check():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        logs, wandb = root / 'ozstar_logs', root / 'wandb'
        logs.mkdir()
        wandb.mkdir()
        jobs = {str(i): 'synthetic_job_' + str(i) for i in range(1, 7)}
        (logs / (c.GROUP + '_synthetic.json')).write_text(json.dumps(dict(
            group=c.GROUP, plans=[dict(job_name=name) for name in jobs.values()],
            submitted={name: job for job, name in jobs.items()})))
        states = {}
        for job, state in zip(jobs, ('RUNNING', 'COMPLETED', 'PENDING', 'FAILED', 'RUNNING', 'RUNNING')):
            states[job] = dict(state=state, exit_code='0:0', elapsed='1:00')
            run = wandb / ('offline-run-20261015_120000-synthetic' + job)
            (run / 'files').mkdir(parents=True)
            (run / ('run-synthetic' + job + '.wandb')).write_bytes(b'synthetic')
            (run / 'files' / 'config.yaml').write_text('wandb_run_name:\n  value: ' + jobs[job])
            (logs / (jobs[job] + '_' + job + '.out')).write_text(str(run) + '\n')
        queue = '1|RUNNING\n2|COMPLETED\n3|PENDING\n4|FAILED\n6|RUNNING\n999|RUNNING\nbad row'
        def queue_command(args):
            if args == ['id', '-un']:
                return 'synthetic_user'
            assert args == ['squeue', '-u', 'synthetic_user', '-h', '-o', '%A|%T']
            return queue
        def known_accounting(selected, repo):
            assert set(selected) == {'1', '6'}  # Never query old or stale RUNNING accounting.
            return {'1': states['1']}  # Job 6 has no confirmed WorkDir/accounting.
        def upload(args, **kwargs):
            assert args[-1].endswith('synthetic1'), args
            return SimpleNamespace(returncode=0)
        with redirect_stdout(io.StringIO()) as output, \
                patch.object(c, 'command', side_effect=queue_command), \
                patch.object(c, 'accounting', side_effect=known_accounting), \
                patch.object(c.subprocess, 'run', side_effect=upload) as uploads:
            c.sync_once(root, root, 'synthetic_python', running_only=True)
            assert uploads.call_count == 1
            assert 'COMPLETED' not in output.getvalue() and 'PENDING' not in output.getvalue()
            assert 'synthetic5' not in output.getvalue() and 'synthetic6' not in output.getvalue()
            c.sync_once(root, root, 'synthetic_python', running_only=True)
            assert uploads.call_count == 1  # Still skip unchanged active data.
        # No active jobs: do not scan local historical runs or query accounting.
        with redirect_stdout(io.StringIO()), patch.object(c, 'running_jobs', return_value={}), \
                patch.object(c, 'accounting') as acct, patch.object(c, 'discover') as discover, \
                patch.object(c.subprocess, 'run') as uploads:
            c.sync_once(root, root, 'synthetic_python', running_only=True)
            acct.assert_not_called()
            discover.assert_not_called()
            uploads.assert_not_called()
        # A job can finish during discovery; re-check just before launching upload.
        (wandb / 'offline-run-20261015_120000-synthetic1' / 'run-synthetic1.wandb').write_bytes(b'new data')
        with redirect_stdout(io.StringIO()), \
                patch.object(c, 'running_jobs', side_effect=[{'1': jobs['1']}, {}]), \
                patch.object(c, 'accounting', return_value={'1': states['1']}), \
                patch.object(c.subprocess, 'run') as uploads:
            c.sync_once(root, root, 'synthetic_python', running_only=True)
            uploads.assert_not_called()
        # Index only the selected job's logs, not every historical .out/.err.
        original_open = Path.open
        def selected_log_only(path, *args, **kwargs):
            assert path.name == jobs['1'] + '_1.out', path
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', autospec=True, side_effect=selected_log_only):
            indexed = c.job_index(logs, wandb, root, job_ids={'1'})
            assert indexed == {'offline-run-20261015_120000-synthetic1': {'1'}}
    print('PASS: live queue intersection; no pending/ended/stale/unknown uploads; active-only log reads; empty round; pre-upload recheck')


def launch_scope_check():
    for flags, expected in (([], '--running-only'), (['--all-states'], '--all-states'),
                            (['--ended-only'], '--ended-only')):
        calls = []
        def tmux(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=1 if args[1] == 'has-session' else 0)
        with redirect_stdout(io.StringIO()), patch.object(c.sys, 'argv', ['sync.py', 'start'] + flags), \
                patch.object(c, 'manifest_jobs', return_value={'1': 'synthetic'}), \
                patch.object(c.subprocess, 'run', side_effect=tmux), \
                patch.object(c, 'command', return_value='0'), patch.object(c.time, 'sleep'):
            c.main()
        launched = next(args[-1] for args in calls if args[1] == 'respawn-pane')
        parts = shlex.split(launched)
        assert parts[parts.index('loop') + 1] == expected
    # An existing live window is not silently restarted to change its scope.
    with redirect_stdout(io.StringIO()), patch.object(c.sys, 'argv', ['sync.py', 'start']), \
            patch.object(c, 'manifest_jobs', return_value={'1': 'synthetic'}), \
            patch.object(c.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as tmux, \
            patch.object(c, 'command', return_value='0'):
        c.main()
        assert tmux.call_count == 1
    print('PASS: tmux defaults RUNNING-only; explicit mode forwarded; existing window not silently changed')


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
        variance = 'smac_5m6m_qmix_state_group_variance_10m_s1_ablate20261015'
        new_manifest = logs / (c.GROUPS[2] + '_20261015_1.json')
        new_manifest.write_text(json.dumps(dict(group=c.GROUPS[2], plans=[dict(job_name=variance), dict(job_name=ablation)],
            submitted={variance: '789'}, retained={ablation: '456'})))
        assert c.manifest_jobs(logs) == {'123': name, '456': ablation, '789': variance}
        gate = 'smac_8m9m_qmix_state_static10_variance_10m_s1_ablate20261015'
        gate_manifest = logs / (c.GROUPS[4] + '_20261015_1.json')
        gate_manifest.write_text(json.dumps(dict(group=c.GROUPS[4], plans=[dict(job_name=gate), dict(job_name=ablation)],
            submitted={gate: '890'}, retained={ablation: '456'})))
        assert c.manifest_jobs(logs) == {'123': name, '456': ablation, '789': variance, '890': gate}
        gate_manifest.unlink()
        new_manifest.unlink()
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
            assert args[1].endswith('/ozstar_wandb_sync_verified.py')
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
            ledger_path = logs / (c.GROUP + '_sync_ledger.json')
            ledger = json.loads(ledger_path.read_text())
            del ledger[run.name]['verification_version']
            ledger_path.write_text(json.dumps(ledger))
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 2  # Old exit-code-only success must be reverified.
            state['123']['state'] = 'COMPLETED'
            c.sync_once(root, root, 'synthetic_python', ended_only=True)
            assert len(calls) == 3  # Final state forces final upload.
            (run / 'run-ab12.wandb').write_bytes(b'new tail')
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 4
            (run / 'run-ab12.wandb').write_bytes(b'failed upload tail')
            with patch.object(c.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                c.sync_once(root, root, 'synthetic_python')
            c.sync_once(root, root, 'synthetic_python')
            assert len(calls) == 5  # Failure never advances ledger.
        with patch.object(c, 'command', return_value='123|FAILED|1:0|00:20:00|{}|\n124|COMPLETED|0:0|01:00:00|/other|'.format(root)):
            result = c.accounting({'123': name, '124': 'other'}, root)
            assert result['123']['state'] == 'FAILED' and '124' not in result
    print('PASS: exact cohort IDs; terminal states; ended backfill; live/final sync; unchanged skip; failure retry; no unrelated uploads')
    running_scope_check()
    launch_scope_check()


if __name__ == '__main__':
    main()
