#!/usr/bin/env python3
"""Verify one-shot ended-only routing without Slurm, W&B or tmux operations."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import ozstar_calibration_suite_sync as suite
import ozstar_sync_ended_once as ended


def main():
    with patch.object(ended, 'sync_once') as sync:
        ended.main([])
        sync.assert_called_once()
        assert sync.call_args.kwargs == {'ended_only': True}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        logs, wandb = root / 'ozstar_logs', root / 'wandb'
        logs.mkdir()
        wandb.mkdir()
        jobs = {'1': 'synthetic_completed', '2': 'synthetic_running',
                '3': 'synthetic_pending', '4': 'synthetic_unknown'}
        (logs / (suite.GROUP + '_synthetic.json')).write_text(json.dumps(dict(
            group=suite.GROUP, plans=[dict(job_name=name) for name in jobs.values()],
            submitted={name: job for job, name in jobs.items()})))
        runs = {}
        for job, name in jobs.items():
            run_id = 'fixture' + job
            directory = wandb / ('offline-run-20261015_120000-' + run_id)
            (directory / 'files').mkdir(parents=True)
            (directory / ('run-' + run_id + '.wandb')).write_bytes(b'synthetic')
            (directory / 'files/config.yaml').write_text('wandb_run_name:\n  value: ' + name)
            (logs / (name + '_' + job + '.out')).write_text(str(directory))
            runs[job] = directory
        states = {job: dict(state=state, exit_code='0:0', elapsed='1:00')
                  for job, state in (('1', 'COMPLETED'), ('2', 'RUNNING'), ('3', 'PENDING'))}
        original_discover = suite.discover
        def only_ended(wandb_root, log_root, repo, selected):
            assert selected == {'1': jobs['1']}
            return original_discover(wandb_root, log_root, repo, selected)
        with redirect_stdout(io.StringIO()) as output, \
                patch.object(suite, 'accounting', return_value=states), \
                patch.object(suite, 'discover', side_effect=only_ended), \
                patch.object(suite.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as upload:
            suite.sync_once(root, root, 'synthetic_python', ended_only=True)
            upload.assert_called_once()
            assert upload.call_args.args[0][-1] == str(runs['1'])
            assert 'synthetic_running' not in output.getvalue()
            assert 'synthetic_pending' not in output.getvalue()
            assert 'synthetic_unknown' not in output.getvalue()
            suite.sync_once(root, root, 'synthetic_python', ended_only=True)
            upload.assert_called_once()
        assert all(path.is_dir() for path in runs.values())
    print('PASS: one pass only; no tmux; only known terminal jobs discovered/uploaded; unchanged skip; all data retained')


if __name__ == '__main__':
    main()
