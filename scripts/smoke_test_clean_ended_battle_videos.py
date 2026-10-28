#!/usr/bin/env python3
"""Synthetic video-only deletion tests; scheduler/SSH/quota are mocked."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import ozstar_clean_ended_battle_videos as clean


def main():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-video-cleanup-') as temp:
        root = Path(temp).resolve()
        runtime, repo = root/'runtime', root/'repo'
        videos = runtime/'results/test_battle_videos'
        logs = runtime/'ozstar_logs'
        for path in (videos, logs, runtime/'wandb', repo/'src'):
            path.mkdir(parents=True)
        keep = runtime/'results/sacred/keep.json'
        keep.parent.mkdir(parents=True)
        keep.write_text('KEEP sacred')
        gate = runtime/'results/mixer_onefill_gate/keep.json'
        gate.parent.mkdir(parents=True)
        gate.write_text('KEEP gate diagnostics')
        source = repo/'src/keep.py'
        source.write_text('KEEP source')
        external = root/'external.json'
        external.write_text('KEEP external source')
        group, name = 'SYNTHETIC_old_group', 'SYNTHETIC_old_run_s1'
        digest = hashlib.sha256(name.encode()).hexdigest()[:12]
        token = videos/(group+'__2026-10-27_00-00-00')
        leaf = token/('seed_1_'+digest)
        leaf.mkdir(parents=True)
        (leaf/'episode.mp4').write_bytes(b'SYNTHETIC video')
        (leaf/'episode.json').write_text('SYNTHETIC trace')
        (leaf/'external-link').symlink_to(external)
        os.utime(leaf/'external-link', (time.time()-1000,)*2, follow_symlinks=False)
        for item in (leaf/'episode.mp4', leaf/'episode.json', leaf):
            os.utime(item, (time.time()-1000,)*2)
        unknown = token/'seed_2_aaaaaaaaaaaa'
        unknown.mkdir()
        protected = videos/(clean.KEEP_GROUP+'__2026-10-28_00-00-00')/'seed_1_aaaaaaaaaaaa'
        protected.mkdir(parents=True)
        (logs/'SYNTHETIC.json').write_text(json.dumps(dict(group=group,
            plans=[dict(job_name=name, seed=1, exports=dict(GROUP_NAME=group))], submitted={name: '12345'})))
        index = clean.receipt_index(logs)
        with redirect_stdout(io.StringIO()):
            assert len(clean.inventory(videos, index, lambda job: True)) == 1
            assert not clean.inventory(videos, index, lambda job: False)
        before = keep.read_text(), gate.read_text(), source.read_text(), external.read_text()
        with patch.multiple(clean.safe, RUNTIME=runtime, REPO=repo), \
                patch.object(clean.safe, 'safety_state'), \
                patch.object(clean.safe, 'command', return_value='SYNTHETIC QUOTA'), \
                patch.object(clean, 'job_finished', return_value=True), redirect_stdout(io.StringIO()):
            clean.main([])
        assert leaf.exists() and not list((runtime/'wandb').glob('*.lock'))
        # A failed process/queue check must precede every destructive action.
        with patch.multiple(clean.safe, RUNTIME=runtime, REPO=repo), \
                patch.object(clean.safe, 'safety_state', side_effect=RuntimeError('writer still active')):
            try: clean.main(['--apply'])
            except RuntimeError: pass
            else: raise AssertionError('Active writer accepted')
        assert leaf.exists()
        # Busy locks must not be bypassed or unlinked.
        lock = runtime/'wandb'/clean.safe.LOCKS[0]
        lock.touch()
        inode = lock.stat().st_ino
        with patch.multiple(clean.safe, RUNTIME=runtime, REPO=repo), \
                patch.object(clean.safe, 'safety_state'), \
                patch.object(clean.safe, 'command', return_value='SYNTHETIC QUOTA'), \
                patch.object(clean.fcntl, 'flock', side_effect=BlockingIOError('busy')), redirect_stdout(io.StringIO()):
            try: clean.main(['--apply'])
            except BlockingIOError: pass
            else: raise AssertionError('Busy lock bypassed')
        assert leaf.exists() and lock.stat().st_ino == inode
        with patch.multiple(clean.safe, RUNTIME=runtime, REPO=repo), \
                patch.object(clean.safe, 'safety_state') as safety, \
                patch.object(clean.safe, 'command', return_value='SYNTHETIC QUOTA'), \
                patch.object(clean, 'job_finished', return_value=True) as account, redirect_stdout(io.StringIO()) as stdout:
            clean.main(['--apply'])
        assert not leaf.exists() and unknown.exists() and protected.exists()
        assert before == (keep.read_text(), gate.read_text(), source.read_text(), external.read_text())
        assert account.call_count >= 2 and safety.call_count >= 3
        assert lock.stat().st_ino == inode and 'Removed 1 old video leaves' in stdout.getvalue()
        audits = list(logs.glob('cleanup_ended_battle_videos_*.jsonl'))
        assert len(audits) == 1
        records = [json.loads(row) for row in audits[0].read_text().splitlines()]
        assert records[0]['backup_verified'] is False
        assert records[-1]['removed_leaves'] == 1
    print('PASS: exact receipt/seed/hash ownership, terminal recheck, preview, active-writer/busy-lock stop, video-only deletion, no symlink traversal, protected/unknown/Sacred/gate/code preservation (services mocked)')


if __name__ == '__main__':
    main()
