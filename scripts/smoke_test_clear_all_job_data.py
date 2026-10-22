#!/usr/bin/env python3
"""Synthetic-only cleanup tests; no real quota, scheduler, SSH or experiment data."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ozstar_clear_all_job_data as clear


class ClearTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC-all-job-cleanup-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = self.root / 'runtime'
        self.repo = self.root / 'repo'
        self.bases = tuple(self.runtime / name for name in ('results', 'wandb', 'figures'))
        for base in self.bases + (self.runtime / 'ozstar_logs', self.repo / 'src'):
            base.mkdir(parents=True)
        self.source = self.repo / 'src' / 'keep.py'
        self.source.write_text('untouched source')
        self.receipt = self.runtime / 'ozstar_logs' / 'submit.json'
        self.receipt.write_text('keep receipt')

    def fixture(self):
        targets = []
        for base in self.bases:
            target = base / 'job'
            target.mkdir()
            (target / 'history.json').write_text('data')
            targets.append(target)
        return targets

    def test_complete_apply_preserves_roots_locks_code_receipts(self):
        targets = self.fixture()
        lock = self.runtime / 'wandb' / '.gomarl-sync-once.lock'
        lock.touch()
        inode = lock.stat().st_ino
        external = self.root / 'external.txt'
        external.write_text('not a job file')
        link = self.runtime / 'wandb' / 'latest-run'
        link.symlink_to(external)
        with patch.multiple(clear, RUNTIME=self.runtime, REPO=self.repo, BASES=self.bases), \
                patch.object(clear, 'safety_state') as safety, \
                patch.object(clear, 'command', side_effect=lambda args: 'SYNTHETIC QUOTA' if args[0] == 'quota' else ''), \
                redirect_stdout(io.StringIO()) as output:
            clear.main(['--apply'])
        self.assertIn('CLEANUP COMPLETE', output.getvalue())
        self.assertTrue(all(not path.exists() for path in targets))
        self.assertTrue(all(base.is_dir() for base in self.bases))
        self.assertEqual(lock.stat().st_ino, inode)
        self.assertEqual(self.source.read_text(), 'untouched source')
        self.assertEqual(self.receipt.read_text(), 'keep receipt')
        self.assertEqual(external.read_text(), 'not a job file')
        self.assertFalse(link.is_symlink())
        self.assertGreaterEqual(safety.call_count, 5)
        logs = list((self.runtime / 'ozstar_logs').glob('cleanup_all_job_data_*.jsonl'))
        self.assertEqual(len(logs), 1)
        records = [json.loads(line) for line in logs[0].read_text().splitlines()]
        self.assertFalse(records[0]['backup_verified'])
        self.assertEqual(records[-1]['action'], 'complete')
        self.assertEqual(records[-1]['deleted_targets'], 4)

    def test_preview_does_not_delete_or_create_lock_files(self):
        targets = self.fixture()
        with patch.multiple(clear, RUNTIME=self.runtime, REPO=self.repo, BASES=self.bases), \
                patch.object(clear, 'safety_state'), patch.object(clear, 'command', return_value='SYNTHETIC QUOTA'), \
                redirect_stdout(io.StringIO()) as output:
            clear.main([])
        self.assertIn('PREVIEW ONLY', output.getvalue())
        self.assertTrue(all(path.is_dir() for path in targets))
        self.assertFalse(list(self.runtime.rglob('*.lock')))
        self.assertFalse(list((self.runtime / 'ozstar_logs').glob('cleanup_all*')))

    def test_queue_failure_before_deletion(self):
        targets = self.fixture()
        with patch.multiple(clear, RUNTIME=self.runtime, REPO=self.repo, BASES=self.bases), \
                patch.object(clear, 'safety_state', side_effect=RuntimeError('active training')):
            with self.assertRaisesRegex(RuntimeError, 'active training'):
                clear.main(['--apply'])
        self.assertTrue(all(path.is_dir() for path in targets))

    def test_busy_shared_lock_preserved_and_no_deletion(self):
        targets = self.fixture()
        with patch.multiple(clear, RUNTIME=self.runtime, REPO=self.repo, BASES=self.bases), \
                patch.object(clear, 'safety_state'), patch.object(clear, 'command', return_value='SYNTHETIC QUOTA'), \
                patch.object(clear.fcntl, 'flock', side_effect=BlockingIOError('busy')), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(BlockingIOError):
                clear.main(['--apply'])
        self.assertTrue(all(path.is_dir() for path in targets))
        self.assertTrue((self.runtime / 'wandb' / clear.LOCKS[0]).is_file())

    def test_changed_file_and_failed_audit_are_retained(self):
        target = self.fixture()[0]
        base = target.parent
        state, _ = clear.snapshot(target, base)
        (target / 'history.json').write_text('a changed longer history')
        with self.assertRaisesRegex(RuntimeError, 'Data changed'):
            clear.remove_target(base, target, state, io.StringIO())
        state, _ = clear.snapshot(target, base)
        with patch.object(clear, 'audit_event', side_effect=OSError('quota exceeded')):
            with self.assertRaises(OSError):
                clear.remove_target(base, target, state, io.StringIO())
        self.assertTrue(target.is_dir())

    def test_symlink_root_special_foreign_owner_and_broad_target_rejected(self):
        target = self.fixture()[0]
        link = self.root / 'linked-data-root'
        link.symlink_to(target.parent, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            clear.snapshot(link / 'job', link)
        with self.assertRaises(RuntimeError):
            clear.snapshot(target.parent, target.parent)
        with patch.object(clear.os, 'getuid', return_value=os.getuid()+1):
            with self.assertRaisesRegex(RuntimeError, 'Foreign owner'):
                clear.snapshot(target, target.parent)
        fifo = target / 'special'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(RuntimeError, 'Special file'):
            clear.snapshot(target, target.parent)

    def test_live_queue_and_writer_on_other_node_stop(self):
        with patch.object(clear, 'command', side_effect=['kyang', '123|training']):
            with self.assertRaisesRegex(RuntimeError, 'queue is not empty'):
                clear.safety_state()
        with patch.object(clear.socket, 'gethostname', return_value='farnarkle1'), \
                patch.object(clear, 'command', side_effect=['kyang', '', '101 /usr/sbin/sshd', '102 wandb sync /data']):
            with self.assertRaisesRegex(RuntimeError, 'farnarkle2'):
                clear.safety_state()
        with patch.object(clear.socket, 'gethostname', return_value='farnarkle1'), \
                patch.object(clear, 'command', side_effect=['kyang', '', '101 /usr/sbin/sshd', OSError('SSH failed')]):
            with self.assertRaises(OSError):
                clear.safety_state()


if __name__ == '__main__':
    unittest.main()
