#!/usr/bin/env python3
"""Final upload and cleanup, only while the user's Slurm queue is empty."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import fcntl


def idle():
    return not subprocess.check_output(
        ['squeue', '-u', subprocess.check_output(['id', '-un'], text=True).strip(),
         '-h', '-o', '%i'], text=True).strip()


def main():
    repo = Path(os.environ.get('REPO_DIR', '/home/kyang/code/gomarl-dual-branch')).resolve()
    root = repo / 'wandb'
    os.chdir(repo)
    with (repo / '.wandb-counter-sync-once.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another sync is running; deferred.', flush=True)
            return
        if not idle():
            print('Queue is not empty; final cleanup deferred.', flush=True)
            return
        failed = 0
        for directory in sorted(root.glob('offline-run-*')):
            if directory.is_symlink() or not directory.is_dir():
                continue
            run_id = directory.name.rsplit('-', 1)[-1]
            data = directory / ('run-' + run_id + '.wandb')
            if not data.is_file() or not idle():
                continue
            before = {str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
                      for p in directory.rglob('*') if p.is_file()}
            print('Final upload: ' + str(directory), flush=True)
            result = subprocess.run([
                sys.executable, '-m', 'wandb', 'sync', '--append',
                '--include-offline', '--include-synced', str(directory)],
                timeout=int(os.environ.get('SYNC_TIMEOUT', '1800')))
            marker = directory / (data.name + '.synced')
            if result.returncode or not marker.exists():
                print('Upload not confirmed; retained ' + str(directory), flush=True)
                failed += 1
                continue
            # Confirm the original run is visible in its original cloud destination.
            # Do not override entity/project and accidentally move historical runs.
            import wandb
            from wandb.sdk.internal import datastore
            from wandb.proto import wandb_internal_pb2
            store = datastore.DataStore()
            store.open_for_scan(str(data))
            cloud_path = None
            while True:
                record_bytes = store.scan_data()
                if record_bytes is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(record_bytes)
                if record.HasField('run'):
                    r = record.run
                    if r.entity and r.project and r.run_id == run_id:
                        cloud_path = '/'.join((r.entity, r.project, r.run_id))
                        break
            if not cloud_path:
                print('Cloud destination unresolved; retained.', flush=True)
                continue
            remote = wandb.Api().run(cloud_path)
            cloud_files = {f.name: f.size for f in remote.files()}
            media = directory / 'files' / 'media'
            if any(cloud_files.get(str(p.relative_to(directory / 'files'))) != p.stat().st_size
                   for p in media.rglob('*') if p.is_file()):
                print('Cloud media incomplete; retained.', flush=True)
                continue
            if not idle() or any(not (directory / p).is_file() or
                ((directory / p).stat().st_size, (directory / p).stat().st_mtime_ns) != stat
                for p, stat in before.items()):
                print('Local data changed; retained.', flush=True)
                continue
            if os.environ.get('CLEAN_SYNCED') == 'YES':
                shutil.rmtree(directory)
                print('Removed local synced W&B run: ' + str(directory), flush=True)
            else:
                print('Verified; set CLEAN_SYNCED=YES to remove local W&B data.', flush=True)
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
