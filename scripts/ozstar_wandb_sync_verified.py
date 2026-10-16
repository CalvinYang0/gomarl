#!/usr/bin/env python3
"""Process-local legacy W&B sync repair plus original-run cloud verification.

No site-packages edits, training imports, new run IDs, cleanup or scheduling.
The caller retains responsibility for selecting jobs and holding the sync lock.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time


@dataclass(frozen=True)
class Snapshot:
    run_id: str
    entity: str
    project: str
    history_count: int
    last_step: object


def read_snapshot(directory):
    """Read complete records present BEFORE upload, not a moving final target."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    directory = Path(directory)
    files = list(directory.glob('run-*.wandb'))
    if len(files) != 1:
        raise ValueError('Expected exactly one run-*.wandb in ' + str(directory))
    expected_id = files[0].stem[4:]
    if directory.name.rsplit('-', 1)[-1] != expected_id:
        raise ValueError('Run directory/file IDs differ; refusing upload')
    store = DataStore()
    identity = None
    count, last_step = 0, None
    store.open_for_scan(str(files[0]))
    try:
        while True:
            try:
                data = store.scan_data()
            except AssertionError:
                if not store.in_last_block():
                    raise
                print('NOTE: incomplete live tail; verifying complete records only', flush=True)
                break
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField('run'):
                current = (record.run.run_id, record.run.entity, record.run.project)
                if current[0] != expected_id or (identity and identity != current):
                    raise ValueError('Offline run identity changed; refusing upload')
                identity = current
            elif record.HasField('history'):
                count += 1
                step = record.history.step.num if record.history.HasField('step') else None
                for item in record.history.item:
                    key = '.'.join(item.nested_key) if item.nested_key else item.key
                    if key == '_step':
                        step = json.loads(item.value_json)
                if step is not None:
                    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
                        raise ValueError('Invalid local history step')
                    last_step = step if last_step is None else max(last_step, step)
    finally:
        store.close()
    if identity is None:
        raise ValueError('No complete run record; retry when local data is available')
    if count and last_step is None:
        raise ValueError('History exists but its step cannot be verified')
    return Snapshot(*identity, count, last_step)


@contextmanager
def legacy_sync_guards():
    """Repair 0.18.7's null introspection and expose hidden worker/run errors."""
    import wandb
    from wandb.sdk.internal import internal_api, sender
    from wandb.sync.sync import SyncThread

    original_probe = internal_api.Api.server_settings_introspection
    original_error = sender.SendManager._handle_error
    original_worker = SyncThread.run
    worker_errors = []

    def nullable_server_settings(api):
        # Match the original probe and only normalize its nullable schema type.
        # Authentication/network/GraphQL errors still propagate, never get skipped.
        if api._server_settings_type is None:
            result = api.gql(internal_api.gql('''
                query ProbeServerSettings {
                    ServerSettingsType: __type(name: "ServerSettings") {
                        fields { name }
                    }
                }
            '''))
            info = (result or {}).get('ServerSettingsType') or {}
            api._server_settings_type = [field.get('name', '') for field in (info.get('fields') or [])]

    def checked_error(manager, record, error, run):
        original_error(manager, record, error, run)
        raise RuntimeError('W&B run creation failed: ' + error.message)

    def checked_worker(worker):
        try:
            original_worker(worker)
        except BaseException as exc:
            worker_errors.append(exc)

    patched_probe = wandb.__version__ == '0.18.7'
    if patched_probe:
        internal_api.Api.server_settings_introspection = nullable_server_settings
        print('COMPAT: W&B 0.18.7 nullable ServerSettings (uploader process only)', flush=True)
    sender.SendManager._handle_error = checked_error
    SyncThread.run = checked_worker
    try:
        yield worker_errors
    finally:
        internal_api.Api.server_settings_introspection = original_probe
        sender.SendManager._handle_error = original_error
        SyncThread.run = original_worker


def sync_cli(directory, entity, project):
    from wandb.cli.cli import cli

    args = ['sync', '-e', entity, '-p', project, '--append', '--include-offline',
            '--include-synced', '--no-mark-synced', '--skip-console', str(directory)]
    with legacy_sync_guards() as errors:
        result = cli.main(args=args, standalone_mode=False)
        if errors:
            raise RuntimeError('W&B sync worker failed: ' + str(errors[0])) from errors[0]
        if result not in (None, 0):
            raise RuntimeError('W&B CLI failed: ' + str(result))


def verify_cloud(snapshot, entity, project, attempts=3, delay=2):
    """Verify visibility and progress through the pre-upload scalar snapshot.

    This is not a full per-metric/media audit, nor evidence training finished.
    """
    import wandb

    path = '/'.join((entity, project, snapshot.run_id))
    for attempt in range(attempts):
        try:
            # Fresh client each attempt: do not reuse a cached missing/old Run.
            remote = wandb.Api(timeout=15).run(path)
            if remote.id != snapshot.run_id:
                raise RuntimeError('Cloud run ID differs from local original ID')
            cloud_step = remote.lastHistoryStep if snapshot.history_count else None
            if snapshot.history_count and (cloud_step is None or cloud_step < snapshot.last_step):
                raise RuntimeError('Cloud history behind snapshot: cloud={} expected>={}'.format(
                    cloud_step, snapshot.last_step))
            print('VERIFIED cloud={} cloud_step={} snapshot_step={} (not a full media audit)'.format(
                path, cloud_step, snapshot.last_step), flush=True)
            return
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('-e', '--entity')
    parser.add_argument('-p', '--project')
    args = parser.parse_args(argv)
    try:
        snapshot = read_snapshot(args.directory)
        entity, project = args.entity or snapshot.entity, args.project or snapshot.project
        if not entity or not project:
            raise ValueError('Missing original destination; provide explicit entity/project')
        sync_cli(args.directory, entity, project)
        verify_cloud(snapshot, entity, project)
    except Exception as exc:
        print('UPLOAD NOT VERIFIED: {}: {}'.format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
