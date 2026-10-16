#!/usr/bin/env python3
"""Real W&B 0.18.7 parser/compat tests; all cloud calls are mocked."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore
from wandb.sdk.internal import internal_api, sender
from wandb.sync.sync import SyncThread

import ozstar_wandb_sync_verified as sync


def fixture(root, run_id='original1'):
    directory = root / ('offline-run-20261016_120000-' + run_id)
    directory.mkdir()
    writer = DataStore()
    writer.open_for_write(str(directory / ('run-' + run_id + '.wandb')))
    record = pb.Record()
    record.run.run_id, record.run.entity, record.run.project = run_id, 'team', 'project'
    writer.write(record)
    for step in (0, 20):
        record = pb.Record()
        item = record.history.item.add()
        item.key, item.value_json = '_step', json.dumps(step)
        writer.write(record)
    writer.close()
    return directory


def main():
    assert wandb.__version__ == '0.18.7', 'Run regression tests with actual affected SDK 0.18.7'
    original_probe = internal_api.Api.server_settings_introspection
    original_error = sender.SendManager._handle_error
    original_worker = SyncThread.run
    api = object.__new__(internal_api.Api)
    api._server_settings_type = None
    api.gql = lambda query: {'ServerSettingsType': None}
    try:
        original_probe(api)
    except Exception as exc:
        assert "'NoneType' object has no attribute 'get'" in str(exc)
    else:
        raise AssertionError('Expected original SDK bug')
    with redirect_stdout(io.StringIO()), sync.legacy_sync_guards():
        for payload, expected in (({'ServerSettingsType': None}, []),
                                  ({'ServerSettingsType': {'fields': [{'name': 'serverMessages'}]}}, ['serverMessages']),
                                  ({}, [])):
            api._server_settings_type = None
            api.gql = lambda query, value=payload: value
            api.server_settings_introspection()
            assert api._server_settings_type == expected
        api._server_settings_type = None
        def denied(query):
            raise RuntimeError('permission denied')
        api.gql = denied
        try:
            api.server_settings_introspection()
        except RuntimeError as exc:
            assert str(exc) == 'permission denied'
        else:
            raise AssertionError('Permission errors must not be suppressed')
    assert internal_api.Api.server_settings_introspection is original_probe
    assert sender.SendManager._handle_error is original_error
    assert SyncThread.run is original_worker
    print('PASS: original null bug reproduced; null/valid/missing schema handled; auth errors retained; patches restored')

    with tempfile.TemporaryDirectory(prefix='gomarl-verified-sync-') as temp:
        root = Path(temp)
        directory = fixture(root)
        binary = directory / 'run-original1.wandb'
        before = binary.read_bytes()
        snapshot = sync.read_snapshot(directory)
        assert snapshot == sync.Snapshot('original1', 'team', 'project', 2, 20)
        remote = SimpleNamespace(id='original1', lastHistoryStep=20)
        with patch.object(wandb, 'Api') as cloud, redirect_stdout(io.StringIO()) as output:
            cloud.return_value.run.return_value = remote
            sync.verify_cloud(snapshot, 'team', 'project', attempts=1)
            cloud.return_value.run.assert_called_with('team/project/original1')
            assert 'VERIFIED cloud=team/project/original1' in output.getvalue()
            assert 'snapshot_step=20' in output.getvalue()
        for value in (SimpleNamespace(id='original1', lastHistoryStep=19),
                      SimpleNamespace(id='wrong-id', lastHistoryStep=30)):
            with patch.object(wandb, 'Api') as cloud:
                cloud.return_value.run.return_value = value
                try:
                    sync.verify_cloud(snapshot, 'team', 'project', attempts=1)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('Wrong ID or stale history must fail')
        with patch.object(wandb, 'Api') as cloud, patch.object(sync.time, 'sleep'):
            cloud.return_value.run.side_effect = [RuntimeError('not found'), remote]
            with redirect_stdout(io.StringIO()):
                sync.verify_cloud(snapshot, 'team', 'project')
            assert cloud.call_count == 2
        with patch.object(wandb, 'Api') as cloud, patch.object(sync.time, 'sleep'):
            cloud.return_value.run.side_effect = RuntimeError('not found')
            try:
                sync.verify_cloud(snapshot, 'team', 'project')
            except RuntimeError:
                assert cloud.call_count == 3
            else:
                raise AssertionError('Missing cloud run must fail')
        print('PASS: real binary parser; original ID; cloud visibility/history progress; bounded retries')

        # Simulate a worker exception that the original CLI silently ignores.
        def broken_worker(worker):
            raise RuntimeError('hidden creation failure')
        def start_worker(**kwargs):
            worker = SyncThread(sync_list=[])
            worker.start()
            worker.join()
        with patch.object(SyncThread, 'run', broken_worker), \
                patch('wandb.cli.cli.cli.main', side_effect=start_worker), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                sync.sync_cli(directory, 'team', 'project')
            except RuntimeError as exc:
                assert 'hidden creation failure' in str(exc)
            else:
                raise AssertionError('Worker error must become process failure')
        # Also expose run_result.error before the SDK prints a misleading link.
        def hidden_run_error(**kwargs):
            manager = object.__new__(sender.SendManager)
            manager._respond_result = lambda result: None
            record, error = pb.Record(), pb.ErrorInfo(message='upsert failed')
            record.control.req_resp = True
            manager._handle_error(record, error, pb.RunRecord(run_id='original1'))
        with patch('wandb.cli.cli.cli.main', side_effect=hidden_run_error), redirect_stdout(io.StringIO()):
            try:
                sync.sync_cli(directory, 'team', 'project')
            except RuntimeError as exc:
                assert 'upsert failed' in str(exc)
            else:
                raise AssertionError('run_result.error must not be ignored')
        assert SyncThread.run is original_worker
        assert sender.SendManager._handle_error is original_error
        print('PASS: real SDK worker/run-result failures are not successful uploads')

        with patch('wandb.cli.cli.cli.main', return_value=0) as cli, redirect_stdout(io.StringIO()):
            sync.sync_cli(directory, 'team', 'project')
            args = cli.call_args.kwargs['args']
            assert '--append' in args and '--no-mark-synced' in args
            assert '--id' not in args and args[-1] == str(directory)
        with patch.object(sync, 'sync_cli'), patch.object(sync, 'verify_cloud', side_effect=RuntimeError('not found')), \
                redirect_stderr(io.StringIO()) as output:
            assert sync.main([str(directory)]) == 1
            assert 'UPLOAD NOT VERIFIED' in output.getvalue()
        with patch.object(sync, 'sync_cli') as upload, patch.object(sync, 'verify_cloud') as verify:
            assert sync.main([str(directory)]) == 0
            upload.assert_called_once_with(directory, 'team', 'project')
            verify.assert_called_once_with(snapshot, 'team', 'project')
        assert binary.read_bytes() == before
        # A writer can leave a partial final header; complete prior steps remain usable.
        with binary.open('ab') as stream:
            stream.write(b'\x01\x02\x03')
        with redirect_stdout(io.StringIO()) as output:
            assert sync.read_snapshot(directory) == snapshot
            assert 'incomplete live tail' in output.getvalue()
        # A checksum failure in an interior block must not be swallowed.
        corrupt = fixture(root, 'corrupt1')
        corrupt_file = corrupt / 'run-corrupt1.wandb'
        damaged = bytearray(corrupt_file.read_bytes())
        damaged[14] ^= 1
        damaged.extend(b'\0' * 100000)
        corrupt_file.write_bytes(damaged)
        try:
            sync.read_snapshot(corrupt)
        except AssertionError:
            pass
        else:
            raise AssertionError('Interior corruption must fail')
        # A different binary ID must not be treated as the original run.
        bad = fixture(root, 'mismatch')
        (bad / 'run-mismatch.wandb').rename(bad / 'run-different.wandb')
        with patch.object(sync, 'sync_cli') as upload, redirect_stderr(io.StringIO()):
            assert sync.main([str(bad)]) == 1
            upload.assert_not_called()
        print('PASS: CLI routing, nonzero verification failure, original bytes/IDs, live tail, corruption/mismatch rejected')


if __name__ == '__main__':
    main()
