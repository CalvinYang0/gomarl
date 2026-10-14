#!/usr/bin/env python3
"""Offline SDK and fake public API tests. Never uploads experiment data."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import threading
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
import wandb
from PIL import Image

import ozstar_backfill_grouping_heatmaps as study
from utils.grouping_diagnostics import membership


def main():
    assert len(study.wanted_plans(Path(__file__).resolve().parents[1])) == 30
    with tempfile.TemporaryDirectory(prefix='gomarl-group-backfill-QA-') as temp:
        root = Path(temp)
        sdk_image_path = root / 'SYNTHETIC_SDK_IMAGE.png'
        Image.new('RGB', (50, 30), 'red').save(sdk_image_path)
        # Real W&B 0.18.7 metadata, not a hand-invented run record.
        with patch('wandb.sdk.data_types.base_types.wb_value._server_accepts_client_ids', return_value=False), \
                patch('requests.sessions.Session.request', side_effect=AssertionError('Offline QA must not access network')), \
                wandb.init(mode='offline', dir=temp, name='SYNTHETIC_GROUP_BACKFILL_QA',
                        config={'seed': 1, 'name': 'SYNTHETIC_GROUP'},
                        settings=wandb.Settings(console='off')) as offline:
            run_id = offline.id
            offline.log({'SYNTHETIC_TABLE': wandb.Table(columns=['step', 'heatmap_file'],
                data=[[10, 'SYNTHETIC_SDK_IMAGE.png']]),
                'SYNTHETIC_IMAGE': wandb.Image(str(sdk_image_path))})
            sdk_dir = Path(offline.dir)
        sdk_table = json.loads(next((sdk_dir / 'media' / 'table').glob('*.table.json')).read_text())
        assert sdk_table['data'][0] == [10, 'SYNTHETIC_SDK_IMAGE.png']
        sources = list(study.source_runs(root / 'wandb', {'SYNTHETIC_GROUP_BACKFILL_QA'}))
        assert len(sources) == 1 and sources[0]['id'] == run_id
        source = sources[0]
        native_image = sdk_dir / 'media' / 'images' / 'mixer_input' / 'partition_probability_heatmap_10_QA.png'
        native_image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (50, 30), 'red').save(native_image)
        directory = root / 'results' / 'mixer_input_control' / 'SYNTHETIC_GROUP__2026-10-14_00-00-00' / (
            'seed_1_' + sha256(source['name'].encode()).hexdigest()[:8])
        directory.mkdir(parents=True)
        for step, rows in ((10, [[1, 0], [0, 1]]), (20, [[1, 1], [0, 0]])):
            path = directory / ('step_{:09d}_update_{:07d}.json'.format(step, step))
            study.write_json(path, dict(t_env=step, train_updates=step, variant='group_qchange',
                state_slot_names=['health', 'cooldown'], row_labels=['important membership', 'unimportant membership'],
                masks_or_probability=rows, ranking_ema=[.8, .2]))
        # One original PNG, one genuinely recovered archived mask.
        Image.new('RGB', (50, 30), 'red').save(sorted(directory.glob('*.json'))[0].with_suffix('.png'))
        assert study.snapshot_directories(root / 'results', source) == [(directory, 'mixer_input')]
        try:
            study.snapshot_directories(root / 'results', dict(source, attempts=2))
            raise AssertionError('Ambiguous attempts must fail closed')
        except ValueError:
            pass
        bundle = study.prepare_bundle(directory, 'mixer_input', source, root / 'output')
        bundle_root, files, summary = bundle
        assert len(files) == 7 and summary['mixer_input/grouping_t_env'] == 20
        archive = bundle_root / summary['mixer_input/grouping_snapshot_archive']
        with zipfile.ZipFile(archive) as package:
            assert sorted(package.namelist()) == sorted(p.name for p in directory.glob('*.json'))
            assert all(package.read(p.name) == p.read_bytes() for p in directory.glob('*.json'))
        assert (bundle_root / native_image.relative_to(sdk_dir)).is_symlink()
        assert summary['mixer_input/important_slot_names'] == ['health', 'cooldown']
        assert summary['mixer_input/unimportant_slot_names'] == []
        history = summary['mixer_input/grouping_history']
        table = json.loads((bundle_root / history['path']).read_text())
        assert [row[0] for row in table['data']] == [10, 20]
        assert all((bundle_root / row[-1]).is_file() for row in table['data'])
        assert wandb.Table.from_json(table, None).data == table['data']
        member = json.loads((bundle_root / summary['mixer_input/grouping_membership']['path']).read_text())
        assert wandb.Table.from_json(member, None).data == member['data']
        # SDK-compatible media descriptors: paths and digest exist on disk.
        for descriptor in summary.values():
            if isinstance(descriptor, dict) and 'path' in descriptor:
                raw = (bundle_root / descriptor['path']).read_bytes()
                assert sha256(raw).hexdigest() == descriptor['sha256']
        # Use the actual old/public summary encoder against a fake GraphQL
        # transport to verify nested image/table metadata and unrelated keys.
        from wandb.old.summary import HTTPSummary
        updates = []
        public = SimpleNamespace(dir=str(root), storage_id='SYNTHETIC_STORAGE_ID',
                                 path=['SYNTHETIC', 'QA', run_id])
        def execute(mutation, variable_values):
            updates.append(json.loads(variable_values['summaryMetrics']))
            return {'upsertBucket': {'bucket': {'id': 'SYNTHETIC_STORAGE_ID'}}}
        summary_api = HTTPSummary(public, SimpleNamespace(execute=execute), summary={'test_return_mean': 11.})
        summary_api.update(summary)
        assert updates[-1]['test_return_mean'] == 11.
        assert updates[-1]['mixer_input/grouping_history']['_type'] == 'table-file'
        uploaded, ledger, remote, batches = [], {}, {}, []
        transfer_lock = threading.Lock()
        original = {'test_battle_won_mean': .75, '_step': 10050000}
        run = SimpleNamespace(name=source['name'], state='finished', summary=original.copy(),
                              url='https://example.invalid/SYNTHETIC_QA')
        def forbid_individual(*a, **kw):
            raise AssertionError('The slow per-file public API must never be called')
        run.upload_file = forbid_individual
        run.files = lambda names, per_page: [SimpleNamespace(name=name, md5=remote[name]['md5'], size=remote[name]['size'])
                                            for name in names if name in remote]
        class FakeTransport:
            def __init__(self, project, source_id):
                assert project == 'SYNTHETIC/QA' and source_id == run_id
            def urls(self, names):
                batches.append(names)
                return [], {name: {'name': name, 'uploadUrl': 'https://example.invalid/QA'} for name in names}
            def transfer(self, path, entry, headers):
                with transfer_lock:
                    uploaded.append(entry['name'])
                    remote[entry['name']] = study.fingerprints(path)
        api = SimpleNamespace(run=lambda path: run, flush=lambda: None)
        state = root / 'ledger.json'
        assert study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state,
                                   workers=2, uploader_factory=FakeTransport, batch_size=3)
        assert len(uploaded) == 6 and len(batches) == 2
        assert all(run.summary[k] == v for k, v in original.items())
        study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state, uploader_factory=FakeTransport)
        assert len(uploaded) == 6  # Idempotent uploads, validated against cloud.
        # Restart can reuse native images uploaded by ordinary sync, even with
        # no backfill ledger. Only the ZIP, two tables, and overview need PUTs.
        remote = {name: info for name, info in remote.items() if name.endswith('.png') and
                  not name.endswith('grouping_evolution.png')}
        ledger.clear()
        study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state, uploader_factory=FakeTransport)
        assert len(uploaded) == 10
        # A stale local success cannot hide a remotely deleted file.
        del remote[summary['mixer_input/grouping_snapshot_archive']]
        study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state, uploader_factory=FakeTransport)
        assert len(uploaded) == 11
        run.state = 'running'
        assert not study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state, uploader_factory=FakeTransport)
        run.name = 'WRONG_RUN'
        try:
            study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state, uploader_factory=FakeTransport)
            raise AssertionError('Cloud identity mismatch must fail closed')
        except ValueError:
            pass
        run.name, run.state = source['name'], 'finished'
        ledger.clear()
        remote.clear()
        class FailedTransport(FakeTransport):
            def transfer(self, *a, **kw):
                raise RuntimeError('Synthetic upload failure')
        try:
            study.upload_bundle(api, 'SYNTHETIC/QA', source, bundle, ledger, state, workers=1,
                                uploader_factory=FailedTransport)
            raise AssertionError('Failed uploads must propagate')
        except RuntimeError:
            assert ledger['SYNTHETIC/QA/' + run_id] == {}
        try:
            membership(dict(state_slot_names=['a'], row_labels=['important'], masks_or_probability=[[1], [1]]))
            raise AssertionError('Overlapping groups must fail')
        except ValueError:
            pass
        # Actual uploader wiring: batched URL scope/headers and relative URLs.
        calls = []
        class FakeBackend:
            api_url = 'https://example.invalid'
            def upload_urls(self, project, names, run, entity):
                calls.append(('urls', project, names, run, entity))
                return 'QA-storage', ['Content-Type:image/png'], {name: {'uploadUrl': '/QA-put'} for name in names}
            def upload_file_retry(self, url, handle, extra_headers):
                calls.append(('put', url, handle.read(), extra_headers))
        with patch.object(study.BatchedRunUploader, '_make_api', return_value=FakeBackend()):
            transport = study.BatchedRunUploader('SYNTHETIC/QA', run_id)
            headers, entries = transport.urls(['a.png', 'b.png'])
            transport.transfer(sdk_image_path, entries['a.png'], headers)
            assert calls[0] == ('urls', 'QA', ['a.png', 'b.png'], run_id, 'SYNTHETIC')
            assert calls[1][1] == 'https://example.invalid/QA-put'
            assert calls[1][3] == {'Content-Type': 'image/png'}
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Offline QA must not access network')):
            actual = study.BatchedRunUploader('SYNTHETIC/QA', run_id)
            assert actual.control.retry_timedelta.total_seconds() == 60
            assert actual.control.FILE_PUSHER_TIMEOUT == 60
            assert actual.control._upload_file_session.put.keywords['timeout'] == 60
    print('PASS: original run/history safety; exact archived JSON ZIP; batch transport; cloud checksum reuse; PNG dedup; restart; removed-file repair; failure ledger; live-summary guard')


if __name__ == '__main__':
    main()
