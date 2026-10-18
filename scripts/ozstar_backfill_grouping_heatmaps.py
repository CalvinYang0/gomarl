#!/usr/bin/env python3
"""Backfill saved partitions into ORIGINAL W&B runs, without resuming training.

Only group experiments in the four input-control cohorts are selected. Public
API file/summary updates do not change history, run status or performance curves.
No project-wide history download. Dry-run is the default; use --upload to write.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import fcntl
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import sys
import threading
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.grouping_diagnostics import membership


def wanted_plans(repo):
    from ozstar_submit_5m6m_slot_input_controls_12jobs import build_plans as heads
    from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as mixers
    from ozstar_submit_qmix_vdn_calibration_gate_18jobs import build_plans as fresh
    from ozstar_submit_5m6m_score_gradient_ablation_6jobs import build_plans as ablation
    from ozstar_submit_5m6m_qchange_three_ablation_9jobs import build_plans as three_ablation
    return {p['job_name']: p for build in (heads, mixers, fresh, ablation, three_ablation)
            for p in build(repo) if '_group_' in p['job_name']}


def source_runs(wandb_root, wanted):
    """Read bounded metadata only. Never scan training history to find a name."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    for directory in sorted(wandb_root.glob('offline-run-*')):
        if directory.is_symlink():
            continue
        paths = list(directory.glob('run-*.wandb'))
        if len(paths) != 1:
            continue
        scanner = DataStore()
        try:
            scanner.open_for_scan(str(paths[0]))
            for _ in range(128):
                raw = scanner.scan_data()
                if raw is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(raw)
                if record.HasField('run'):
                    if record.run.display_name in wanted:
                        config = {item.key: json.loads(item.value_json) for item in record.run.config.update}
                        run_id = paths[0].stem.removeprefix('run-')
                        if record.run.run_id != run_id:
                            raise ValueError('Run ID/file name mismatch: ' + str(paths[0]))
                        yield dict(id=run_id, name=record.run.display_name, config=config, directory=directory)
                    break
        except (OSError, ValueError) as exc:
            print('UNREADABLE metadata: {}: {}'.format(paths[0], exc), flush=True)
        finally:
            scanner.close()


def snapshot_directories(results, source):
    config = source['config']
    identity = sha256(source['name'].encode()).hexdigest()[:8]
    leaf = 'seed_{}_{}'.format(config['seed'], identity)
    selected = []
    for kind, prefix in (('mixer_input_control', 'mixer_input'), ('slot_input_control', 'slot_input')):
        for directory in sorted((results / kind).glob(config['name'] + '__*/' + leaf)):
            if directory.is_symlink():
                continue
            records = sorted(directory.glob('step_*.json'))
            if not records:
                continue
            data = json.loads(records[-1].read_text())
            explicit = data.get('source_run_id')
            if explicit and explicit != source['id']:
                continue
            if not explicit and source.get('attempts', 1) > 1:
                raise ValueError('Multiple offline attempts; legacy snapshot has no source ID: ' + str(directory))
            if data.get('source_run_name', source['name']) != source['name']:
                raise ValueError('Snapshot/run name mismatch: ' + str(directory))
            membership(data)
            selected.append((directory, prefix))
    if len(selected) > 1:
        raise ValueError('Ambiguous old attempts: multiple snapshot directories for ' + source['name'])
    return selected


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def file_descriptor(path, root, kind):
    raw = path.read_bytes()
    result = dict(_type=kind, path=path.relative_to(root).as_posix(),
                  sha256=sha256(raw).hexdigest(), size=len(raw))
    if kind == 'image-file':
        from PIL import Image
        with Image.open(path) as image:
            result.update(width=image.width, height=image.height, format='png')
    return result


def render_partition(path, names, matrix, title, steps=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(14, len(names) * .2), 6), constrained_layout=True)
    try:
        pixel = ax.imshow(matrix, interpolation='nearest', aspect='auto', vmin=0, vmax=1, cmap='Reds')
        ax.set_xticks(range(len(names)), labels=names, rotation=90, fontsize=6)
        if steps is None:
            ax.set_yticks([0, 1], labels=['important', 'unimportant'])
        else:
            indices = sorted(set([0, len(steps) - 1] + list(range(0, len(steps), max(1, len(steps) // 8)))))
            ax.set_yticks(indices, labels=[str(steps[i]) for i in indices])
            ax.set_ylabel('Saved environment step (rows oldest to newest)')
        ax.set_title(title)
        fig.colorbar(pixel, ax=ax, label='binary membership')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)


def prepare_bundle(directory, prefix, source, output):
    """Exact archived membership only; missing PNGs may be rendered from JSON."""
    import shutil
    root = output / source['id']
    target = root / 'grouping_backfill' / prefix / directory.parent.name
    records, files, json_files, previous_names = [], [], [], None
    for path in sorted(directory.glob('step_*.json')):
        data = json.loads(path.read_text())
        names, selected = membership(data)
        if previous_names is not None and names != previous_names:
            raise ValueError('Slot order changed within one run: ' + str(path))
        previous_names = names
        if data.get('source_run_id') not in (None, source['id']):
            raise ValueError('Mixed source run IDs: ' + str(path))
        if data.get('source_run_name', source['name']) != source['name']:
            raise ValueError('Mixed source run names: ' + str(path))
        json_target, image_target = target / path.name, target / path.with_suffix('.png').name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, json_target)
        original_image = path.with_suffix('.png')
        if original_image.is_file():
            # Large PNGs need no second disk copy on the quota-limited /home.
            if image_target.is_symlink():
                if image_target.resolve() != original_image.resolve():
                    raise ValueError('Unexpected staged media target: ' + str(image_target))
            elif not image_target.exists():
                image_target.symlink_to(original_image.resolve())
            elif image_target.read_bytes() != original_image.read_bytes():
                raise ValueError('Different existing staged media: ' + str(image_target))
        else:
            render_partition(image_target, names, data['masks_or_probability'],
                             source['name'] + ' — recovered saved partition t_env=' + str(data['t_env']))
        data['_selected'], data['_image'] = selected, file_descriptor(image_target, root, 'image-file')
        records.append(data)
        json_files.append(json_target)
        files.append(image_target)
    records.sort(key=lambda r: (r['t_env'], r['train_updates']))
    if not records:
        raise ValueError('No saved partitions: ' + str(directory))
    last = records[-1]
    # Keep every original JSON byte in one deterministic, compressed archive.
    # Hundreds of small HTTP PUTs convey no extra information over this ZIP.
    archive = target / 'grouping_snapshots.zip'
    with zipfile.ZipFile(archive, 'w') as package:
        for path in json_files:
            entry = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            package.writestr(entry, path.read_bytes())
    files.append(archive)
    # Repair original history image references as well, when their native
    # offline media files still exist. Keep the SDK's original relative paths.
    native_root = source.get('directory', Path('/nonexistent-SYNTHETIC-source')) / 'files'
    native_prefixes = ('media/images/' + prefix + '/partition_probability_heatmap_',
                       'media/images/' + prefix + '_partition_probability_heatmap_',
                       'media/images/' + prefix + '/grouping_heatmap_',
                       'media/images/' + prefix + '_grouping_heatmap_')
    for native in sorted((native_root / 'media' / 'images').rglob('*.png')):
        relative = native.relative_to(native_root)
        if not relative.as_posix().startswith(native_prefixes) or not native.is_file():
            continue
        staged = root / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.exists() and not staged.is_symlink():
            staged.symlink_to(native.resolve())
        elif staged.resolve() != native.resolve():
            raise ValueError('Unexpected native-media stage: ' + str(staged))
        files.append(staged)
    names = last.get('state_slot_names', last.get('slot_names'))
    scores = last.get('ranking_ema', last.get('score', [None] * len(names)))
    if len(scores) != len(names):
        raise ValueError('Ranking/slot dimension mismatch')
    tables = {
        'grouping_history': dict(columns=['t_env', 'train_updates', 'important_count', 'heatmap_file'],
            data=[[r['t_env'], r['train_updates'], sum(r['_selected']), r['_image']['path']] for r in records]),
        'grouping_membership': dict(columns=['slot', 'group', 'ranking_score'],
            data=[[name, 'important' if yes else 'unimportant', score]
                  for name, yes, score in zip(names, last['_selected'], scores)]),
    }
    summary = {prefix + '/grouping_heatmap': last['_image'],
               prefix + '/partition_probability_heatmap': last['_image'],
               prefix + '/grouping_t_env': last['t_env'],
               prefix + '/important_slot_names': [n for n, yes in zip(names, last['_selected']) if yes],
               prefix + '/unimportant_slot_names': [n for n, yes in zip(names, last['_selected']) if not yes],
               prefix + '/grouping_backfill_snapshot_count': len(records)}
    summary[prefix + '/grouping_snapshot_archive'] = archive.relative_to(root).as_posix()
    for key, table in tables.items():
        path = target / (key + '.table.json')
        write_json(path, table)
        descriptor = file_descriptor(path, root, 'table-file')
        descriptor.update(ncols=len(table['columns']), nrows=len(table['data']))
        summary[prefix + '/' + key] = descriptor
        files.append(path)
    evolution = target / 'grouping_evolution.png'
    render_partition(evolution, names, [r['_selected'] for r in records],
                     source['name'] + ' — important-group evolution', [r['t_env'] for r in records])
    summary[prefix + '/grouping_evolution'] = file_descriptor(evolution, root, 'image-file')
    files.append(evolution)
    return root, files, summary


def prepare_score_repair_bundle(directory, prefix, source, output):
    """Re-render only the latest saved mixer scores; preserve every source file.

    New media paths avoid overwriting old offline-history image references.
    Only summary images change on upload, not historical training records.
    """
    from utils.mixer_group_visualization import render_group_score_snapshot
    if prefix != 'mixer_input':
        raise ValueError('Score repair supports state mixer snapshots only')
    paths = sorted(directory.glob('step_*.json'))
    if not paths:
        raise ValueError('No saved mixer snapshots')
    path = paths[-1]
    data = json.loads(path.read_text())
    if data.get('source_run_id') not in (None, source['id']):
        raise ValueError('Snapshot/run ID mismatch')
    if data.get('source_run_name', source['name']) != source['name']:
        raise ValueError('Snapshot/run name mismatch')
    root = output / source['id']
    target = root / 'grouping_score_repair_v3' / directory.parent.name
    image = target / (path.stem + '_by_group.png')
    title = source['name'] + '\nt_env=' + str(data['t_env'])
    repaired = render_group_score_snapshot(data, image, title=title)
    record = image.with_suffix('.json')
    repaired['original_snapshot_file'] = str(path)
    repaired['original_snapshot_sha256'] = sha256(path.read_bytes()).hexdigest()
    write_json(record, repaired)
    descriptor = file_descriptor(image, root, 'image-file')
    summary = {prefix + '/grouping_heatmap': descriptor,
               prefix + '/partition_probability_heatmap': descriptor,
               prefix + '/grouping_scores_by_group': descriptor,
               prefix + '/grouping_score_snapshot': record.relative_to(root).as_posix(),
               prefix + '/grouping_score_partition_note': repaired['score_partition_note'],
               prefix + '/grouping_score_t_env': data['t_env'],
               prefix + '/grouping_backfill_snapshot_count': 1}
    return root, [image, record], summary


def cloud_inventory(run, files, root):
    """One paginated metadata query, no downloads or training-history scan."""
    return {f.name: dict(md5=f.md5, size=f.size) for f in
            run.files(names=[p.relative_to(root).as_posix() for p in files], per_page=1000)}


def fingerprints(path):
    raw = path.read_bytes()
    digest = md5(raw).digest()
    return dict(sha256=sha256(raw).hexdigest(), md5=base64.b64encode(digest).decode(),
                md5hex=digest.hex(), size=len(raw))


def matches_cloud(meta, local):
    return bool(meta and meta.get('size') == local['size'] and
                meta.get('md5') in (local['md5'], local['md5hex']))


def reuse_images(bundle, remote):
    """Share one identical PNG per run; prefer already uploaded native media."""
    root, files, summary = bundle
    groups, aliases = {}, {}
    for path in files:
        if path.suffix == '.png':
            info = fingerprints(path)
            groups.setdefault(info['sha256'], []).append((path, info))
    for members in groups.values():
        existing = [p for p, info in members
                    if matches_cloud(remote.get(p.relative_to(root).as_posix()), info)]
        candidates = existing or [p for p, _ in members]
        chosen = min(candidates, key=lambda p: (not p.relative_to(root).as_posix().startswith('media/images/'), str(p)))
        canonical = chosen.relative_to(root).as_posix()
        aliases.update({p.relative_to(root).as_posix(): canonical for p, _ in members
                        if p != chosen})
    for descriptor in summary.values():
        if isinstance(descriptor, dict) and descriptor.get('_type') == 'image-file':
            descriptor['path'] = aliases.get(descriptor['path'], descriptor['path'])
    history_key = next((k for k in summary if k.endswith('/grouping_history')), None)
    if history_key is not None:
        descriptor = summary[history_key]
        path = root / descriptor['path']
        table = json.loads(path.read_text())
        for row in table['data']:
            row[-1] = aliases.get(row[-1], row[-1])
        write_json(path, table)
        descriptor.update(file_descriptor(path, root, 'table-file'))
    filtered = [p for p in files if p.relative_to(root).as_posix() not in aliases]
    return (root, filtered, summary), len(aliases)


class BatchedRunUploader:
    """W&B 0.18.7's own transport, with batched URL requests and reused clients.

    No per-file public File lookup. Each transfer thread keeps its own SDK
    session. Only the main thread requests upload URLs. No run init/resume.
    """
    def __init__(self, project, source_id):
        from wandb.sdk.internal.internal_api import Api
        self.entity, self.project = project.split('/', 1)
        self.run_id = source_id
        self.factory = Api
        self.local = threading.local()
        self.control = self._make_api()

    def _make_api(self):
        # Instance-scoped class setting, consumed during SDK construction.
        # Bound one stalled PUT and retry cycle without changing user settings.
        base = self.factory
        class BoundedApi(base):
            FILE_PUSHER_TIMEOUT = 60
        return BoundedApi(default_settings={'entity': self.entity, 'project': self.project},
                          retry_timedelta=timedelta(seconds=60))

    def urls(self, names):
        _, headers, entries = self.control.upload_urls(self.project, names,
                                                      run=self.run_id, entity=self.entity)
        if set(entries) != set(names):
            raise RuntimeError('Incomplete upload URL batch; no missing file may be acknowledged')
        return headers, entries

    def transfer(self, path, entry, headers):
        if not hasattr(self.local, 'api'):
            self.local.api = self._make_api()
        api = self.local.api
        url = entry['uploadUrl']
        if url.startswith('/'):
            url = api.api_url + url
        extra = dict(header.split(':', 1) for header in headers)
        with path.open('rb') as handle:
            # This is the same retry/check-response operation used by Api.push.
            api.upload_file_retry(url, handle, extra_headers=extra)


def save_ledger(state_path, ledger):
    temporary = state_path.with_suffix('.tmp')
    write_json(temporary, ledger)
    temporary.replace(state_path)


def upload_bundle(api, project, source, bundle, ledger, state_path, workers=8,
                  uploader_factory=BatchedRunUploader, batch_size=64):
    run_path = project + '/' + source['id']
    run = api.run(run_path)
    if run.name != source['name']:
        raise ValueError('Refuse to upload into different cloud run: ' + run_path)
    root, files, summary = bundle
    remote = cloud_inventory(run, files, root)
    bundle, reused = reuse_images(bundle, remote)
    root, files, summary = bundle
    # Isolate the exact source ID and the selected project in the upload ledger.
    done = ledger.setdefault(run_path, {})
    pending, cloud_skipped = [], 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        info = fingerprints(path)
        if matches_cloud(remote.get(relative), info):
            done[relative] = info['sha256']
            cloud_skipped += 1
        else:
            # Cloud existence/checksum, not an old local ledger alone, is
            # authoritative. Repair externally removed or replaced files too.
            pending.append((path, info['sha256']))
    save_ledger(state_path, ledger)
    print('UPLOAD {}: {} files needed; {} already on cloud; {} duplicate PNGs reused'.format(
        source['name'], len(pending), cloud_skipped, reused), flush=True)
    uploaded = 0
    if pending:
        transport = uploader_factory(project, source['id'])
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                names = [p.relative_to(root).as_posix() for p, _ in batch]
                headers, entries = transport.urls(names)
                futures = {pool.submit(transport.transfer, p, entries[p.relative_to(root).as_posix()], headers):
                           (p, digest) for p, digest in batch}
                try:
                    for future in as_completed(futures):
                        future.result()  # Failed PUTs never advance their fingerprint.
                        path, digest = futures[future]
                        done[path.relative_to(root).as_posix()] = digest
                        save_ledger(state_path, ledger)
                        uploaded += 1
                        if uploaded % 10 == 0 or uploaded == len(pending):
                            print('  files acknowledged: {}/{}'.format(uploaded, len(pending)), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()  # Do not drain hundreds of queued uploads on Ctrl+C.
                    raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    # Refresh after uploads. Preserve unrelated summary/performance keys.
    api.flush()
    run = api.run(run_path)
    if run.name != source['name']:
        raise ValueError('Cloud run identity changed')
    if run.state == 'running':
        print('FILES UPLOADED; live run summary deferred: ' + source['name'], flush=True)
        return False
    # No SDK init/resume, no history log, no run-state/config mutation.
    run.summary.update(summary)
    print('BACKFILLED {} snapshots -> {}'.format(
        summary[next(k for k in summary if k.endswith('/grouping_backfill_snapshot_count'))], run.url), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path(os.environ.get('RUNTIME_ROOT',
        '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--refresh-mixer-score-plots', action='store_true',
                        help='Repair latest mixer score image from saved JSON; no history rewrites')
    parser.add_argument('--run-name', action='append',
                        help='Exact submitted run name; repeat to select multiple runs')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    runtime = args.runtime_root
    plans = wanted_plans(Path(__file__).resolve().parents[1])
    if args.run_name:
        unknown = set(args.run_name) - set(plans)
        if unknown:
            parser.error('Unknown submitted run names: ' + ', '.join(sorted(unknown)))
        plans = {name: plans[name] for name in args.run_name}
    if args.refresh_mixer_score_plots:
        plans = {name: plan for name, plan in plans.items() if '_qmix_state_group_' in name}
    seen, failures, completed = set(), 0, 0
    output = runtime / 'grouping_backfill'
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'uploaded_files.json'
    ledger = json.loads(state_path.read_text()) if state_path.exists() else {}
    import wandb
    api = wandb.Api(timeout=60) if args.upload else None
    with (runtime / 'wandb' / '.gomarl-sync-once.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Upload/cleanup currently holds shared lock; retry shortly')
        sources = list(source_runs(runtime / 'wandb', plans))
        for source in sources:
            source['attempts'] = sum(other['name'] == source['name'] for other in sources)
            seen.add(source['name'])
            try:
                plan = plans[source['name']]
                if (source['config'].get('seed') != plan['seed'] or
                        source['config'].get('name') != plan['exports']['GROUP_NAME']):
                    raise ValueError('Offline seed/cohort differs from expected submitted plan')
                dirs = snapshot_directories(runtime / 'results', source)
                if not dirs:
                    print('MISSING saved partitions: ' + source['name'], flush=True)
                    failures += 1
                    continue
                directory, prefix = dirs[0]
                count = len(list(directory.glob('step_*.json')))
                print('{}: {} snapshots -> original run {}'.format(source['name'], count, source['id']), flush=True)
                if args.refresh_mixer_score_plots:
                    bundle = prepare_score_repair_bundle(directory, prefix, source, output)
                    print('RENDERED corrected latest score plot: ' + str(bundle[1][0]), flush=True)
                    if args.upload:
                        completed += int(upload_bundle(api, args.project, source, bundle, ledger, state_path, args.workers))
                elif args.upload:
                    bundle = prepare_bundle(directory, prefix, source, output)
                    completed += int(upload_bundle(api, args.project, source, bundle, ledger, state_path, args.workers))
            except Exception as exc:
                failures += 1
                print('FAILED {}: {}'.format(source['name'], exc), flush=True)
        for name in sorted(set(plans) - seen):
            print('NO LOCAL SOURCE (not run yet or removed): ' + name, flush=True)
        print('Original runs updated={}; missing/failed={}; no local source={}; mode={}'.format(
              completed, failures, len(set(plans) - seen), 'upload' if args.upload else 'dry-run'), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
