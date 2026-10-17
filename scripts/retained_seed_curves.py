"""Read explicitly selected raw seed CSV snapshots with their exact-run inventory."""
import csv
from pathlib import Path


def load_retained_curves(path, plans, scenes, timestamp):
    """Never infer a cohort from a model label or concatenate run attempts."""
    import numpy as np

    path = Path(path).resolve()
    inventory_path = path.with_name('seed_inventory.csv')
    wanted = {p['job_name']: p for p in plans}
    metadata = {}
    with inventory_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'scene', 'model', 'seed', 'run_name', 'run_id', 'attempt_start'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Retained inventory lacks exact-run provenance: ' + str(inventory_path))
        for row in reader:
            plan = wanted.get(row['run_name'])
            if plan is None:
                continue
            if (row['scene'] != plan['scene'] or row['model'] != plan['label'] or
                    int(row['seed']) != plan['seed']):
                raise ValueError('Retained inventory identity mismatch: ' + row['run_name'])
            if not row['run_id']:
                continue
            if not row['attempt_start'] or not timestamp(row['attempt_start']):
                raise ValueError('Retained attempt start is unknown: ' + row['run_name'])
            if scenes[plan['scene']][1] != 'test_battle_won_mean':
                raise ValueError('Retained win_fraction CSV only supports SMAC test-win histories')
            key = (row['scene'], row['model'], int(row['seed']), row['run_id'])
            if key in metadata:
                raise ValueError('Duplicate retained inventory identity: ' + repr(key))
            metadata[key] = row

    points = {key: {} for key in metadata}
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'scene', 'model', 'seed', 'run_id', 'step', 'win_fraction'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Not a raw seed_curves CSV: ' + str(path))
        for row in reader:
            key = (row['scene'], row['model'], int(row['seed']), row['run_id'])
            if key not in points:
                continue
            step, value = float(row['step']), float(row['win_fraction'])
            if not np.isfinite(step) or not np.isfinite(value) or step < 0 or not 0 <= value <= 1:
                raise ValueError('Invalid retained test-win point: ' + repr(key))
            previous = points[key].get(step)
            if previous is not None and previous != value:
                raise ValueError('Conflicting retained values for one step: ' + repr(key))
            points[key][step] = value

    result = {}
    for key, values in points.items():
        row = metadata[key]
        steps = sorted(values)
        curve = (np.asarray(steps), np.asarray([values[step] for step in steps]))
        payload = dict(curve=curve, csv_path=str(path), inventory_path=str(inventory_path))
        result.setdefault(row['run_name'], []).append(
            (timestamp(row['attempt_start']), row['run_id'], 'retained_csv', payload))
    return result


def load_retained_directory(root, output, plans, scenes, timestamp):
    """Index local snapshots; duplicate copies remain one original run, never extra seeds."""
    import numpy as np

    by_run = {}
    for path in sorted(Path(root).glob('*/seed_curves.csv')):
        if path.parent.resolve() == Path(output).resolve():
            continue
        if not path.with_name('seed_inventory.csv').is_file():
            continue
        try:
            found = load_retained_curves(path, plans, scenes, timestamp)
        except (OSError, ValueError, KeyError) as exc:
            print('SKIP unverifiable local CSV {}: {}'.format(path, exc), flush=True)
            continue
        if not found:
            continue
        print('LOCAL CSV: {} ({} exact names)'.format(path, len(found)), flush=True)
        for name, candidates in found.items():
            for candidate in candidates:
                key = (name, candidate[1])
                previous = by_run.get(key)
                if previous is not None:
                    if previous[0] != candidate[0]:
                        raise ValueError('Conflicting attempt times for retained run: ' + repr(key))
                    old_x, old_y = previous[3]['curve']
                    new_x, new_y = candidate[3]['curve']
                    _, a, b = np.intersect1d(old_x, new_x, return_indices=True)
                    if not np.array_equal(old_y[a], new_y[b]):
                        raise ValueError('Conflicting local copies of retained run: ' + repr(key))
                    # Select one intact snapshot, do not stitch their time ranges.
                    quality = lambda item: (item[3]['curve'][0][-1]
                                            if len(item[3]['curve'][0]) else -1,
                                            len(item[3]['curve'][0]))
                    if quality(candidate) <= quality(previous):
                        continue
                by_run[key] = candidate
    result = {}
    for (name, _), candidate in by_run.items():
        result.setdefault(name, []).append(candidate)
    return result
