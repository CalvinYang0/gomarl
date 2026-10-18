#!/usr/bin/env python3
"""One selectable W&B analysis run per model: unsmoothed three-seed win curves."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import plot_5m6m_8m9m_qmix_variants_3seeds as study
from retained_seed_curves import load_retained_curves


def aggregate_models(curves_path, plans, scenes, target=10000000):
    """Exact cohort/IDs, all three seeds, shared interval, no smoothing/extrapolation."""
    cached = load_retained_curves(curves_path, plans, scenes, study.charts.timestamp)
    models, skipped = [], []
    for scene, labels in study.SCENE_MODELS.items():
        for model in labels:
            selected = sorted((p for p in plans if p['scene'] == scene and p['label'] == model),
                              key=lambda p: p['seed'])
            if not selected:
                continue
            sources = []
            for plan in selected:
                candidates = cached.get(plan['job_name'], [])
                if not candidates:
                    continue
                candidate = max(candidates, key=lambda item: item[0])
                x, y = candidate[3]['curve']
                if not x.size:
                    continue
                sources.append((plan, candidate, x, y))
            if {p['seed'] for p, _, _, _ in sources} != {1, 2, 3}:
                skipped.append(dict(scene=scene, model=model, reason='requires seeds 1/2/3',
                                    available_seeds=[p['seed'] for p, _, _, _ in sources]))
                continue
            start = max(x[0] for _, _, x, _ in sources)
            end = min(target, min(x[-1] for _, _, x, _ in sources))
            grid = np.arange(np.ceil(start / 10000) * 10000, end + .5, 10000)
            if not grid.size:
                skipped.append(dict(scene=scene, model=model, reason='no shared step interval'))
                continue
            aligned = np.stack([np.interp(grid, x, y) for _, _, x, y in sources])
            mean, std = aligned.mean(0), aligned.std(0, ddof=1)
            rows = []
            for i, step in enumerate(grid):
                row = dict(env_steps=int(step), test_battle_won_mean=float(mean[i]),
                           test_battle_won_mean_std=float(std[i]),
                           test_battle_won_mean_lower=float(max(0, mean[i] - std[i])),
                           test_battle_won_mean_upper=float(min(1, mean[i] + std[i])), seed_count=3)
                row.update({'seed_{}/test_battle_won_mean'.format(seed): float(aligned[seed - 1, i])
                            for seed in (1, 2, 3)})
                rows.append(row)
            provenance = [dict(seed=p['seed'], run_name=p['job_name'], run_id=candidate[1],
                               attempt_start=candidate[0], source_end_step=float(x[-1]))
                          for p, candidate, x, _ in sources]
            models.append(dict(scene=scene, model=model, label=study.LABELS[model][0],
                               sources=provenance, rows=rows, source_csv=str(Path(curves_path).resolve())))
    return models, skipped


def upload_model(record, project, output, wandb):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(version=1, project=project, scene=record['scene'], model=record['model'],
                    sources=record['sources'], rows=record['rows'])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Content-addressed snapshot: unchanged retries resume; changed histories get
    # a new analysis snapshot instead of silently appending revised old means.
    rid = 'mean3' + digest[:20]
    basename = record['scene'] + '__' + record['model'] + '__' + digest[:8]
    csv_path, metadata = output / (basename + '.csv'), output / (basename + '.json')
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record['rows'][0]))
        writer.writeheader()
        writer.writerows(record['rows'])
    config = dict(scene=record['scene'], model=record['model'], label=record['label'],
                  analysis_only=True, aggregated_seeds=[1, 2, 3], sources=record['sources'],
                  source_csv=record['source_csv'], smoothing='none', win_units='fraction [0,1]',
                  alignment='10k env-step grid; linear interpolation; shared seed interval only',
                  std_definition='sample std across three seeds, ddof=1; not a confidence interval',
                  source_digest=digest, start_step=record['rows'][0]['env_steps'],
                  end_step=record['rows'][-1]['env_steps'])
    metadata.write_text(json.dumps(config, indent=2))
    entity, project_name = project.split('/', 1)
    with wandb.init(entity=entity, project=project_name, id=rid, resume='allow', mode='online',
                    name=record['scene'].removeprefix('smac_') + ' | ' + record['model'] + ' | mean3 | ' + digest[:6],
                    group=record['scene'] + '_three_seed_scalars', job_type='three-seed-aggregate',
                    tags=['three-seed-aggregate', 'not-training', record['scene']],
                    config=config, dir=str(output)) as run:
        run.define_metric('env_steps')
        run.define_metric('*', step_metric='env_steps')
        next_step = run.step
        for row in record['rows']:
            if row['env_steps'] >= next_step:
                run.log(row, step=row['env_steps'])
        artifact = wandb.Artifact(basename, type='three-seed-scalar-data')
        artifact.add_file(str(csv_path))
        artifact.add_file(str(metadata))
        run.log_artifact(artifact)
        url = run.url
    print('UPLOADED MEAN3: {} {} -> {}'.format(record['scene'], record['model'], url), flush=True)
    return dict(scene=record['scene'], model=record['model'], run_id=rid, run_url=url,
                source_digest=digest, points=len(record['rows']),
                end_step=record['rows'][-1]['env_steps'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path(os.environ.get(
        'RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')))
    parser.add_argument('--project', default='hjh331-sjtu/gomarl')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args(argv)
    study.configure()
    plans = study.build_plans(study.charts.ROOT)
    records, skipped = [], []
    for map_name in ('5m6m', '8m9m'):
        path = args.runtime_root / 'figures' / (map_name + '_local_available_qmix_variants_3seeds') / 'seed_curves.csv'
        selected = [p for p in plans if p['scene'] == 'smac_' + map_name]
        models, missing = aggregate_models(path, selected, study.charts.SCENES)
        records.extend(models)
        skipped.extend(missing)
    print('Scalar upload plan: {} models with 3 seeds; {} skipped'.format(len(records), len(skipped)), flush=True)
    for missing in skipped:
        print('SKIP MEAN3: ' + json.dumps(missing), flush=True)
    output = args.runtime_root / 'figures/three_seed_scalar_runs'
    output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    if not args.no_upload:
        import wandb
        for record in records:
            try:
                results.append(upload_model(record, args.project, output, wandb))
            except Exception as exc:
                failures.append(dict(scene=record['scene'], model=record['model'], error=str(exc)))
                print('FAILED MEAN3: {} {} ({})'.format(record['scene'], record['model'], type(exc).__name__), flush=True)
    report = dict(uploaded=results, skipped=skipped, failed=failures, dry_run=args.no_upload)
    (output / 'upload_report.json').write_text(json.dumps(report, indent=2))
    if failures:
        raise SystemExit('Some scalar uploads failed; see ' + str(output / 'upload_report.json'))
    return report


if __name__ == '__main__':
    main()
