"""Readable per-model seed images and an explicitly sectioned W&B workspace."""
import json
from pathlib import Path
import textwrap


def render_model_panels(scene, entries, output_dir, plotter, window, labels, models, target_steps):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    section = scene.removeprefix('smac_')
    records = []
    for index, model in enumerate(models, 1):
        label = labels[model][0]
        fig, axis = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
        shown, ends = set(), []
        for seed, (x, y) in entries.get(model, []):
            keep = x <= target_steps
            x, y = x[keep], y[keep]
            if not len(x):
                continue
            shown.add(seed)
            ends.append(float(x[-1]))
            color = 'C' + str(seed - 1)
            axis.plot(x, y * 100, alpha=.16, linewidth=.6, color=color)
            axis.plot(x, plotter.centered_rolling_mean(y, window) * 100,
                      linewidth=2, color=color, label='seed {} (to {:.2f}M)'.format(seed, x[-1] / 1e6))
        axis.set(title='{} — {}\n{}'.format(section, '\n'.join(textwrap.wrap(label, 62)),
                                          'raw: faint; smoothed: solid; window={}'.format(window)),
                 xlabel='Environment steps', ylabel='Test win rate (%)',
                 xlim=(0, target_steps), ylim=(0, 100))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:g}M'.format(v / 1e6)))
        axis.tick_params(labelsize=11)
        axis.grid(alpha=.3, linestyle='--')
        if shown:
            axis.legend(frameon=False, fontsize=11, loc='lower right')
        missing = sorted({1, 2, 3} - shown)
        if missing:
            axis.text(.5, .08, 'Missing test history: seed ' + ', '.join(map(str, missing)),
                      transform=axis.transAxes, ha='center', fontsize=11)
        filename = '{}__{:02d}_{}_seeds.png'.format(section, index, model)
        path = Path(output_dir) / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        records.append(dict(section=section, model=model, title=label, path=str(path),
                            key='{}/{:02d}_{}'.format(section, index, model),
                            seeds=sorted(shown), end_steps=ends))
    return records


def make_workspace(entity, project, run_id, records, summaries):
    # Pin to a Python-3.9-compatible official release on OzSTAR (0.4.5).
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    sections = [ws.Section(name='Media', is_open=False,
        layout_settings=ws.SectionLayoutSettings(columns=1, rows=2),
        panels=[wr.MediaBrowser(title=title, media_keys=[key], num_columns=1)
                for title, key in summaries])]
    names = [name for name in ('5m6m', '8m9m') if any(r['section'] == name for r in records)]
    for name in names:
        panels = [wr.MediaBrowser(title=r['title'], media_keys=[r['key']], num_columns=1)
                  for r in records if r['section'] == name]
        sections.append(ws.Section(name=name, is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=1, rows=max(1, len(panels))),
            panels=panels))
    return ws.Workspace(entity=entity, project=project,
        name='Three-seed models: ' + ' and '.join(names) + ' (' + run_id + ')', sections=sections,
        auto_generate_panels=False,
        runset_settings=ws.RunsetSettings(filters=[expr.Metric('id') == run_id]))


def upload_sectioned(project, output_dir, outputs, inventory, window, charts, records):
    # Check layout support before uploading, avoiding an apparently successful
    # upload with the old unreadable default Media layout.
    try:
        import wandb_workspaces.workspaces  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install layout support first: python -m pip install --no-deps wandb-workspaces==0.4.5') from exc
    import wandb
    from wandb.sdk.lib.runid import generate_id
    entity, project_name = project.split('/', 1)
    run_id = generate_id()
    summaries = [(scene.removeprefix('smac_') + ' mean ± std', 'Media/' + scene + '_mean_std')
                 for scene in outputs]
    with wandb.init(entity=entity, project=project_name, id=run_id, resume='never',
                    name=charts.GROUP + '_figures', group=charts.GROUP, job_type='analysis', mode='online',
                    dir=str(output_dir), config=dict(target_steps=charts.TARGET_STEPS, mean_window=window,
                        selection='latest attempt per seed', seeds=[1, 2, 3],
                        individual_seed_layout='one readable image per model; Media/5m6m/8m9m sections')) as run:
        payload = {'seed_inventory': wandb.Table(columns=list(charts.FIELDS),
                   data=charts.inventory_table_data(inventory))}
        for (_, key), (scene, paths) in zip(summaries, outputs.items()):
            payload[key] = wandb.Image(str(paths[0]))
        for record in records:
            payload[record['key']] = wandb.Image(record['path'], caption=record['title'])
        run.log(payload)
        artifact = wandb.Artifact(charts.GROUP + '_available_seed_data', type='dataset')
        for path in sorted(Path(output_dir).glob('*.csv')):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run_url = run.url
    # The run is flushed/finished before the workspace references its images.
    saved = dict(project=project, run_id=run_id, run_url=run_url, panels=records, summaries=summaries)
    metadata = Path(output_dir) / 'model_panel_upload.json'
    metadata.write_text(json.dumps(saved, indent=2))
    print('Uploaded figures: ' + run_url, flush=True)
    try:
        workspace = make_workspace(entity, project_name, run_id, records, summaries).save()
    except Exception as exc:
        print('Images uploaded, but workspace layout failed: ' + str(exc), flush=True)
        print('Retry layout without re-upload: python scripts/wandb_seed_figure_sections.py ' + str(metadata), flush=True)
        raise
    saved['workspace_url'] = workspace.url
    metadata.write_text(json.dumps(saved, indent=2))
    print('Model sections workspace: ' + workspace.url, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retry only the W&B model section layout, not data upload.')
    parser.add_argument('metadata', type=Path)
    args = parser.parse_args()
    data = json.loads(args.metadata.read_text())
    entity, project = data['project'].split('/', 1)
    workspace = make_workspace(entity, project, data['run_id'], data['panels'], data['summaries']).save()
    data['workspace_url'] = workspace.url
    args.metadata.write_text(json.dumps(data, indent=2))
    print('Model sections workspace: ' + workspace.url)
