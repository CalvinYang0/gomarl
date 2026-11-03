#!/usr/bin/env python3
"""Audit latest retry of the six 3s5z variants; originals + full10M mean3 only.

No training, cleanup, videos, tmux or project-wide cloud scan. --status-only is
read-only. Receipt candidates are resolved by sacct Submit, never best outcome.
"""
from datetime import datetime
from pathlib import Path
import re
from unittest.mock import patch

import ozstar_submit_3s5z_soft_refresh_live_9jobs as refresh
import ozstar_submit_3s5z_onefill_warmup_9jobs as warmup
import ozstar_submit_3s5z_refresh_and_warmup_18jobs as cohort
import ozstar_upload_recent_21jobs_20261028 as uploader

GROUP = '3s5z_refresh_warmup_backfill_20261103'
LABELS = {
    refresh.MODES['eval_live'][0]: ('Soft one-fill p0=.1 / refresh2M / live evaluation', '#1f77b4'),
    refresh.MODES['faster_stages'][0]: ('Soft one-fill p0=.1 / faster stage transitions / live by8M', '#ff7f0e'),
    refresh.MODES['larger_jumps'][0]: ('Soft one-fill p0=.1 / larger refresh jumps / live by8M', '#2ca02c'),
    warmup.MODES['initial1'][0]: ('Projected one-fill / exact p0=1 / trainable throughout', '#d62728'),
    warmup.MODES['freeze5m'][0]: ('Sigmoid one-fill p0=.1 / freeze binary p>.9 at5M', '#9467bd'),
    warmup.MODES['freeze5m_refresh'][0]: ('Sigmoid one-fill p0=.1 / freeze5M + refresh2M', '#8c564b'),
}


def build_plans(repo):
    plans = cohort.build_plans(repo)
    if (len(plans) != 18 or {p['scene'] for p in plans} != {'smac_3s5z'} or
            {p['label'] for p in plans} != set(LABELS) or
            any({p['seed'] for p in plans if p['label'] == m} != {1, 2, 3} for m in LABELS)):
        raise RuntimeError('Expected six exact 3s5z models x seeds1/2/3')
    return plans


def detailed_accounting(jobs, repo):
    if not jobs:
        return {}
    output = uploader.sync.command(['sacct', '-j', ','.join(sorted(jobs, key=int)), '-S', '2026-11-01',
        '-X', '-n', '-P', '--format=JobIDRaw,State,ExitCode,Elapsed,WorkDir%4096,Submit,Start,End,JobName%256'])
    result = {}
    for line in output.splitlines():
        fields = [x.strip() for x in line.rstrip('|').split('|')]
        if len(fields) != 9 or fields[0] not in jobs:
            continue
        job, state, exit_code, elapsed, workdir, submit, start, end, name = fields
        if Path(workdir) != repo or name != jobs[job]:
            continue
        try:
            datetime.fromisoformat(submit)
        except ValueError:
            continue
        row = dict(state=state.split()[0].rstrip('+'), exit_code=exit_code, elapsed=elapsed,
            submitted=submit, started=start, ended=end)
        if job in result and result[job] != row:
            raise RuntimeError('Conflicting sacct rows: '+job)
        result[job] = row
    return result


def audit_latest(repo, runtime, plans, manifest_reader, chosen):
    expected = {p['job_name'] for p in plans}
    try:
        jobs = {j: n for j, n in manifest_reader(runtime/'ozstar_logs').items() if n in expected}
    except RuntimeError as exc:
        if 'No submitted/retained job IDs' not in str(exc):
            raise
        jobs = {}
    states = detailed_accounting(jobs, repo)
    rows = []
    chosen.clear()
    for p in plans:
        ids = sorted((j for j, n in jobs.items() if n == p['job_name']), key=int)
        row = dict(name=p['job_name'], scene=p['scene'], model=p['label'], seed=p['seed'],
            job_ids=[], all_job_ids=ids, older_attempts=[])
        if not ids:
            row.update(state='MISSING_RECEIPT', exit_code='?')
        elif any(j not in states for j in ids):
            # Missing a candidate's submission time must not fall back to an old
            # successful attempt, or hide an unverified/requeued newer attempt.
            row.update(state='UNKNOWN_ACCOUNTING_OR_WORKDIR', exit_code='?')
        else:
            latest = max(ids, key=lambda j: (states[j]['submitted'], int(j)))
            row.update(states[latest], job_ids=[latest])
            row['older_attempts'] = [dict(job=j, **states[j]) for j in ids if j != latest]
            chosen[latest] = p['job_name']
        rows.append(row)
        print('{} {:20} exit={} elapsed={} older={} {}'.format(
            ','.join(row['job_ids']) or '-', row['state'], row['exit_code'], row.get('elapsed', '?'),
            len(row['older_attempts']), row['name']), flush=True)
    return rows


def run_ownership(repo, runtime, jobs):
    """Prove W&B IDs and Sacred paths from ONLY selected exact job logs."""
    index = uploader.sync.job_index(runtime/'ozstar_logs', runtime/'wandb', repo, job_ids=jobs.keys())
    result = {name: dict(ids=set(), sacred=set()) for name in jobs.values()}
    for dirname, ids in index.items():
        rid = dirname.rsplit('-', 1)[-1]
        for job in ids & jobs.keys():
            result[jobs[job]]['ids'].add(rid)
    for job, name in jobs.items():
        texts = []
        for suffix in ('out', 'err'):
            log = runtime/'ozstar_logs'/('{}_{}.{}'.format(name, job, suffix))
            if log.is_file() and not log.is_symlink():
                texts.append(log.read_text(errors='replace'))
        text = re.sub(r'\x1b\[[0-9;]*m', '', '\n'.join(texts))
        roots = re.findall(r'Saving to FileStorageObserver in (.+?)\.\s*$', text, re.M)
        sacred_ids = re.findall(r'Started run with ID [\"\x27]?(\d+)', text)
        if len(set(roots)) == 1 and len(set(sacred_ids)) == 1:
            path = Path(roots[0])/sacred_ids[0]
            if path.is_absolute():
                path = path.resolve()
                if (runtime/'results/sacred').resolve() in path.parents:
                    result[name]['sacred'].add(str(path))
    return result


def strict_render(runtime, project, output, plans, chosen, base_render):
    charts = uploader.runner.study.charts
    ownership = run_ownership(charts.ROOT, runtime, {j: n for j, n in chosen.items()
        if n in {p['job_name'] for p in plans}})
    original_select = charts.select_latest
    cloud_cache = {}

    def select(plan, metric, local, cloud, sacred, wandb_root, plotter, retained=None):
        name = plan['job_name']
        owned = ownership.get(name, dict(ids=set(), sacred=set()))
        allowed = owned['ids'] | {'sacred:'+p for p in owned['sacred']}
        a = {name: [p for p in local.get(name, []) if p.stem.removeprefix('run-') in owned['ids']]}
        b = {name: [r for r in cloud.get(name, []) if r.id in owned['ids']]}
        c = {name: [(t, p) for t, p in sacred.get(name, []) if str(p) in owned['sacred']]}
        # Sacred numeric directories can be reused after a full local cleanup;
        # only immutable W&B IDs may authenticate a retained CSV on their own.
        d = {name: [r for r in (retained or {}).get(name, []) if r[1] in owned['ids']]}
        if not any((a[name], b[name], c[name], d[name])) and owned['ids']:
            # Local sources were cleaned: query only proven original IDs, never
            # a whole-project search or a same-name old attempt.
            import wandb
            if 'api' not in cloud_cache:
                cloud_cache['api'] = wandb.Api(timeout=30)
            for rid in sorted(owned['ids']):
                r = cloud_cache['api'].run(project+'/'+rid)
                if r.name != name:
                    raise RuntimeError('Cloud original identity differs from selected job: '+rid)
                b[name].append(r)
        if not allowed:
            raise RuntimeError('No exact selected-job W&B ID or Sacred path in job logs: '+name)
        return original_select(plan, metric, a, b, c, wandb_root, plotter, d)

    with patch.object(charts, 'select_latest', side_effect=select):
        record, inventory = base_render(runtime, project, output, plans)
    if any(row['run_id'] not in (ownership[row['run_name']]['ids'] |
            {'sacred:'+p for p in ownership[row['run_name']]['sacred']}) for row in inventory):
        raise RuntimeError('Rendered curve is not owned by the selected latest job')
    return record, inventory


def main(argv=None):
    build_plans(Path(__file__).resolve().parents[1])
    chosen = {}
    original_manifest = uploader.sync.manifest_jobs
    original_render = uploader.render_model

    def selected_manifest(logs):
        # This is called by raw sync after audit_latest; no old failed retries.
        return dict(chosen)

    with patch.multiple(uploader, GROUP=GROUP, SOURCES=(refresh, warmup), LABELS=LABELS,
            build_plans=build_plans,
            audit_jobs=lambda repo, runtime, plans: audit_latest(repo, runtime, plans,
                original_manifest, chosen),
            render_model=lambda runtime, project, output, plans: strict_render(
                runtime, project, output, plans, chosen, original_render)), \
            patch.object(uploader.sync, 'manifest_jobs', side_effect=selected_manifest):
        return uploader.main(argv)


if __name__ == '__main__':
    main()
