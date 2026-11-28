#!/usr/bin/env python3
"""Noise1x: init1/init0.1 + 5M curriculum/binary freeze; 3 maps x 3 seeds.

Plan-only by default. Real learner preflight runs in a compute allocation;
SUBMIT=YES on the login node requires that exact source/plan receipt.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import ozstar_submit_fixed_mask_noise_27jobs as base
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys

ROOT = Path(__file__).resolve().parents[1]
GROUP = 'noise1x_init1_init01_fast5m_freeze5m90_three_maps_18jobs_20261129'
VARIANTS = {'init1': ('qmix_projected_init1_faststages5m_freeze90_noise1x', '1.0'),
            'init01': ('qmix_projected_init01_faststages5m_freeze90_noise1x', '0.1')}
SMOKE = 'scripts/smoke_test_noise_freeze5m.py'


def build_plans(repo, scene=None, variant=None):
    if variant is not None and variant not in VARIANTS:
        raise ValueError('Unknown initialization variant')
    templates = [p for p in base.build_plans(repo, scene, 'noise') if p['label'] == base.MODELS[2]]
    result = []
    for key in ((variant,) if variant else VARIANTS):
        label, initial = VARIANTS[key]
        for template in templates:
            p = copy.deepcopy(template)
            x = dict(v.split('=',1) for v in shlex.split(p['exports']['EXTRA_ARGS']))
            x.update(learner='noise_projected_freeze_refresh_learner',
                mixer_input_initial_probability=initial, mixer_state_refresh_full_steps='5000000',
                mixer_soft_gate_freeze_steps='5000000', mixer_soft_gate_freeze_threshold='0.9')
            p['label'] = label
            p['exports'].update(GROUP_NAME=GROUP, EXTRA_ARGS=shlex.join(k+'='+v for k,v in x.items()))
            rename(p, '{}_{}_10m_s{}_noisefreeze20261129'.format(p['scene'], label, p['seed']))
            p['inventory_note'] = (
                'Fresh noise1x; p0='+initial+'; faster_stages full at5M; first update >=5M freezes '
                'ONE online-derived mask into online+target gates, p>0.9 ->1 else0; closed fills1; '
                'base mixer/actor keep training. Same collected U[0,1) noise, replay, TD, actor, '
                'live evaluation, 10M, 3 seeds, 48h, 28CPU/32G, no video/checkpoint. '
                'Noise is not identified by the model; either real or noise fields may be closed.')
            result.append(p)
    expected = (1 if scene else 3)*(1 if variant else 2)*3
    if len(result) != len({p['job_name'] for p in result}) or len(result) != expected:
        raise RuntimeError('Expected exact noise freeze map/model/seed coverage')
    return result


def fingerprint(repo, plans):
    files = subprocess.check_output(['git','ls-files','--','src','scripts'], cwd=repo, text=True).splitlines()
    files = sorted(set(files) | {Path(__file__).resolve().relative_to(repo).as_posix(), SMOKE})
    digest = hashlib.sha256(json.dumps(plans, sort_keys=True).encode())
    for name in files:
        if Path(name).suffix not in {'.py','.yaml','.yml','.sh','.sbatch'}:
            continue
        path = repo/name
        if path.is_symlink():
            raise RuntimeError('Linked preflight source: '+name)
        digest.update(name.encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def receipt_path(runtime, plans):
    scope = hashlib.sha256(json.dumps(sorted(p['job_name'] for p in plans)).encode()).hexdigest()[:12]
    return runtime/'ozstar_logs'/('.'+GROUP+'_preflight_'+scope+'.json')


def require_receipt(repo, runtime, plans):
    path = receipt_path(runtime, plans)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('Run --preflight in a compute allocation before submission')
    record = json.loads(path.read_text())
    if (record.get('fingerprint') != fingerprint(repo, plans) or
            record.get('python') != str(Path(sys.executable).resolve()) or
            not str(record.get('slurm_job_id','')).isdigit() or
            not 0 <= time.time()-record.get('time',0) < 86400):
        raise RuntimeError('Preflight receipt is stale or source/plan/Python changed; rerun --preflight')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=tuple(row[0] for row in base.MAPS))
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    repo = ROOT
    runtime = Path(os.environ.get('RUNTIME_ROOT','/home/kyang/gomarl-runtime/gomarl-dual-branch'))
    plans = build_plans(repo, args.scene, args.variant)
    validate_config_keys(repo, plans)
    if args.preflight:
        if not os.environ.get('SLURM_JOB_ID','').isdigit():
            raise SystemExit('Learner preflight must run in an allocated compute job, not on a login node')
        if runtime.is_symlink() or (runtime/'ozstar_logs').is_symlink():
            raise SystemExit('Unsafe preflight output path')
        before = fingerprint(repo, plans)
        base.validate_maps(args.scene)
        subprocess.run([sys.executable, str(repo/SMOKE)], cwd=repo, check=True)
        if fingerprint(repo, plans) != before:
            raise RuntimeError('Source changed during preflight')
        path = receipt_path(runtime, plans)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise RuntimeError('Linked preflight receipt')
        with path.open('w') as handle:
            json.dump(dict(fingerprint=before, python=str(Path(sys.executable).resolve()),
                slurm_job_id=os.environ['SLURM_JOB_ID'], time=time.time()), handle, indent=2)
        print('PREFLIGHT VERIFIED:', path, flush=True)
        return
    if os.environ.get('SUBMIT') == 'YES':
        require_receipt(repo, runtime, plans)
        quota = subprocess.check_output(['quota','-v'], text=True, timeout=30)
        base.parent.require_capacity(base.parent.parse_home_quota(quota))
    os.environ['REPO_DIR'] = str(repo)
    # No learner/test-suite compute on the login node. The receipt above pins
    # the exact tested source and plan; existing Slurm dry-run/dedup guards stay.
    base.parent.submit_suite(GROUP, __file__, lambda _: plans, (),
                             map_validator=lambda: base.validate_maps(args.scene))


if __name__ == '__main__':
    main()
