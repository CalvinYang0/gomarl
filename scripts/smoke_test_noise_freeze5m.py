#!/usr/bin/env python3
"""Synthetic real-learner QA for the 18 noise1x freeze jobs; no game/cloud/jobs."""
import copy
import json
from pathlib import Path
import shlex
import sys
import tempfile
import time
from unittest.mock import patch

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_noise_freeze5m_18jobs as suite
from smoke_test_fixed_mask_noise import make, extras, noise_checks, runner_checks
from learners import REGISTRY
from utils.mixer_state_refresh import scheduled_segments


def equal(a, b):
    return set(a) == set(b) and all(th.equal(a[k], b[k]) for k in a)


def plan_checks():
    plans = suite.build_plans(ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 18
    assert len(suite.build_plans(ROOT, variant='init01')) == 9
    assert len(suite.build_plans(ROOT, scene='smac_corridor')) == 6
    suite.validate_config_keys(ROOT, plans)
    references = {(p['scene'],p['seed']):p for p in suite.base.build_plans(ROOT, experiment='noise')
                  if p['label'] == suite.base.MODELS[2]}
    for p in plans:
        x, old = extras(p), extras(references[p['scene'],p['seed']])
        changed = {'learner','mixer_input_initial_probability','mixer_state_refresh_full_steps',
                   'mixer_soft_gate_freeze_steps','mixer_soft_gate_freeze_threshold'}
        assert {k:v for k,v in x.items() if k not in changed} == {k:v for k,v in old.items() if k not in changed}
        assert x['mixer_state_noise'] and x['mixer_state_refresh_enabled']
        assert x['mixer_state_refresh_full_steps'] == x['mixer_soft_gate_freeze_steps'] == 5000000
        assert x['mixer_soft_gate_freeze_threshold'] == .9
        assert '--time=2-00:00:00' in p['sbatch_args'] and p['exports']['T_MAX'] == '10050000'
        assert p['mixer_state_dim'] == 2*p['raw_state_dim']
    print('PASS exact18 plans; only initialization/freeze/5M curriculum change; same noise, TD, actor, 48h')
    return plans


def train(learner, batch, step, episode):
    # Exercise freeze logging without rendering large synthetic heatmaps.
    with patch.object(learner, '_snapshot'):
        learner.train(batch, step, episode)


def check_model(plan, temp):
    learner, batch = make(plan, temp)
    m = learner.mixer
    initial = extras(plan)['mixer_input_initial_probability']
    assert th.all(m.coefficients() == initial)
    assert learner.freeze_steps == 5000000
    assert scheduled_segments(learner.args, 5000000, 170) == 170
    assert scheduled_segments(learner.args, 0, 170, test_mode=True) == 170
    # Same initialized actor/mixer and same collected batch: the freeze class
    # must not change a single pre-freeze TD update.
    ordinary_args = copy.deepcopy(learner.args)
    ordinary_args.learner = 'noise_projected_refresh_learner'
    ordinary = REGISTRY[ordinary_args.learner](copy.deepcopy(learner.mac), batch.scheme,
                                             learner.logger, ordinary_args)
    ordinary.mixer.base.load_state_dict(m.base.state_dict())
    ordinary.target_mixer.base.load_state_dict(learner.target_mixer.base.state_dict())
    train(ordinary, batch, 4999999, 1)
    train(learner, batch, 4999999, 1)
    assert equal(ordinary.mac.agent.state_dict(), learner.mac.agent.state_dict())
    assert equal(ordinary.mixer.base.state_dict(), m.base.state_dict())
    assert th.equal(ordinary.mixer.coefficients(), m.coefficients())
    assert m.gate_parameter.grad is not None and m.gate_parameter in learner.optimiser.state
    dim = m.state_dim//2
    with th.no_grad():
        m.gate_parameter.fill_(.2)
        m.gate_parameter[:3] = th.tensor([.89,.90,.91])
        m.gate_parameter[dim:dim+3] = th.tensor([.89,.90,.91])
        learner.target_mixer.gate_parameter.fill_(.99)
    target_before = copy.deepcopy(learner.target_mixer.base.state_dict())
    actor_before = copy.deepcopy(learner.mac.agent.state_dict())
    base_before = copy.deepcopy(m.base.state_dict())
    train(learner, batch, 5000003, 2)
    mask = m.coefficients().detach().clone()
    assert mask[:3].tolist() == mask[dim:dim+3].tolist() == [0,0,1]
    assert mask.sum() == 2 and th.equal(mask, learner.target_mixer.coefficients())
    assert m.freeze_t_env.item() == learner.target_mixer.freeze_t_env.item() == 5000003
    assert equal(target_before, learner.target_mixer.base.state_dict()), 'Freeze copied unrelated target weights'
    assert not equal(actor_before, learner.mac.agent.state_dict())
    assert not equal(base_before, m.base.state_dict())
    assert not m.gate_parameter.requires_grad and m.gate_parameter.grad is None
    assert m.gate_parameter not in learner.optimiser.state
    values = m.controlled_input(batch['state'], m.gate_values(batch['state']))
    assert th.all(values[...,mask==0] == 1)
    assert th.equal(values[...,mask==1], batch['state'][...,mask==1])
    metrics = learner._gate_metrics(mask)
    assert metrics['mixer_input/noise_exact_closed_slots'] == dim-1
    assert metrics['mixer_input/real_exact_closed_slots'] == dim-1
    train(learner, batch, 9000000, 3)
    assert th.equal(mask, m.coefficients()) and m.freeze_t_env.item() == 5000003
    path = Path(temp)/plan['job_name']
    path.mkdir()
    learner.save_models(path)
    restored, replay = make(plan, temp)
    restored.load_models(path)
    assert not restored.mixer.gate_parameter.requires_grad
    assert not restored.target_mixer.gate_parameter.requires_grad
    train(restored, replay, 9500000, 4)
    assert th.equal(mask, restored.mixer.coefficients())
    # A fully closed mask is a permitted outcome, not silently repaired.
    closed, replay = make(plan, temp)
    with th.no_grad(): closed.mixer.gate_parameter.fill_(.1)
    train(closed, replay, 5000000, 1)
    assert not closed.mixer.coefficients().any()
    assert th.all(closed.mixer.controlled_input(replay['state'], closed.mixer.gate_values(replay['state'])) == 1)
    print('PASS real TD/freeze/targets/resume/all-closed:', plan['map_name'], initial, flush=True)


def receipt_checks(plans, temp):
    runtime = Path(temp)
    path = suite.receipt_path(runtime, plans)
    path.parent.mkdir(exist_ok=True)
    try: suite.require_receipt(ROOT, runtime, plans)
    except RuntimeError: pass
    else: raise AssertionError('Accepted missing compute preflight')
    record = dict(fingerprint=suite.fingerprint(ROOT, plans),
                  python=str(Path(sys.executable).resolve()), slurm_job_id='123', time=time.time())
    path.write_text(json.dumps(record))
    suite.require_receipt(ROOT, runtime, plans)
    record['fingerprint'] = 'changed-source'
    path.write_text(json.dumps(record))
    try: suite.require_receipt(ROOT, runtime, plans)
    except RuntimeError: pass
    else: raise AssertionError('Accepted changed source')
    print('PASS compute-preflight receipt required and source/plan fingerprint checked')


def startup_checks(plans):
    import logging
    from types import SimpleNamespace
    from smoke_test_no_video_startup import config_for
    from run.run import run_sequential, args_sanity_check, r_REGISTRY
    from utils.logging import Logger
    class Reached(Exception): pass
    for plan in plans:
        if plan['seed'] != 1:
            continue
        cfg = config_for(plan)
        cfg['clean_model_type'] = plan['exports']['MODEL_TYPE']
        log = logging.getLogger('SYNTHETIC-noisefreeze-startup')
        args = SimpleNamespace(**args_sanity_check(cfg, log))
        def stop(args, logger):
            assert args.env_args['mixer_state_noise']
            assert args.learner == 'noise_projected_freeze_refresh_learner'
            raise Reached()
        with patch.dict(r_REGISTRY, {args.runner: stop}):
            try: run_sequential(args, Logger(log))
            except Reached: pass
            else: raise AssertionError('Startup did not reach runner construction')
    print('PASS six production startup preflights and noise learner routing')


def main():
    th.set_num_threads(1)
    plans = plan_checks()
    startup_checks(plans)
    noise_checks()
    runner_checks()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-noisefreeze-') as temp:
        for plan in plans:
            if plan['seed'] == 1:
                check_model(plan, temp)
        receipt_checks(plans, temp)
    print('PASS no simulator, no scheduler submission, no W&B upload, no real data deleted')


if __name__ == '__main__': main()
