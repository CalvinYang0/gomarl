#!/usr/bin/env python3
"""Four-map Qtot-sum plan/model/mock-Slurm QA; no simulator/cloud/real jobs."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch

import torch as th
import yaml

import ozstar_submit_qtot_sum_four_maps_12jobs as suite
import ozstar_submit_5m6m_linear_baseline_recheck_3seeds as base
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from smoke_test_gate_score_supervision import equal_state
from learners import REGISTRY
from modules.mixers.qtot_composition import QTotCompositionMixer

ROOT = Path(__file__).resolve().parents[1]


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def plan_checks():
    plans = suite.build_plans(ROOT)
    references = {p['seed']: p for p in suite.reference_plans(ROOT) if p['label'] == suite.LABEL}
    assert len(plans) == len({p['job_name'] for p in plans}) == 12
    assert {p['label'] for p in plans} == {'qmix_qtot_sum_p10'}
    assert {p['map_name'] for p in plans} == {'8m_vs_9m', 'corridor', 'MMM2', '3s5z_vs_3s6z'}
    for scene, name, _, _ in suite.MAPS:
        assert {p['seed'] for p in plans if p['scene'] == scene and p['map_name'] == name} == {1, 2, 3}
    for plan in plans:
        reference = references[plan['seed']]
        assert extras(plan) == extras(reference), 'Map transfer changed training flags'
        exports = plan['exports']
        for key, value in reference['exports'].items():
            if key not in {'MAP_NAME', 'RUN_NAME', 'GROUP_NAME'}:
                assert exports[key] == value, key
        assert exports['MODEL_TYPE'] == 'qmix_minimal' and exports['GROUP_NAME'] == suite.GROUP
        assert exports['T_MAX'] == '10050000' and exports['TEST_INTERVAL'] == '10000'
        assert '--time=1-00:00:00' in plan['sbatch_args'] and '--mem=32G' in plan['sbatch_args']
        assert plan['memory'] == '32G' and plan['job_name'].endswith('_qtotsum20261026')
        expected = [a.replace(reference['job_name'], plan['job_name']) for a in reference['sbatch_args']]
        assert plan['sbatch_args'] == ['--mem=32G' if a.startswith('--mem=') else a for a in expected]
    validate_config_keys(ROOT, plans)
    print('PASS: four exact maps x seeds 1/2/3, same historical Qtot-sum recipe/head/TD/media; 10M/24h, no other models')
    return plans


def model_checks(plans):
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-qtot-transfer-model-') as temp:
        for _, name, agents, _ in suite.MAPS:
            plan = next(p for p in plans if p['map_name'] == name and p['seed'] == 1)
            overrides = {k: yaml.safe_load(v) for k, v in extras(plan).items() if '.' not in k}
            overrides['clean_model_type'] = 'qmix_minimal'
            th.manual_seed(101)
            mac, _, batch, logger = make_case('baseline', name, production_shapes=True, config_overrides=overrides)
            args = mac.args
            args.local_results_path = temp
            args.unique_token = 'SYNTHETIC_QTOT_TRANSFER_' + name
            args.wandb_run_name = 'SYNTHETIC_QA_NOT_EXPERIMENT'
            args.seed = 1
            actor = copy.deepcopy(mac.agent.state_dict())
            learner = REGISTRY['qtot_composition_learner'](mac, batch.scheme, logger, args)
            mixer = learner.mixer
            assert isinstance(mixer, QTotCompositionMixer)
            assert equal_state(actor, mac.agent.state_dict())
            assert args.n_agents == agents and len(mac.agent.fixed_head) == 3
            assert isinstance(mac.agent.fixed_head[1], th.nn.ELU)
            assert mac.agent.fixed_head[-1].out_features == args.n_actions
            assert mac.agent.hyper_out_w is None and mixer.alpha_logit is None
            assert mixer.output_combination == 'sum' and not mixer.secondary_ones
            th.testing.assert_close(mixer.gate_logits.sigmoid(), th.full_like(mixer.gate_logits, .1))
            q = th.randn(2, 4, agents)
            states = batch['state'][:, :-1]
            mask = mixer.sample_mask(states, 'train')
            assert set(mask.detach().unique().tolist()) <= {0., 1.}
            controlled = mixer.controlled_input(states, mask)
            assert th.equal(controlled[mask == 0], th.ones_like(controlled[mask == 0]))
            expected = mixer.base(q, th.ones_like(states)) + mixer.secondary(q, controlled)
            assert th.equal(mixer.conditioned_value(q, states, mask), expected)
            before_main = copy.deepcopy(mixer.base.state_dict())
            before_sub = copy.deepcopy(mixer.secondary.state_dict())
            before_gate = mixer.gate_logits.detach().clone()
            learner.train(batch, 10, 1)
            learner.train(batch, 11, 200)
            assert th.isfinite(learner.last_losses).all()
            assert not equal_state(actor, mac.agent.state_dict())
            assert not equal_state(before_main, mixer.base.state_dict())
            assert not equal_state(before_sub, mixer.secondary.state_dict())
            assert not th.equal(before_gate, mixer.gate_logits)
            assert equal_state(mixer.state_dict(), learner.target_mixer.state_dict())
            assert logger.stats['mixer_branches/qtot_main_mean']
            assert logger.stats['mixer_branches/qtot_secondary_mean']
            records = list((Path(temp)/'mixer_onefill_gate'/args.unique_token).glob('*_qtot.json'))
            assert records and json.loads(records[-1].read_text())['branches']['combined']['count'] > 0
            print('PASS:', name, 'real dimensions, unchanged actor head, Qtot algebra, joint actor/gate/two-mixer updates and diagnostics')


def asset_checks():
    import smac.env.starcraft2.maps.smac_maps as installed
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-qtot-transfer-assets-') as temp:
        root = Path(temp)
        binary = root/'Versions/Base000/SC2_x64'
        binary.parent.mkdir(parents=True)
        binary.write_text('SYNTHETIC placeholder')
        (root/'Maps').mkdir()
        registry = {}
        for _, name, agents, enemies in suite.MAPS:
            registry[name] = dict(n_agents=agents, n_enemies=enemies)
            (root/'Maps'/(name+'.SC2Map')).write_text('SYNTHETIC map')
        with patch.dict(os.environ, SC2PATH=temp), patch.object(installed, 'get_smac_map_registry', return_value=registry):
            suite.validate_maps()
            (root/'Maps/3s5z_vs_3s6z.SC2Map').unlink()
            try: suite.validate_maps()
            except RuntimeError: pass
            else: raise AssertionError('Exact map assets must be required')


def scheduler_checks():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-qtot-transfer-submit-') as temp:
        logs, calls = Path(temp), []
        plans = suite.build_plans(ROOT)
        # One active and one successful run are retained; only ten are missing.
        active = {plans[0]['job_name']: 'SYNTHETIC_ACTIVE'}
        completed = {plans[1]['job_name']: 'SYNTHETIC_COMPLETED'}
        def run(cmd, **kwargs):
            if cmd == ['id', '-un']: return 'SYNTHETIC_USER'
            if cmd == ['git', 'rev-parse', 'HEAD']: return 'SYNTHETIC_COMMIT'
            assert cmd[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 10
            assert kwargs['env']['GROUP_NAME'] == suite.GROUP
            calls.append('submit')
            return str(98000 + calls.count('submit'))
        def subprocess_run(cmd, **kwargs):
            if cmd == ['quota', '-v']:
                return subprocess.CompletedProcess(cmd, 0, '/home 8388608 0 20969520 - 100000 0 200000 -', '')
            if cmd[0] == 'sbatch':
                assert cmd[1] == '--test-only' and 'submit' not in calls
                calls.append('preflight')
            else:
                assert cmd[1] in suite.SMOKE_SCRIPTS
            return subprocess.CompletedProcess(cmd, 0)
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='NO', REPO_DIR=str(ROOT)), \
                    patch.object(base, 'run', side_effect=AssertionError('Preview cannot call scheduler')):
                suite.main()
            for row in ('/home 18000000 0 20969520 - 100000 0 200000 -',
                        '/home 8388608 0 20969520 - 150001 0 200000 -'):
                with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES'), \
                        patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, row, '')), \
                        patch.object(suite, 'submit_suite') as submit:
                    try: suite.main()
                    except RuntimeError: pass
                    else: raise AssertionError('Insufficient bytes/inodes must block')
                    submit.assert_not_called()
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(ROOT), RUNTIME_ROOT=temp), \
                    patch.object(suite, 'validate_maps') as validate, \
                    patch.object(base, 'home_quota_free_gib', return_value=6), \
                    patch.object(base, 'guard_and_route', return_value={'logs': logs}), \
                    patch.object(base, 'active_jobs', return_value=active), \
                    patch.object(base, 'completed_jobs', return_value=completed), \
                    patch.object(base, 'run', side_effect=run), \
                    patch.object(subprocess, 'run', side_effect=subprocess_run):
                suite.main()
                validate.assert_called_once()
                assert calls == ['preflight']*10 + ['submit']*10
                record = json.loads(next(logs.glob(suite.GROUP+'_*.json')).read_text())
                assert len(record['submitted']) == 10 and len(record['retained']) == 2
                active.update(record['submitted'])
                calls.clear()
                suite.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: preview, byte/inode guards, all Slurm preflights first, manifest, active/completed retention and repeat idempotence (mocked)')


def main():
    plans = plan_checks()
    model_checks(plans)
    asset_checks()
    scheduler_checks()


if __name__ == '__main__':
    main()
