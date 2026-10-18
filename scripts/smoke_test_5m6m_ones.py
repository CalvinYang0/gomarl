#!/usr/bin/env python3
"""5m6m ones config, real learner, Slurm idempotence; no simulator/cloud/jobs."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
from unittest.mock import patch

import torch as th
import yaml

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
import ozstar_submit_5m6m_ones_3seeds as study
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import build_plans as previous_ones
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def main():
    th.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    validate_config_keys(repo, plans)
    original = {p['seed']: p for p in previous_ones(repo) if p['label'] == 'qmix_state_ones'}
    assert len(plans) == len({p['job_name'] for p in plans}) == 3
    for p in plans:
        assert p['scene'] == 'smac_5m6m' and p['map_name'] == '5m_vs_6m'
        assert p['label'] == 'qmix_state_ones' and extras(p) == extras(original[p['seed']])
        assert p['exports']['MODEL_TYPE'] == 'qmix_minimal'
        assert int(p['exports']['T_MAX']) == 10050000 and p['target_steps'] == 10000000
        assert p['exports']['RUN_NAME'] == p['job_name']
        assert p['job_name'] == 'smac_5m6m_qmix_state_ones_10m_s{}_ablate20261018'.format(p['seed'])
    with tempfile.TemporaryDirectory(prefix='gomarl-5m6m-ones-QA-') as temp:
        th.manual_seed(41)
        overrides = {key: yaml.safe_load(value) for key, value in extras(plans[0]).items()}
        overrides['clean_model_type'] = 'qmix_minimal'
        mac, _, batch, logger = make_case('baseline', '5m_vs_6m', production_shapes=True,
                                         config_overrides=overrides)
        args = mac.args
        args.local_results_path, args.unique_token = temp, 'SYNTHETIC_5M6M_ONES_QA'
        args.wandb_run_name, args.seed = 'SYNTHETIC_5M6M_ONES_QA_NOT_EXPERIMENT', 1
        learner = MixerInputLearner(mac, batch.scheme, logger, args)
        mixer = learner.mixer
        assert (args.n_agents, args.n_actions, args.state_shape) == (5, 12, 98)
        assert not mixer.grouped and not mixer.append_mask and mixer.gate_logits is None
        assert len(mixer.mixers) == 1 and mixer.mixers[0].state_dim == 98
        assert mac.agent.fixed_head is not None and mac.agent.hyper_out_w is None
        assert learner.gate_l1_coef == learner.gate_variance_coef == 0
        states = batch['state'][:, :1].clone().requires_grad_(True)
        q = th.randn(batch.batch_size, 1, 5, requires_grad=True)
        for mode in ('train', 'target', 'eval'):
            value = mixer.values(q, states, mode)
            assert th.equal(value, mixer.values(q, states + 999, mode))
            assert th.equal(mixer.controlled_input(states, mixer.masks(states, mode)[0]), th.ones_like(states))
            state_grad, q_grad = th.autograd.grad(value.sum(), (states, q), allow_unused=True)
            assert state_grad is None and q_grad.abs().sum() > 0
        ordinary = learner._rollout(mac, batch).detach()
        altered = copy.deepcopy(batch)
        altered['state'].add_(100)
        assert th.equal(ordinary, learner._rollout(mac, altered).detach())
        altered['obs'].add_(1)
        assert not th.equal(ordinary, learner._rollout(mac, altered).detach())
        actor_before = [p.detach().clone() for p in mac.parameters()]
        mixer_before = [p.detach().clone() for p in mixer.parameters()]
        learner.train(batch, 10, 1)
        assert learner.last_losses.numel() == 1 and th.isfinite(learner.last_losses).all()
        assert any(not th.equal(a, b) for a, b in zip(actor_before, mac.parameters()))
        assert any(not th.equal(a, b) for a, b in zip(mixer_before, mixer.parameters()))
        snapshot = json.loads(next((Path(temp) / 'mixer_input_control').rglob('step_*.json')).read_text())
        assert snapshot['variant'] == 'ones' and snapshot['mixer_condition_dim'] == 98
        assert all(value == 1 for row in snapshot['masks_or_probability'] for value in row)
        checkpoint = Path(temp) / 'checkpoint'
        checkpoint.mkdir()
        learner.save_models(checkpoint)
        learner.load_models(checkpoint)
        # Register original runs in one-shot ended sync and in both map/mean3
        # inventories. Running-only tmux discovers repository jobs unchanged.
        import ozstar_calibration_suite_sync as sync
        import plot_5m6m_8m9m_qmix_variants_3seeds as plots
        assert study.GROUP in sync.GROUPS
        assert 'qmix_state_ones' in plots.SCENE_MODELS['smac_5m6m']
        registered = {p['job_name'] for p in plots.build_plans(repo)}
        assert all(p['job_name'] in registered for p in plans)
        calls, retained = [], {}
        def preflight(command, **kwargs):
            if command[0] == 'sbatch':
                assert command[1] == '--test-only' and 'submit' not in calls
                calls.append('preflight')
        def command(command, **kwargs):
            if command == ['id', '-un']:
                return 'SYNTHETIC_USER'
            if command == ['git', 'rev-parse', 'HEAD']:
                return 'SYNTHETIC_COMMIT'
            assert command[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 3
            calls.append('submit')
            return str(91000 + calls.count('submit'))
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ,
                    SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch('ozstar_submit_5m6m_id_kl80_3seeds.validate_installed_map'), \
                 patch.object(study.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(study.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(study.submitter, 'active_jobs', return_value=retained), \
                 patch.object(study.submitter, 'completed_jobs', return_value={}), \
                 patch.object(study.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(study.submitter, 'run', side_effect=command):
                study.main()
                assert calls.count('preflight') == calls.count('submit') == 3
                manifest = json.loads(next(Path(temp).glob(study.GROUP + '_*.json')).read_text())
                retained.update(manifest['submitted'])
                calls.clear()
                study.main()
                assert not calls
                assert set(sync.manifest_jobs(Path(temp)).values()) == set(retained)
        finally:
            os.chdir(cwd)
    print('PASS: exactly three seeds/10M; same factors as 8m9m ones; real 5m6m learner; '
          'constant mixer condition, ordinary actor; media/checkpoint; sync/mean3 registry; '
          'Slurm preflight and idempotence (mocked, no jobs submitted)')


if __name__ == '__main__':
    main()
