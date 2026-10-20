#!/usr/bin/env python3
"""Synthetic QA, no simulator/cloud/Slurm writes: five weight-composition models."""
import copy
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from smoke_test_onefill_gate import case, extras, small_mixer
from smoke_test_gate_score_supervision import equal_state
from modules.mixers.onefill_gate import OneFillGateMixer
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
import ozstar_submit_weight_composition_15jobs as study
import ozstar_submit_8m9m_ones_gate_regularization_9jobs as common


def mixer_for(combination='centered', fill=1, both=False, layers=2):
    return OneFillGateMixer(SimpleNamespace(state_shape=4, n_agents=2, mixing_embed_dim=8,
        hypernet_layers=layers, hypernet_embed=8, mixer_input_initial_probability=.1,
        mixer_input_gate_temperature=.5, mixer_onefill_residual=True, mixer_onefill_residual_scale=.1,
        mixer_residual_fill_value=fill, mixer_residual_combination=combination,
        mixer_residual_secondary_ones=both, mixer_residual_alpha_initial=.1))


def algebra():
    th.manual_seed(43)
    q = th.randn(2, 3, 2, requires_grad=True)
    states = th.randn(2, 3, 4)
    for layers in (1, 2):
        for mode, fill, both in [('centered', 1, False), ('centered', 0, False),
                ('bounded', 1, False), ('sum', 1, False), ('alpha_sum', 1, False), ('sum', 1, True)]:
            mixer = mixer_for(mode, fill, both, layers)
            initial = mixer.weight_components(mixer.controlled_input(states, th.ones_like(states)))
            # Zero heads do NOT deadlock: abs is applied after raw addition.
            for part in initial.values():
                if mode == 'alpha_sum':
                    assert th.allclose(part['effective'], .1 * part['raw_main'].abs())
                else:
                    assert th.equal(part['effective'], part['raw_main'].abs())
            value = mixer(q, states, mode='train')
            value.square().sum().backward()
            for network in (mixer.residual_w1, mixer.residual_w2):
                head = network if isinstance(network, th.nn.Linear) else network[-1]
                assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
            if mode == 'alpha_sum':
                assert math.isclose(mixer.alpha().item(), .1, abs_tol=1e-7)
                assert mixer.alpha_logit.grad.abs() > 0
                for logit in (-100., 100.):
                    with th.no_grad():
                        mixer.alpha_logit.fill_(logit)
                    assert 0 < mixer.alpha().item() < 1
                with th.no_grad():
                    mixer.alpha_logit.fill_(math.log(.1/.9))
            with th.no_grad():
                for network in (mixer.residual_w1, mixer.residual_w2):
                    head = network if isinstance(network, th.nn.Linear) else network[-1]
                    head.weight.normal_(std=10)
                    head.bias.normal_(std=10)
            controlled = mixer.controlled_input(states, th.ones_like(states))
            for layer, part in mixer.weight_components(controlled).items():
                secondary_head = mixer.residual_w1 if layer == 'w1' else mixer.residual_w2
                closed = th.full_like(controlled, fill).reshape(-1, 4)
                if mode == 'centered':
                    expected = (part['raw_main'] + .1 * (part['raw_secondary'] - secondary_head(closed))).abs()
                elif mode == 'bounded':
                    expected = part['raw_main'].abs() * (1 + .1 * (part['raw_secondary'] - secondary_head(closed)).tanh())
                    assert th.all(part['effective'] <= 1.100001 * part['raw_main'].abs())
                    assert th.all(part['effective'] >= .899999 * part['raw_main'].abs())
                elif mode == 'sum':
                    expected = (part['raw_main'] + part['raw_secondary']).abs()
                else:
                    expected = (.1 * part['raw_main'] + .9 * part['raw_secondary']).abs()
                assert th.allclose(part['effective'], expected, atol=1e-5)
            result = mixer.conditioned_value(q, states, th.ones_like(states))
            assert (th.autograd.grad(result.sum(), q)[0] >= 0).all()
            if mode in {'centered', 'bounded'}:
                assert th.equal(mixer.conditioned_value(q, states, th.zeros_like(states)), mixer.base(q, th.ones_like(states)))
            if both:
                rng = th.get_rng_state()
                assert th.equal(mixer.sample_mask(states, 'train'), th.ones_like(states))
                assert th.equal(th.get_rng_state(), rng)
                assert not mixer.gate_logits.requires_grad
                assert th.equal(mixer(q, states), mixer(q, states + 1000))
            print('PASS algebra:', mode, 'fill=', fill, 'both_ones=', both, 'layers=', layers)
    # Preserve the old default initializer/state_dict and output exactly.
    th.manual_seed(123)
    old = small_mixer(True, 2)
    th.manual_seed(123)
    explicit = mixer_for(layers=2)
    assert equal_state(old.state_dict(), explicit.state_dict())
    assert th.equal(old(q, states), explicit(q, states))
    zero = mixer_for(fill=0)
    x = states.detach().clone().requires_grad_(True)
    mask = th.tensor([1., 0., 1., 0.])
    zero.controlled_input(x, mask).sum().backward()
    assert th.equal(x.grad, mask.expand_as(x))
    # Explicit negative cancellation distinguishes abs(A+B) from abs(A)+abs(B).
    add = mixer_for('sum', layers=1)
    with th.no_grad():
        for network in (add.base.hyper_w1, add.base.hyper_w_final):
            network.weight.zero_(); network.bias.fill_(2)
        for network in (add.residual_w1, add.residual_w2):
            network.weight.zero_(); network.bias.fill_(-3)
    for part in add.weight_components(states).values():
        assert th.equal(part['effective'], th.ones_like(part['effective']))
    print('PASS: exact old defaults, zero-field derivatives, abs-after-sum cancellation')


def production(root):
    plans = study.build_plans(Path(__file__).resolve().parents[1])
    for plan in [p for p in plans if p['seed'] == 1]:
        extra = extras(plan)
        kwargs = dict(mixer_residual_combination=extra.get('mixer_residual_combination', 'centered'),
            mixer_residual_fill_value=int(extra.get('mixer_residual_fill_value', 1)),
            mixer_residual_secondary_ones=extra.get('mixer_residual_secondary_ones') == 'True',
            mixer_residual_alpha_initial=.1)
        for optimizer in ('adam', 'rmsprop'):
            learner, batch, logger = case(plan['map_name'], 'td', True, optimizer, root, **kwargs)
            learner.args.unique_token = 'SYNTHETIC_' + plan['label'] + '_' + optimizer
            before = copy.deepcopy(learner.mixer.state_dict())
            learner.train(batch, 10, 1)
            learner.train(batch, 11, 2)  # gate gets a path after zero heads have moved
            assert learner.train_steps == 2 and th.isfinite(learner.last_losses).all()
            assert not equal_state(before, learner.mixer.state_dict())
            if learner.mixer.secondary_ones:
                assert th.equal(before['gate_logits'], learner.mixer.gate_logits)
                assert 'mixer_input/gate_probability_mean' not in logger.stats
            else:
                assert not th.equal(before['gate_logits'], learner.mixer.gate_logits)
            assert learner.target_mixer.gate_logits.grad is None
            summary = ValueDiagnosticSummary()
            hidden_before = learner.mac.hidden_states
            rng = th.get_rng_state()
            collect_value_diagnostics(learner.mac, learner.mixer, batch, learner.args.gamma, summary)
            assert th.equal(th.get_rng_state(), rng) and learner.mac.hidden_states is hidden_before
            summary.log(logger, 11)
            for layer in ('w1', 'w_final'):
                for part in ('raw_main', 'raw_secondary', 'main_contribution', 'secondary_contribution', 'effective'):
                    assert 'test_value/mixer_{}_{}_norm_mean'.format(layer, part) in logger.stats
            if learner.mixer.alpha_logit is not None:
                assert 'test_value/mixer_branch_alpha_mean' in logger.stats
                assert 'mixer_branches/alpha_main' in logger.stats
            path = root / ('checkpoint_' + learner.args.unique_token)
            path.mkdir()
            learner.save_models(path)
            fresh, _, _ = case(plan['map_name'], 'td', True, optimizer, root, **kwargs)
            fresh.load_models(path)
            assert equal_state(learner.mixer.state_dict(), fresh.mixer.state_dict())
            assert equal_state(learner.target_mixer.state_dict(), fresh.target_mixer.state_dict())
            assert equal_state(learner.optimiser.state_dict(), fresh.optimiser.state_dict())
            learner._update_targets()
            fresh._update_targets()
            assert equal_state(learner.mixer.state_dict(), learner.target_mixer.state_dict())
            th.manual_seed(551)
            learner.train(batch, 12, 3)
            th.manual_seed(551)
            fresh.train(batch, 12, 3)
            assert all(th.equal(a, b) for a, b in zip(learner.params, fresh.params))
            # Mismatch rejected before any weights/optimizer are changed.
            fresh.contract['combination'] = 'SYNTHETIC_MISMATCH'
            before = copy.deepcopy(fresh.mixer.state_dict())
            try:
                fresh.load_models(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched composition checkpoint accepted')
            assert equal_state(before, fresh.mixer.state_dict())
            learner.logger.use_wandb = True
            learner.logger.wandb_current_t, learner.logger.wandb_current_data = 12, {}
            learner.logger.wandb_module = SimpleNamespace(Image=lambda p: ('SYNTHETIC_IMAGE', p), save=lambda *a, **k: None)
            valid = batch['filled'][:, :-1].float() * batch['filled'][:, 1:].float()
            valid[:, 1:] *= (1 - batch['terminated'][:, :-2].float()).cumprod(1)
            learner._snapshot(12, batch['state'][:, :-1], valid)
            assert 'mixer_branches/weight_components_heatmap' in learner.logger.wandb_current_data
            assert learner.logger._wandb_metric_allowed('mixer_branches/alpha_main')
            if learner.mixer.alpha_logit is not None:
                learner.logger.wandb_minimal_logging = True
                learner.logger.log_stat('mixer_branches/alpha_main', learner.mixer.alpha().item(), 12)
                assert 'mixer_branches/alpha_main' in learner.logger.wandb_current_data
            directory = root / 'mixer_onefill_gate' / learner.args.unique_token
            record = json.loads(next(directory.glob('*.json')).read_text())
            assert record['weights']['sample_count'] == int(valid.sum())
            assert set(record['weights']['layers']) == {'w1', 'w_final'}
            if learner.mixer.secondary_ones:
                assert record['keep_probability'] is None
            if os.environ.get('WEIGHT_QA_OUTPUT') and optimizer == 'adam':
                import shutil
                shutil.copytree(directory, Path(os.environ['WEIGHT_QA_OUTPUT']) / plan['label'], dirs_exist_ok=True)
            print('PASS production:', plan['label'], optimizer, 'training/target/resume/eval/heatmap')


def submission(root):
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    validate_config_keys(repo, plans)
    assert len(plans) == len({p['job_name'] for p in plans}) == 15
    assert sum(p['scene'] == 'smac_5m6m' for p in plans) == 12
    assert sum(p['scene'] == 'smac_8m9m' for p in plans) == 3
    for p in plans:
        flags = extras(p)
        assert flags['mixer_onefill_objective'] == 'td'
        assert flags['mixer_input_initial_probability'] == '0.1'
        assert flags['mixer_onefill_residual'] == 'True'
        assert flags['mixer_input_detach_unimportant'] == 'False'
        assert flags['mixer_gate_aux_score'] == 'none'
        assert not any('threshold' in key for key in flags)
        assert p['exports']['T_MAX'] == '10050000'
    import ozstar_submit_5m6m_onefill_gate_15jobs as old
    from ozstar_submit_5m6m_residual_zerofill_3jobs import build_plans as zero
    for p in zero(repo):
        reference = next(o for o in old.build_plans(repo) if o['label'] == 'qmix_onefill_p10_residual' and o['seed'] == p['seed'])
        a, b = extras(p), extras(reference)
        assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {'mixer_residual_fill_value'}
    selected = {p['label']: extras(p) for p in plans}
    a, b = selected['qmix_weight_sum_p10'], selected['qmix_weight_sum_bothones']
    assert {k for k in a if a[k] != b[k]} == {'mixer_residual_secondary_ones'}
    a, b = selected['qmix_weight_sum_p10'], selected['qmix_weight_alpha10_p10']
    assert {k for k in a if a[k] != b[k]} == {'mixer_residual_combination'}
    calls = []
    retained = {p['job_name']: str(100+i) for i, p in enumerate(plans[:3])}
    def checked_run(command, **kwargs):
        if command[0] == 'sbatch':
            assert command[1] == '--test-only' and 'submit' not in calls
            calls.append('preflight')
    def run(command, **kwargs):
        if command == ['id', '-un']:
            return 'SYNTHETIC_USER'
        if command == ['git', 'rev-parse', 'HEAD']:
            return 'SYNTHETIC_COMMIT'
        assert command[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 12
        calls.append('submit')
        return str(500+calls.count('submit'))
    cwd = Path.cwd()
    try:
        with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=str(root)), \
             patch.object(common, 'validate_maps'), \
             patch.object(common.submitter, 'home_quota_free_gib', return_value=2.1), \
             patch.object(common.submitter, 'guard_and_route', return_value={'logs': root}), \
             patch.object(common.submitter, 'active_jobs', return_value=retained), \
             patch.object(common.submitter, 'completed_jobs', return_value={}), \
             patch.object(common.submitter.subprocess, 'run', side_effect=checked_run), \
             patch.object(common.submitter, 'run', side_effect=run):
            study.main()
            manifest = json.loads(next(root.glob(study.GROUP + '_*.json')).read_text())
            assert len(manifest['submitted']) == 12
            retained.update(manifest['submitted']); calls.clear()
            study.main()
            assert not calls
    finally:
        os.chdir(cwd)
    import plot_5m6m_8m9m_qmix_variants_3seeds as charts
    import ozstar_calibration_suite_sync as sync
    import ozstar_clean_except_qmix_variants as cleanup
    assert study.GROUP in sync.GROUPS and study.GROUP in cleanup.KEEP_GROUPS
    inventory = charts.build_plans(repo)
    assert len(inventory) == len({p['job_name'] for p in inventory}) == 108
    assert {p['job_name'] for p in plans} <= {p['job_name'] for p in inventory}
    assert 'qmix_weight_bounded_p10' not in charts.SCENE_MODELS['smac_5m6m']
    assert 'qmix_weight_sum_p10' not in charts.SCENE_MODELS['smac_8m9m']
    print('PASS: exact maps/seeds, clean differences, mocked preflight/idempotence, plot/sync/protection inventory')


def main():
    th.set_num_threads(1)
    algebra()
    with tempfile.TemporaryDirectory(prefix='gomarl-weights-SYNTHETIC-QA-') as temp:
        root = Path(temp)
        production(root)
        submission(root)
    print('PASS: all weight composition checks; NO real training, cloud upload or Slurm submission')


if __name__ == '__main__':
    main()
