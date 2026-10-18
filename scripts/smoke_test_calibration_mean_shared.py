#!/usr/bin/env python3
"""MC-error score ablation: exact perturbations/gradients, real learner, no jobs."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
from unittest.mock import patch

import torch as th

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
from utils.mixer_state_importance import score_state_slot_calibration
import ozstar_submit_5m6m_calibration_mean_shared_3seeds as study
from ozstar_submit_5m6m_mixer_input_controls_9jobs import build_plans as controls
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def formulas():
    class Mixer:
        state_dim = 2
        def masks(self, s, mode):
            return s.new_ones(1, 2)
        def conditioned_value(self, q, s, group, mask):
            return ((s * mask)[..., :1] + 2 * (s * mask)[..., 1:2]) * q[..., :1]
    q = th.ones(1, 3, 1)
    s = th.tensor([[[1., 1.], [3., 1.], [9999., 9999.]]])
    valid = th.tensor([[[1.], [1.], [0.]]])
    g = th.tensor([[[3.], [5.], [0.]]])
    mean, count, before, after = score_state_slot_calibration(Mixer(), q, s, g, valid,
        perturbation='batch_mean', reference_valid=valid)
    assert count == 2 and th.equal(before, th.zeros(2))
    assert th.equal(after, th.tensor([1., 0.])) and th.allclose(mean, th.tensor([-1/17, 0.]))
    legacy, *_ = score_state_slot_calibration(Mixer(), q, s, g, valid)
    explicit, *_ = score_state_slot_calibration(Mixer(), q, s, g, valid, perturbation='mask_zero')
    assert th.equal(legacy, explicit) and th.allclose(legacy, th.tensor([-5/17, -4/17]))
    gain, *_ = score_state_slot_calibration(Mixer(), q, s, g * 0 + 4, valid,
                                           perturbation='batch_mean')
    assert th.allclose(gain, th.tensor([1/16, 0.]))  # Replacement can improve calibration.
    scaled, *_ = score_state_slot_calibration(Mixer(), q * 100, s, g * 100, valid,
                                             perturbation='batch_mean')
    assert th.allclose(mean, scaled)
    # Complete episodes select scoring probes, but field means still use all
    # valid TD samples like the Q-change control, never padding.
    s2 = th.tensor([[[1., 1.], [3., 1.], [8., 1.], [9999., 9999.]]])
    complete = th.tensor([[[1.], [1.], [0.], [0.]]])
    reference = th.tensor([[[1.], [1.], [1.], [0.]]])
    q2, g2 = th.ones(1, 4, 1), th.tensor([[[3.], [5.], [0.], [0.]]])
    result, _, _, perturbed = score_state_slot_calibration(Mixer(), q2, s2, g2, complete,
        perturbation='batch_mean', reference_valid=reference)
    assert th.allclose(perturbed, th.tensor([5., 0.]))  # reference first field = 4.
    assert th.allclose(result, th.tensor([-5/17, 0.]))
    calls = []
    class Grouped(Mixer):
        def masks(self, s, mode):
            return s.new_tensor([[1., 0.], [0., 1.]])
        def conditioned_value(self, q, s, group, mask):
            expected = s.new_tensor([[1., 0.], [0., 1.]])[group]
            assert th.equal(mask, expected), 'Batch-mean probes must not alter membership bits'
            calls.append((group, s.clone()))
            return super().conditioned_value(q, s, group, mask)
    score_state_slot_calibration(Grouped(), q, s, g, valid, perturbation='batch_mean')
    assert len(calls) == 4  # Per branch: one base and one active slot only.
    assert score_state_slot_calibration(Mixer(), q, s, g, valid * 0,
                                        perturbation='batch_mean') is None
    try:
        score_state_slot_calibration(Mixer(), q, s, g, valid,
                                      perturbation='batch_mean', reference_valid=valid * 0)
        raise AssertionError('Reference mask must contain every complete-episode probe')
    except ValueError:
        pass


def main():
    th.set_num_threads(1)
    formulas()
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    validate_config_keys(repo, plans)
    base = {p['seed']: p for p in controls(repo) if p['label'] == 'qmix_state_group_qchange'}
    for p in plans:
        new = extras(p)
        assert new.pop('mixer_calibration_perturbation') == 'batch_mean'
        assert new['mixer_input_control'] == 'group_calibration'
        new['mixer_input_control'] = 'group_qchange'
        assert new == extras(base[p['seed']]), 'No changes besides score choice/protocol selector'
        assert p['scene'] == 'smac_5m6m' and p['map_name'] == '5m_vs_6m'
        assert p['exports']['T_MAX'] == '10050000'
    with tempfile.TemporaryDirectory(prefix='gomarl-calibration-mean-QA-') as temp:
        learners, batches = [], []
        for criterion, perturbation in (('group_qchange', 'mask_zero'),
                                         ('group_calibration', 'batch_mean')):
            th.manual_seed(73)
            mac, _, batch, logger = make_case('baseline', '5m_vs_6m', production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control=criterion,
                    mixer_calibration_perturbation=perturbation, mixer_input_detach_unimportant=False,
                    slot_group_interval=1, slot_importance_samples=8))
            args = mac.args
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_' + criterion
            args.wandb_run_name, args.seed = 'SYNTHETIC_MC_MEAN_QA_NOT_EXPERIMENT', 1
            batch['filled'][1, 3] = 1
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            assert not learner.detach_unimportant and learner.other_optimiser is None
            assert learner.mixer.append_mask and learner.mixer.mixers[0].state_dim == 196
            assert learner.gate_l1_coef == learner.gate_variance_coef == 0
            learner.mixer.important_slots[:2] = True
            values = learner._rollout(mac, batch)
            chosen = values[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
            joint = learner.mixer.values(chosen, batch['state'][:, :-1], 'train')
            grads = th.autograd.grad(joint[..., 1].square().mean(), list(mac.parameters()), allow_unused=True)
            assert any(g is not None and g.abs().sum() > 0 for g in grads)
            learners.append(learner)
            batches.append(batch)
        assert all(th.equal(a, b) for a, b in zip(learners[0].params, learners[1].params))
        for learner, batch in zip(learners, batches):
            learner.train(batch, 10, 1)
        # Same data/initial model => identical TD losses and parameter update;
        # only the subsequent importance ranking/partition may differ.
        assert th.equal(learners[0].last_losses, learners[1].last_losses)
        assert all(th.equal(a, b) for a, b in zip(learners[0].params, learners[1].params))
        learner, batch = learners[1], batches[1]
        snapshot = json.loads(next((Path(temp) / 'mixer_input_control' / 'SYNTHETIC_group_calibration').rglob('step_*.json')).read_text())
        assert snapshot['calibration_perturbation'] == 'batch_mean'
        assert snapshot['calibration_probe_count'] > 0
        assert not snapshot['unimportant_agent_gradient_detached']
        assert 'group_scores' in snapshot and snapshot['append_membership_features']
        checkpoint = Path(temp) / 'checkpoint'
        checkpoint.mkdir()
        learner.save_models(checkpoint)
        learner.load_models(checkpoint)
        learner.mixer.calibration_perturbation = 'mask_zero'
        try:
            learner.load_models(checkpoint)
            raise AssertionError('Different scoring intervention must not resume the same run')
        except ValueError:
            pass
        learner.mixer.calibration_perturbation = 'batch_mean'
        partition, updates = learner.mixer.important_slots.clone(), learner.mixer.grouping_updates.clone()
        batch['terminated'].zero_()
        learner.train(batch, 20, 2)
        assert th.equal(partition, learner.mixer.important_slots)
        assert th.equal(updates, learner.mixer.grouping_updates)
        assert learner.mixer.calibration_probe_count.item() == 0
        incomplete = sorted((Path(temp) / 'mixer_input_control' / 'SYNTHETIC_group_calibration').rglob('step_*.json'))[-1]
        assert not json.loads(incomplete.read_text())['calibration_score_available']
        import ozstar_calibration_suite_sync as sync
        import ozstar_backfill_grouping_heatmaps as media
        import ozstar_clean_except_qmix_variants as storage
        import plot_5m6m_8m9m_qmix_variants_3seeds as plots
        assert study.GROUP in sync.GROUPS and study.GROUP in storage.KEEP_GROUPS
        assert set(p['job_name'] for p in plans) <= set(media.wanted_plans(repo))
        assert set(p['job_name'] for p in plans) <= set(p['job_name'] for p in plots.build_plans(repo))
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
            return str(92000 + calls.count('submit'))
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch('ozstar_submit_5m6m_id_kl80_3seeds.validate_installed_map'), \
                 patch.object(study.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(study.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(study.submitter, 'active_jobs', return_value=retained), \
                 patch.object(study.submitter, 'completed_jobs', return_value={}), \
                 patch.object(study.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(study.submitter, 'run', side_effect=command):
                study.main()
                assert calls.count('preflight') == calls.count('submit') == 3
                record = json.loads(next(Path(temp).glob(study.GROUP + '_*.json')).read_text())
                retained.update(record['submitted'])
                calls.clear()
                study.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: exact MC-MSE mean replacement/sign/normalization; fixed membership; '
          'original valid reference vs complete MC probes; unchanged first TD update; '
          'shared auxiliary actor gradient; skip incomplete returns; checkpoint guard; '
          'three seeds/10M; sync/media/plot/storage registration; mock Slurm idempotence')


if __name__ == '__main__':
    main()
