#!/usr/bin/env python3
"""Both maps, independent input/gradient factors, native media, mock Slurm."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th
import wandb
import yaml

from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner
import ozstar_submit_calibration_mean_ablation_12jobs as study
import ozstar_submit_8m9m_ones_gate_regularization_9jobs as common
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(x.split('=', 1) for x in shlex.split(plan['exports']['EXTRA_ARGS']))


def main():
    th.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    plans = study.build_plans(repo)
    assert len(plans) == len({p['job_name'] for p in plans}) == 12
    validate_config_keys(repo, plans)
    for scene, map_name in (('smac_5m6m', '5m_vs_6m'), ('smac_8m9m', '8m_vs_9m')):
        references = {p['seed']: p for p in study.reference_plans(repo, scene, map_name)}
        selected = [p for p in plans if p['scene'] == scene]
        assert len(selected) == 6
        for label in study.LABELS:
            assert {p['seed'] for p in selected if p['label'] == label} == {1, 2, 3}
        for plan in selected:
            new = extras(plan)
            if plan['label'].endswith('_isolated'):
                assert new.pop('mixer_input_detach_unimportant') == 'True'
                assert 'mixer_input_append_mask' not in new
            else:
                assert new.pop('mixer_input_append_mask') == 'False'
                assert 'mixer_input_detach_unimportant' not in new
            assert new == extras(references[plan['seed']]), 'Confounded ablation settings'
            assert new['mixer_input_control'] == 'group_calibration'
            assert new['mixer_calibration_perturbation'] == 'batch_mean'
            assert plan['map_name'] == map_name and plan['exports']['T_MAX'] == '10050000'
            assert plan['exports']['RUN_NAME'] == plan['job_name']
            assert plan['exports']['GROUP_NAME'] == study.GROUP
            assert '--job-name=' + plan['job_name'] in plan['sbatch_args']
    with tempfile.TemporaryDirectory(prefix='gomarl-two-map-MC-ablation-QA-') as temp:
        for plan in (p for p in plans if p['seed'] == 1):
            isolated = plan['label'].endswith('_isolated')
            append = isolated
            overrides = {key: yaml.safe_load(value) for key, value in extras(plan).items()}
            overrides.update(clean_model_type='qmix_minimal', slot_group_interval=1, slot_importance_samples=8)
            th.manual_seed(89)
            mac, _, batch, logger = make_case('baseline', plan['map_name'], production_shapes=True,
                                             config_overrides=overrides)
            args = mac.args
            tag = plan['scene'] + '_' + plan['label']
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_' + tag
            args.wandb_run_name, args.seed = 'SYNTHETIC_MC_ABLATION_QA_NOT_EXPERIMENT_' + tag, 1
            saved = []
            logger.use_wandb, logger.wandb_current_t, logger.wandb_current_data = True, -1, {}
            logger.wandb = SimpleNamespace(id='SYNTHETIC_QA', summary={}, log=lambda *a, **kw: None)
            logger.wandb_module = SimpleNamespace(Image=wandb.Image, save=lambda path, **kw: saved.append(Path(path)))
            batch['filled'][1, 3] = 1
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            mixer = learner.mixer
            dim = 98 if plan['scene'] == 'smac_5m6m' else 179
            assert args.state_shape == dim and mixer.append_mask == append
            assert all(core.state_dim == dim * (2 if append else 1) for core in mixer.mixers)
            assert learner.detach_unimportant == isolated
            assert (learner.other_optimiser is not None) == isolated
            assert learner.gate_l1_coef == learner.gate_variance_coef == 0
            assert mixer.calibration_perturbation == 'batch_mean'
            mixer.important_slots[:3] = True
            state = batch['state'][:, :1].detach().clone().requires_grad_(True)
            q = th.randn(batch.batch_size, 1, args.n_agents)
            mask = mixer.masks(state)[0]
            observed = mixer.controlled_input(state, mask)
            assert th.equal(observed[..., :dim], state * mask)
            assert observed.shape[-1] == dim * (2 if append else 1)
            baseline = mixer.conditioned_value(q, state, 0, mask)
            altered = state.detach().clone()
            altered[..., 3:] += 1000
            assert th.equal(baseline.detach(), mixer.conditioned_value(q, altered, 0, mask))
            gradient = th.autograd.grad(baseline.sum(), state)[0]
            assert th.count_nonzero(gradient[..., 3:]) == 0
            values = learner._rollout(mac, batch)
            chosen = values[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
            joint = mixer.values(chosen, batch['state'][:, :-1], 'train', detach_unimportant=isolated)
            actor_grad = th.autograd.grad(joint[..., 1].square().mean(), list(mac.parameters()), allow_unused=True)
            blocked = all(g is None or th.count_nonzero(g) == 0 for g in actor_grad)
            assert blocked == isolated
            before = [[p.detach().clone() for p in core.parameters()] for core in mixer.mixers]
            learner.train(batch, 10, 1)
            assert learner.last_losses.numel() == 2 and th.isfinite(learner.last_losses).all()
            assert all(any(not th.equal(a, b) for a, b in zip(old, core.parameters()))
                       for old, core in zip(before, mixer.mixers))
            assert mixer.calibration_probe_count.item() > 0
            assert th.equal(mixer.important_slots, learner.target_mixer.important_slots)
            record = json.loads(saved[-1].read_text())
            assert record['calibration_perturbation'] == 'batch_mean'
            assert record['unimportant_agent_gradient_detached'] == isolated
            assert record['append_membership_features'] == append
            assert record['mixer_condition_dim'] == dim * (2 if append else 1)
            assert record['calibration_score_available']
            assert record['score_important_membership'][:4] == [True, True, True, False]
            assert all(value == 0 for rows in record['group_scores'].values()
                       for row, row_mask in zip(rows, record['score_masks'])
                       for value, keep in zip(row, row_mask) if not keep)
            assert isinstance(logger.wandb_current_data['mixer_input/grouping_heatmap'], wandb.Image)
            assert logger.wandb_current_data['mixer_input/grouping_membership']['nrows'] == dim
            checkpoint = Path(temp) / tag
            checkpoint.mkdir()
            learner.save_models(checkpoint)
            learner.load_models(checkpoint)
            learner.detach_unimportant = not isolated
            try:
                learner.load_models(checkpoint)
                raise AssertionError('Wrong gradient route must not resume')
            except ValueError:
                pass
            learner.detach_unimportant = isolated
            mixer.append_mask = not append
            try:
                learner.load_models(checkpoint)
                raise AssertionError('Wrong input encoding must not resume')
            except ValueError:
                pass
            mixer.append_mask = append
            print('PASS: {} {}; real dimensions/gradient route/input/masking/TD/media/checkpoint'.format(plan['scene'], plan['label']))
        import ozstar_calibration_suite_sync as sync
        import ozstar_backfill_grouping_heatmaps as heatmaps
        import ozstar_clean_except_qmix_variants as storage
        import plot_5m6m_8m9m_qmix_variants_3seeds as plots
        names = {p['job_name'] for p in plans}
        assert study.GROUP in sync.GROUPS and study.GROUP in storage.KEEP_GROUPS
        assert names <= set(heatmaps.wanted_plans(repo))
        assert names <= {p['job_name'] for p in plots.build_plans(repo)}
        # No reference models are silently included in the requested 12 jobs.
        assert not any(p['label'].endswith('_mean_shared') for p in plans)
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
            assert command[:2] == ['sbatch', '--parsable'] and calls.count('preflight') == 12
            calls.append('submit')
            return str(93000 + calls.count('submit'))
        cwd = Path.cwd()
        try:
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, SUBMIT='YES', REPO_DIR=str(repo), RUNTIME_ROOT=temp), \
                 patch.object(common, 'validate_maps'), \
                 patch.object(common.submitter, 'home_quota_free_gib', return_value=2.1), \
                 patch.object(common.submitter, 'guard_and_route', return_value={'logs': Path(temp)}), \
                 patch.object(common.submitter, 'active_jobs', return_value=retained), \
                 patch.object(common.submitter, 'completed_jobs', return_value={}), \
                 patch.object(common.submitter.subprocess, 'run', side_effect=preflight), \
                 patch.object(common.submitter, 'run', side_effect=command):
                study.main()
                assert calls.count('preflight') == calls.count('submit') == 12
                record = json.loads(next(Path(temp).glob(study.GROUP + '_*.json')).read_text())
                retained.update(record['submitted'])
                calls.clear()
                study.main()
                assert not calls
        finally:
            os.chdir(cwd)
    print('PASS: 12 exact names, 2 maps x 2 independent factors x 3 seeds, 10M; '
          'same MC-MSE and mean perturbation; registry coverage; no extra reference jobs; '
          'mock Slurm preflight/idempotence; no simulator or cloud writes')


if __name__ == '__main__':
    main()
