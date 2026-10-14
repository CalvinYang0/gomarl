#!/usr/bin/env python3
"""Actual learner updates for both missing factorial cells; no simulator."""
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import wandb
import torch as th
from smoke_test_counter_transformer_nine import make_case
from learners.mixer_input_learner import MixerInputLearner


def main():
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp:
        for score, isolated in (('group_qchange', True), ('group_calibration', False)):
            th.manual_seed(47)
            mac, _, batch, logger = make_case('baseline', '5m_vs_6m', production_shapes=True,
                config_overrides=dict(clean_model_type='qmix_minimal', mixer_input_control=score,
                    mixer_input_detach_unimportant=isolated, slot_group_interval=1, slot_importance_samples=8))
            args = mac.args
            args.local_results_path, args.unique_token = temp, 'SYNTHETIC_ABLATION_' + score
            args.wandb_run_name, args.seed = 'SYNTHETIC_ABLATION_NOT_EXPERIMENT', 1
            # Exercise actual W&B media construction and buffering, without any
            # network writes or production run identities.
            saved, buffered = [], []
            logger.use_wandb = True
            logger.wandb_current_t, logger.wandb_current_data = -1, {}
            logger.wandb = SimpleNamespace(id='synthetic-' + score, summary={},
                log=lambda data, step: buffered.append((step, data.copy())))
            logger.wandb_module = SimpleNamespace(Image=wandb.Image, Table=wandb.Table,
                save=lambda path, **kw: saved.append(Path(path)))
            batch['filled'][1, 3] = 1
            learner = MixerInputLearner(mac, batch.scheme, logger, args)
            values = learner._rollout(mac, batch)
            chosen = values[:, :-1].gather(-1, batch['actions'][:, :-1].long()).squeeze(-1)
            joint = learner.mixer.values(chosen, batch['state'][:, :-1], 'train', detach_unimportant=isolated)
            grads = th.autograd.grad(joint[..., 1].square().mean(), list(mac.parameters()), allow_unused=True)
            zero = all(g is None or th.count_nonzero(g) == 0 for g in grads)
            assert zero == isolated, 'Wrong auxiliary agent gradient routing for ' + score
            assert (learner.other_optimiser is not None) == isolated
            learner.train(batch, 10, 1)
            assert learner.last_losses.numel() == 2 and th.isfinite(learner.last_losses).all()
            assert learner.mixer.grouping_updates.item() == 1
            if score == 'group_calibration':
                assert learner.mixer.calibration_probe_count.item() > 0
            assert th.equal(learner.mixer.important_slots, learner.target_mixer.important_slots)
            media = logger.wandb_current_data
            assert isinstance(media['mixer_input/grouping_heatmap'], wandb.Image)
            descriptor = media['mixer_input/grouping_membership']
            assert descriptor['_type'] == 'table-file' and descriptor['nrows'] == 98
            table_path = next(p for p in saved if p.name == descriptor['path'])
            contents = json.loads(table_path.read_text())
            assert len(wandb.Table.from_json(contents, None).data) == 98
            assert logger.wandb.summary['mixer_input/grouping_t_env'] == 10
            assert saved and json.loads(saved[-1].read_text())['source_run_id'] == 'synthetic-' + score
            assert len(logger.wandb.summary['mixer_input/important_slot_names']) == int(learner.mixer.important_slots.sum())
            print('PASS: {} detach={} real 5m6m shapes; auxiliary actor gradient; two TD losses; grouping; target partition'.format(score, isolated))


if __name__ == '__main__':
    main()
