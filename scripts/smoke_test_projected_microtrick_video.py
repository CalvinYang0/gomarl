#!/usr/bin/env python3
"""Synthetic real-TD, freeze/checkpoint and first-one-video checks; no cloud/SC2."""
import copy
import logging
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_projected_microtrick_video_18jobs as suite
from learners import REGISTRY
from smoke_test_counter_transformer_nine import make_case
from smoke_test_no_video_startup import config_for
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from smoke_test_periodic_battle_videos import start_fake_workers, setup
from runners.parallel_runner import ParallelRunner
from utils.battle_video import BattleVideoSession, render_battle_video, draw_video_frame, video_bounds
from utils.logging import Logger
from utils.mixer_state_refresh import scheduled_segments, begin, extra_scheme
from utils.visualization_requirements import preflight_visualizations
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys


def extras(plan):
    return dict(v.split('=', 1) for v in shlex.split(plan['exports']['EXTRA_ARGS']))


def equal_state(a, b):
    return a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)


def held_batch(raw, args):
    scheme = {k: v for k, v in raw.scheme.items() if k not in {'filled', 'actions_onehot'}}
    scheme.update(extra_scheme(args, args.state_shape, 7))
    batch = EpisodeBatch(scheme, raw.groups, raw.batch_size, raw.max_seq_length,
        preprocess={'actions': ('actions_onehot', [OneHot(args.n_actions)])})
    batch.update({k: v.clone() for k, v in raw.data.transition_data.items()
                  if k != 'actions_onehot'}, mark_filled=False)
    for idx in range(raw.batch_size):
        cache = begin(args, 7, 1, (idx+1)*2000000)
        for t in range(raw.max_seq_length):
            data = cache.apply(dict(state=[raw['state'][idx, t].numpy()]), t)
            batch.update(data, bs=[idx], ts=t, mark_filled=False)
    return batch


def make(plan, temp):
    config = config_for(plan)
    config['clean_model_type'] = plan['exports']['MODEL_TYPE']
    th.manual_seed(99)
    mac, _, raw, logger = make_case('baseline', plan['map_name'], production_shapes=True,
                                   config_overrides=config)
    args = mac.args
    args.local_results_path, args.unique_token = temp, plan['job_name']
    args.wandb_run_name = 'SYNTHETIC_QA_NOT_EXPERIMENT'
    batch = held_batch(raw, args)
    learner = REGISTRY[args.learner](mac, batch.scheme, logger, args)
    learner.last_image_t = 10000000
    return learner, batch


def check_plans():
    plans = suite.build_plans(ROOT)
    assert len(plans) == 18
    assert {(p['scene'], p['label'], p['seed']) for p in plans} == {
        (m[0], v[0], s) for m in suite.MAPS for v in suite.VARIANTS.values() for s in (1, 2, 3)}
    for p in plans:
        base = next(q for q in suite.parent.build_plans(ROOT, 'smac_5m6m') if q['seed'] == p['seed'])
        x, old = extras(p), extras(base)
        _, initial, full, freeze = next(v for v in suite.VARIANTS.values() if v[0] == p['label'])
        old.update(mixer_input_initial_probability=initial, mixer_state_refresh_full_steps=full,
            disable_video_media='False', test_battle_videos='True', test_battle_video_interval='1000000',
            test_battle_video_episodes='1', test_battle_video_fps='6')
        if freeze:
            old.update(learner='projected_onefill_freeze_refresh_learner', mixer_soft_gate_freeze_threshold='0.9')
        assert x == old, 'Unrequested hyperparameter change'
        assert '--time=2-00:00:00' in p['sbatch_args']
        assert '--time=1-00:00:00' not in p['sbatch_args']
        for k in ('MODEL_TYPE', 'T_MAX', 'TEST_INTERVAL', 'BATCH_SIZE_RUN', 'USE_CUDA'):
            assert p['exports'][k] == base['exports'][k]
        assert p['exports']['MODEL_TYPE'] == 'qmix_minimal'
    validate_config_keys(ROOT, plans)
    for plan in plans:
        args = SN(**config_for(plan))
        preflight_visualizations(args)  # Real encoder dependency check, no SC2 launch.
        assert not args.disable_video_media and args.test_battle_videos
        assert args.test_battle_video_episodes == 1 and args.test_battle_video_interval == 1000000
    assert len(suite.build_plans(ROOT, 'smac_corridor')) == 9
    assert len(suite.build_plans(ROOT, variant='freeze5m')) == 6
    for bad in ('smac_3s5z', 'grf_counter', 'smac_5m6m'):
        try:
            suite.build_plans(ROOT, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unexpected map accepted')
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(suite, 'submit_suite') as submit:
        suite.main(['--scene', 'smac_corridor', '--variant', 'freeze5m'])
        assert len(submit.call_args.args[2](ROOT)) == 3
    return plans


def check_td(scene, temp):
    for option in suite.VARIANTS:
        plan = suite.build_plans(ROOT, scene, option)[0]
        learner, batch = make(plan, temp)
        initial = float(extras(plan)['mixer_input_initial_probability'])
        m, target = learner.mixer, learner.target_mixer
        assert not hasattr(m, 'gate_logits') and th.all(m.coefficients() == initial)
        assert th.allclose(m.controlled_input(batch['state'], m.gate_values(batch['state'])),
                           initial*batch['state']+(1-initial))
        learner.train(batch, 1000000, 1)
        assert m.gate_parameter.grad is not None
        assert th.all((m.coefficients() >= 0) & (m.coefficients() <= 1))
        if option != 'freeze5m':
            assert scheduled_segments(learner.args, 5000000, 170) < 170
            assert scheduled_segments(learner.args, 8000000, 170) == 170
            learner.train(batch, 6000000, 2)
            assert m.gate_parameter.requires_grad
            continue
        assert learner.freeze_steps == 5000000
        assert scheduled_segments(learner.args, 5000000, 170) == 170
        learner._before_training_update(4999999)
        assert m.freeze_t_env.item() == -1
        with th.no_grad():
            m.gate_parameter.fill_(.5)
            m.gate_parameter[:3] = th.tensor([.89, .90, .91])
            target.gate_parameter.fill_(.99)
        actor_before = copy.deepcopy(learner.mac.agent.state_dict())
        target_before = copy.deepcopy(target.base.state_dict())
        base_before = copy.deepcopy(m.base.state_dict())
        learner.train(batch, 5000003, 2)
        assert m.coefficients()[:3].tolist() == [0, 0, 1]
        assert m.coefficients().sum() == 1
        assert th.equal(m.coefficients(), target.coefficients())
        assert m.freeze_t_env.item() == target.freeze_t_env.item() == 5000003
        assert not m.gate_parameter.requires_grad and m.gate_parameter not in learner.optimiser.state
        assert not target.gate_parameter.requires_grad
        assert not equal_state(actor_before, learner.mac.agent.state_dict())
        assert not equal_state(base_before, m.base.state_dict())
        assert equal_state(target_before, target.base.state_dict()), 'Unrequested target sync at freeze'
        mask = m.coefficients().clone()
        learner._update_targets()
        learner.train(batch, 6000000, 3)
        assert th.equal(m.coefficients(), mask)
        directory = Path(temp)/(scene+'_checkpoint')
        directory.mkdir()
        learner.save_models(directory)
        restored, replay = make(plan, temp)
        restored.load_models(directory)
        restored.train(replay, 7000000, 4)
        assert th.equal(restored.mixer.coefficients(), mask)
        assert not restored.mixer.gate_parameter.requires_grad
    print('PASS actual TD / projection / exact freeze90 / other weights train / checkpoint:', scene)


def check_one_video(temp):
    args = SN(env='sc2', env_args=dict(map_name='SYNTHETIC_NOT_EXPERIMENT'), batch_size_run=8,
        test_nepisode=32, device='cpu', runner_log_interval=10000, seed=1, name='synthetic',
        unique_token='fixture', local_results_path=temp, test_battle_videos=True,
        test_battle_video_interval=1000000, test_battle_video_episodes=1, test_battle_video_fps=6)
    logger = Logger(logging.getLogger('SYNTHETIC-one-video'))
    with patch.object(ParallelRunner, '_start_workers', start_fake_workers):
        runner = ParallelRunner(args, logger)
    setup(runner)
    session = BattleVideoSession(args, logger)
    calls, captured = [], []
    logger.log_test_battle_video = lambda path, t, ep, trace, fps: calls.append((t, ep))
    def encode(trace, path, fps):
        captured.append(trace)
        return str(path)
    for milestone in (1000000, 2000000):
        session.begin(milestone-1)
        assert not session.active
        session.begin(milestone+123)
        session.due = session.active  # Same audit handoff as run/run.py.
        assert session.active and session.due
        runner.t_env = milestone+123
        requested = []
        with patch('utils.battle_video.render_battle_video', side_effect=encode):
            for _ in range(4):
                session.request(runner)
                if runner.battle_video_request:
                    requested.append(runner.battle_video_request['count'])
                runner.run(test_mode=True)
                session.consume(runner)
            session.finish()
        assert requested == [1] and session.collected == session.rendered == 1
        assert session.due and not session.active
        assert runner.t_env == milestone+123
        assert captured[-1]['episode_index'] == 0
        assert captured[-1]['frames'][-1]['actions'] is None
        session.begin(milestone+9999)
        session.due = session.active
        assert not session.active and not session.due
    assert calls == [(1000123, 1), (2000123, 1)]
    assert logger.stats['test_battle_won_mean'][-1][1] == .5  # Still all32 episodes.
    assert all(conn.commands.count('reset') == 8 for conn in runner.parent_conns)
    media = Logger(logging.getLogger('SYNTHETIC-media'))
    media.use_wandb, media.disable_video_media = True, False
    media.wandb_current_t, media.wandb_current_data = 1000123, {}
    media.wandb_module = SN(Video=lambda path, **kw: (path, kw))
    media.log_test_battle_video('SYNTHETIC_NOT_UPLOADED.mp4', 1000123, 1, captured[0])
    assert set(media.wandb_current_data) == {'test_battle_video/episode_01'}
    # Exercise the actual encoder/decoder without publishing synthetic media.
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    trace = copy.deepcopy(captured[0])
    trace['synthetic'] = True
    th_rng, np_rng = th.get_rng_state().clone(), np.random.get_state()
    video = render_battle_video(trace, Path(temp)/'SYNTHETIC_not_experiment.mp4')
    with imageio.get_reader(video) as reader:
        assert reader.count_frames() == len(trace['frames'])
        assert reader.get_data(0).shape == (720, 1120, 3)
    assert th.equal(th.get_rng_state(), th_rng)
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    preview = Path(os.environ.get('SYNTHETIC_VIDEO_PREVIEW_DIR', temp))
    preview.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_video_frame(trace['frames'][0], trace, video_bounds(trace['frames']))).save(
        preview/'SYNTHETIC_not_experiment.png')
    print('PASS exactly1 of32 normal tests per1M; first worker, terminal, media key; no new forwards/RNG')


def main():
    th.set_num_threads(1)
    check_plans()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-microtrick-video-') as temp:
        for scene, *_ in suite.MAPS:
            check_td(scene, temp)
        check_one_video(temp)
    print('PASS all18 configs; no simulator, submission or cloud upload')


if __name__ == '__main__':
    main()
