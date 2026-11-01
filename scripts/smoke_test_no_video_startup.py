#!/usr/bin/env python3
"""Reproduce the real pre-SC2 startup gate for all18 configs; no jobs or SC2."""
import copy
import logging
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace as SN
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import ozstar_submit_3s5z_refresh_and_warmup_18jobs as suite
from run.run import args_sanity_check, run_sequential, r_REGISTRY
from utils.battle_video import BattleVideoSession
from utils.logging import Logger
from utils.visualization_requirements import preflight_visualizations, record_visualization_inventory


class RunnerReached(Exception):
    pass


def config_for(plan):
    config = {}
    for path in ('default.yaml', 'envs/sc2.yaml', 'algs/clean_hyper.yaml'):
        config.update(yaml.safe_load((ROOT/'src/config'/path).read_text()))
    for entry in shlex.split(plan['exports']['EXTRA_ARGS']):
        key, value = entry.split('=', 1)
        if '.' in key:
            parent, leaf = key.split('.', 1)
            config[parent][leaf] = yaml.safe_load(value)
        else:
            config[key] = yaml.safe_load(value)
    config.update(batch_size_run=int(plan['exports']['BATCH_SIZE_RUN']),
        use_cuda=yaml.safe_load(plan['exports']['USE_CUDA']),
        seed=plan['seed'], name=plan['exports']['GROUP_NAME'], wandb_run_name=plan['job_name'])
    config['env_args']['map_name'] = plan['map_name']
    return config


def expect_error(args, error, fragment):
    try:
        preflight_visualizations(args)
    except error as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError('Expected preflight failure: '+fragment)


def main():
    log = logging.getLogger('SYNTHETIC-no-video-startup')
    plans = suite.build_plans(ROOT)
    assert len(plans) == 18
    for p in plans:
        raw = config_for(p)
        assert raw['test_visualizations_required'] and raw['test_policy_importance']
        config = args_sanity_check(raw, log)
        args = SN(**config)
        assert args.disable_video_media and not args.test_battle_videos
        # Call actual run_sequential: the very first call is the preflight that
        # crashed production. Stop immediately AFTER it at runner construction.
        with patch('utils.visualization_requirements.check_video_dependencies',
                   side_effect=AssertionError('No encoder check allowed')), \
                patch.dict(r_REGISTRY, {args.runner: lambda **kw: (_ for _ in ()).throw(RunnerReached())}):
            try:
                run_sequential(args, Logger(log))
            except RunnerReached:
                pass
            else:
                raise AssertionError('Production startup did not reach runner construction')
    base = SN(**args_sanity_check(config_for(plans[0]), log))
    # Video parameters are unused; static diagnostics must STILL be checked.
    no_video = copy.deepcopy(base)
    no_video.test_battle_video_interval = no_video.test_battle_video_episodes = 0
    with patch('utils.visualization_requirements.check_video_dependencies',
               side_effect=AssertionError('No encoder check allowed')):
        preflight_visualizations(no_video)
        bad = copy.deepcopy(base)
        bad.test_policy_importance = False
        expect_error(bad, ValueError, 'test_policy_importance')
        bad = copy.deepcopy(base)
        bad.test_policy_importance_interval = 0
        expect_error(bad, ValueError, 'interval')
        bad = copy.deepcopy(base)
        bad.test_policy_importance_episodes = 10000
        expect_error(bad, ValueError, 'episodes')
    # Explicit video opt-in retains encoder checks; this is NOT a global bypass.
    video = copy.deepcopy(base)
    video.disable_video_media, video.test_battle_videos = False, True
    with patch('utils.visualization_requirements.check_video_dependencies', side_effect=ImportError('encoder')):
        expect_error(video, RuntimeError, 'imageio')
    with patch('utils.visualization_requirements.check_video_dependencies') as check:
        preflight_visualizations(video)
        check.assert_called_once_with()
    with patch.dict(sys.modules, {'matplotlib.pyplot': None}):
        expect_error(base, RuntimeError, 'matplotlib')
    # Disabled video sessions can still participate in a required image inventory.
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-no-video-inventory-') as temp:
        base.local_results_path, base.unique_token = temp, 'SYNTHETIC_fixture'
        logger = Logger(log)
        videos = BattleVideoSession(base, logger)
        videos.begin(1000000)
        assert not videos.enabled and not videos.active
        policy = SN(due=True, t_env=1000000, error=None,
            files=dict(parameter_sensitivity='synthetic.png', observation_sensitivity='synthetic2.png'))
        hyper = SN(enabled=False, due=False, error=None)
        record_visualization_inventory(base, logger, videos, policy, hyper)
        assert len(list(Path(temp).rglob('*.json'))) == 1
    print('PASS: all18 production startup preflights pass with no videos/encoder; '
          'static image requirements and explicit video opt-in remain enforced; no SC2/jobs/uploads')


if __name__ == '__main__':
    main()
