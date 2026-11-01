#!/usr/bin/env python3
"""No network, simulator, uploads or deletions; verify video-only policy."""
import ast
from copy import deepcopy
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from utils.video_media_policy import apply_video_media_policy
from utils.logging import Logger
from utils.battle_video import BattleVideoSession


def main():
    config = dict(test_battle_videos=True, battle_trace_make_video=True,
                  wandb_test_trajectory_video=True, test_visualizations_required=True,
                  test_policy_importance=True, test_hyper_obs_importance=True,
                  test_nepisode=32, seed=2, mixer_state_refresh_stage_steps=2000000)
    before = deepcopy(config)
    assert apply_video_media_policy(config) is config
    expected = dict(before, test_battle_videos=False, battle_trace_make_video=False,
                    wandb_test_trajectory_video=False)
    assert config == expected
    opt_in = dict(before, disable_video_media=False)
    assert apply_video_media_policy(deepcopy(opt_in)) == opt_in

    # Disabled capture needs no encoder, produces no clips and requests no rollout.
    logger = Logger(logging.getLogger('no-video-policy-fixture'))
    session = BattleVideoSession(SimpleNamespace(**config, env='sc2',
        local_results_path='/not-created-policy-fixture', unique_token='fixture', name='fixture'), logger)
    assert not session.enabled
    session.begin(2000000)
    session.request(SimpleNamespace())
    session.consume(SimpleNamespace())

    emitted = []
    def forbidden_video(*args, **kwargs):
        raise AssertionError('Video construction/upload forbidden')
    logger.use_wandb = True
    logger.wandb_module = SimpleNamespace(Video=forbidden_video, Image=lambda p: ('image', p))
    logger._update_wandb_buffer = lambda key, value, step: emitted.append((key, value, step))
    paths = dict(video='fixture.mp4', relation_dynamics_video='fixture.gif',
                 battle_overview='overview.png', similarity='similarity.png', alignment='alignment.png')
    for minimal in (True, False):
        logger.wandb_minimal_logging = minimal
        logger.log_test_battle_video('fixture.mp4', 1, 1, {})
        logger.log_battle_trace_media(paths, 1)
    assert len(emitted) == 3 and all(value[0] == 'image' for _, value, _ in emitted)
    # Explicit opt-in is still supported; policy does not remove old diagnostics.
    logger.disable_video_media = False
    logger.wandb_module.Video = lambda *args, **kwargs: ('video', args[0])
    logger.log_test_battle_video('fixture.mp4', 2, 1, {})
    assert emitted[-1][1] == ('video', 'fixture.mp4')

    # Every production run route passes post-merge config through the policy.
    for filename, function in (('src/main.py', 'my_main'), ('src/run/run.py', 'args_sanity_check')):
        tree = ast.parse((ROOT/filename).read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == 'apply_video_media_policy' for n in ast.walk(node))
    print('PASS: old explicit video flags overridden; no capture/Video objects; images, '
          'evaluation and model settings preserved; explicit opt-in supported')


if __name__ == '__main__':
    main()
