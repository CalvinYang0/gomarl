#!/usr/bin/env python3
"""Six-map plan and learner checks; no simulator launch or Slurm submission."""
from collections import Counter
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch as th
import yaml

import ozstar_submit_projected_faststages_six_maps_54jobs as suite
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys
from smoke_test_counter_transformer_nine import make_case
from smoke_test_projected_fastrefresh import model_check
from utils.mixer_state_refresh import scheduled_segments


def config_for(plan):
    exports = plan['exports']
    config = {}
    for path in ('default.yaml', 'envs/'+exports['ENV_CONFIG']+'.yaml', 'algs/clean_hyper.yaml'):
        config.update(yaml.safe_load((suite.ROOT/'src/config'/path).read_text()))
    for key, raw in suite.extras(exports['EXTRA_ARGS']).items():
        target = config
        if '.' in key:
            parent, key = key.split('.', 1)
            target = config[parent]
        target[key] = yaml.safe_load(raw)
    config['env_args']['map_name'] = exports['MAP_NAME']
    return config


def plan_checks():
    plans = suite.build_plans(suite.ROOT)
    assert len(plans) == len({p['job_name'] for p in plans}) == 54
    assert Counter(p['domain'] for p in plans) == {'smac': 27, 'grf': 27}
    assert {p['map_name'] for p in plans} == {'6h_vs_8z', '10m_vs_11m', '3s_vs_5z',
        'academy_counterattack_easy', 'academy_pass_and_shoot_with_keeper', 'academy_3_vs_1_with_keeper'}
    validate_config_keys(suite.ROOT, plans)
    for scene, name, domain, _, _ in suite.SCENES:
        subset = suite.build_plans(suite.ROOT, domain, scene)
        assert subset == [p for p in plans if p['scene'] == scene]
        assert {(p['label'], p['seed']) for p in subset} == {(m, s) for m in suite.LABELS for s in suite.SEEDS}
        for seed in suite.SEEDS:
            q, ones, variant = [p for p in subset if p['seed'] == seed]
            raw, constant, gated = map(config_for, (q, ones, variant))
            assert raw['qmix_condition_input'] == 'state' and constant['qmix_condition_input'] == 'ones'
            assert {k for k in raw if raw[k] != constant[k]} == {'qmix_condition_input'}
            assert raw['learner'] == constant['learner'] == 'clean_learner'
            assert not raw['mixer_state_refresh_enabled'] and not constant['mixer_state_refresh_enabled']
            assert gated['learner'] == 'projected_onefill_refresh_learner'
            assert gated['mixer_input_initial_probability'] == 1
            assert gated['mixer_onefill_objective'] == 'td' and not gated['mixer_onefill_residual']
            assert gated['mixer_state_refresh_enabled'] and gated['mixer_state_refresh_eval_live']
            assert gated['mixer_state_refresh_schedule'] == 'faster_stages'
            assert gated['mixer_state_refresh_full_steps'] == 8000000
            for key in ('optimizer', 'lr', 'gamma', 'td_lambda', 'double_q', 'test_nepisode', 'env_args'):
                assert raw[key] == gated[key], key
            for p, cfg in zip((q, ones, variant), (raw, constant, gated)):
                assert p['map_name'] == name and p['exports']['MAP_NAME'] == name
                assert p['win_metric'] == ('test_game_win_mean' if domain == 'grf' else 'test_battle_won_mean')
                assert p['exports']['MODEL_TYPE'] == 'qmix_minimal' and p['exports']['T_MAX'] == '10050000'
                assert '--time=1-00:00:00' in p['sbatch_args'] and '--mem=32G' in p['sbatch_args']
                assert cfg['disable_video_media'] and not cfg['test_battle_videos'] and not cfg['save_model']
                if domain == 'grf':
                    assert not any(cfg['env_args'][key] for key in ('write_video', 'write_goal_dumps',
                                                                   'write_full_episode_dumps', 'dump_freq', 'render'))
    with patch.dict(os.environ, SUBMIT='YES'), patch.object(suite.subprocess, 'run') as quota, \
            patch.object(suite, 'parse_home_quota', return_value={'free_gib': 5, 'free_inodes': 49999}), \
            patch.object(suite, 'submit_suite', side_effect=AssertionError('Quota guard bypass')):
        quota.return_value.stdout = 'SYNTHETIC quota'
        try:
            suite.main(['--scene', 'grf_counter'])
        except RuntimeError as exc:
            assert 'headroom' in str(exc)
        else:
            raise AssertionError('Inode shortage accepted')
    with patch.dict(os.environ, SUBMIT='NO'), patch.object(suite, 'submit_suite') as submit:
        suite.main(['--domain', 'grf'])
        assert len(submit.call_args.args[2](suite.ROOT)) == 27
    print('PASS: 54 plans; exact six maps; matched historical training configs; no-video and quota guards')
    return plans


def environment_checks():
    from pysc2 import maps

    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-sixmap-assets-') as temp:
        root = Path(temp)
        binary = root/'Versions/Base1/SC2_x64'
        binary.parent.mkdir(parents=True)
        binary.touch(mode=0o755)
        for _, name, _, _, _ in suite.selected_scenes('smac'):
            asset = root/'Maps'/maps.get(name).path
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.touch()
        with patch.dict(os.environ, SC2PATH=temp), redirect_stdout(io.StringIO()):
            suite.validate_environments(suite.selected_scenes('smac'))
            (root/'Maps'/maps.get('10m_vs_11m').path).unlink()
            try:
                suite.validate_environments(suite.selected_scenes('smac', 'smac_10m11m'))
            except RuntimeError as exc:
                assert '10m_vs_11m.SC2Map' in str(exc)
            else:
                raise AssertionError('Missing map asset accepted')
    with patch.object(suite.subprocess, 'run') as probe:
        suite.validate_environments(suite.selected_scenes('grf'))
        assert [call.args[0][-1] for call in probe.call_args_list] == [s[1] for s in suite.selected_scenes('grf')]
        assert all(call.kwargs['check'] and call.kwargs['timeout'] == 60 for call in probe.call_args_list)
    print('PASS: SC2 assets must exist at PySC2 paths; only selected GRF probes are called')


def model_checks(plans):
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-faststages-transfer-') as temp:
        for scene, name, domain, _, _ in suite.SCENES:
            q, ones, variant = [p for p in plans if p['scene'] == scene and p['seed'] == 1]
            pair = []
            for plan in (q, ones):
                args = config_for(plan)
                args['clean_model_type'] = 'qmix_minimal'
                th.manual_seed(71)
                pair.append(make_case('baseline', name, production_shapes=True, config_overrides=args))
            (mac, live, batch, _), (_, constant, _, _) = pair
            assert len(mac.agent.fixed_head) == 3 and isinstance(mac.agent.fixed_head[1], th.nn.ELU)
            assert all(th.equal(a, b) for a, b in zip(live.params, constant.params))
            q_i = th.rand(2, 5, mac.args.n_agents)
            state = batch['state']
            assert th.equal(live.mixer(q_i, th.ones_like(state)), constant.mixer(q_i, state))
            assert th.equal(constant.mixer(q_i, state+9), constant.mixer(q_i, state))
            for learner in (live, constant):
                learner.train(batch, 1, 1)
                assert all(th.isfinite(p).all() for p in learner.params)
            cfg = config_for(variant)
            limit = cfg['env_args']['time_limit'] if domain == 'grf' else _smac_limit(name)
            args = SimpleNamespace(**cfg)
            assert scheduled_segments(args, 0, limit) == 1
            assert 1 < scheduled_segments(args, 4000000, limit) < limit
            assert scheduled_segments(args, 8000000, limit) == limit
            assert scheduled_segments(args, 0, limit, test_mode=True) == limit
            model_check(variant, cfg['optimizer'], temp)


def _smac_limit(name):
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    return get_smac_map_registry()[name]['limit']


if __name__ == '__main__':
    environment_checks()
    model_checks(plan_checks())
    print('PASS: all six map dimensions for all three models; synthetic only; no jobs or uploads')
