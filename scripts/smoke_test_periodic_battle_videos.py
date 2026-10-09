#!/usr/bin/env python3
"""Simulator-free runner integration and MP4 encode; synthetic data, no uploads."""
import argparse
from collections import deque
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.transforms import OneHot
from envs import REGISTRY
from runners.parallel_runner import ParallelRunner, EnvironmentWorkerError, env_worker
from runners.episode_runner import EpisodeRunner
from utils.battle_video import BattleVideoSession, focus_groups, action_label, render_battle_video, draw_video_frame, video_bounds
from utils.logging import Logger


class FakeEnv:
    episode_limit = 15
    map_name = "synthetic_video_fixture"
    def __init__(self, index=0, **kwargs):
        self.index, self.t = index, 0
    def reset(self):
        self.t = 0
    def get_state(self):
        return np.full(8, self.t, dtype=np.float32)
    def get_obs(self):
        return np.full((5, 7), self.t, dtype=np.float32)
    def get_avail_actions(self):
        return np.ones((5, 8), dtype=int)
    def get_env_info(self):
        return dict(n_agents=5, n_actions=8, state_shape=8, obs_shape=7, episode_limit=15)
    def get_battle_snapshot(self):
        def unit(i, enemy=False):
            hp = max(0, 45-self.t*3-(i if enemy else 0))
            return dict(id=i, tag=i+(100 if enemy else 0), x=10+i*.8+(2 if enemy else -self.t*.1),
                        y=12+i*.7+(2 if enemy else 0), health=hp, health_max=45,
                        alive=hp > 0, shield=0, shield_max=0, can_heal=False)
        return dict(n_actions_no_attack=6, allies=[unit(i) for i in range(5)],
                    enemies=[unit(i, True) for i in range(2)])
    def step(self, actions):
        self.t += 1
        done = self.t == 3+self.index
        return 1., done, dict(battle_won=done and self.index % 2 == 0, episode_limit=False)
    def get_stats(self):
        return {}
    def close(self):
        pass


class FakeConn:
    def __init__(self, index):
        self.env, self.queue, self.commands = FakeEnv(index), deque(), []
    def send(self, message):
        cmd, payload = message
        self.commands.append(cmd)
        if cmd == "get_env_info":
            self.queue.append(self.env.get_env_info())
            return
        if cmd == "get_stats":
            self.queue.append({})
            return
        if cmd == "reset":
            self.env.reset()
        else:
            reward, done, info = self.env.step(payload)
        data = dict(state=self.env.get_state(), obs=self.env.get_obs(), avail_actions=self.env.get_avail_actions())
        if cmd != "reset":
            data.update(reward=reward, terminated=done, info=info)
        if cmd in {"reset", "step_snapshot", "step_trace"}:
            data["snapshot"] = self.env.get_battle_snapshot()
        self.queue.append(data)
    def poll(self, timeout):
        return bool(self.queue)
    def recv(self):
        return self.queue.popleft()


class FakeMAC:
    action_selector = SimpleNamespace(epsilon=0)
    def init_hidden(self, batch_size):
        pass
    def select_actions(self, batch, t_ep, t_env, bs=None, test_mode=False):
        count = batch.batch_size if bs is None else len(bs)
        return th.tensor([[6, 6, 6, 6, 7]]*count, dtype=th.long).reshape(count, 5)


def setup(runner):
    scheme = dict(state=dict(vshape=8), obs=dict(vshape=7, group="agents"),
                  avail_actions=dict(vshape=8, group="agents", dtype=th.int),
                  actions=dict(vshape=1, group="agents", dtype=th.long),
                  reward=dict(vshape=1), terminated=dict(vshape=1, dtype=th.uint8))
    runner.setup(scheme, {"agents": 5}, {"actions": ("actions_onehot", [OneHot(8)])}, FakeMAC())


def start_fake_workers(runner):
    runner.parent_conns = tuple(FakeConn(i) for i in range(runner.batch_size))
    runner.worker_conns = ()
    runner.ps = [SimpleNamespace(is_alive=lambda: True, exitcode=None) for _ in range(runner.batch_size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args_cli = parser.parse_args()
    logger = Logger(logging.getLogger("battle-video-smoke"))
    with tempfile.TemporaryDirectory(prefix="gomarl-battle-video-test-") as temp:
        args = SimpleNamespace(env="sc2", env_args=dict(map_name="synthetic_video_fixture"), batch_size_run=8,
                               test_nepisode=32, device="cpu", runner_log_interval=10000, seed=1,
                               name="synthetic_smoke_only", unique_token="fixture", local_results_path=temp,
                               test_battle_videos=True, test_battle_video_interval=1000000,
                               test_battle_video_episodes=10, test_battle_video_fps=6)
        with patch.object(ParallelRunner, "_start_workers", start_fake_workers):
            runner = ParallelRunner(args, logger)
        setup(runner)
        session = BattleVideoSession(args, logger)
        session.begin(999999)
        session.request(runner)
        assert runner.battle_video_request is None
        session.begin(1003210)
        runner.t_env = 1003210
        runner.log_train_stats_t = runner.t_env
        calls, captured, requested = [], [], []
        def fake_render(trace, path, fps):
            captured.append(trace)
            Path(path).write_bytes(b"synthetic stub -- never uploaded")
            return str(path)
        logger.log_test_battle_video = lambda path, t, ep, trace, fps: calls.append((t, ep))
        with patch("utils.battle_video.render_battle_video", side_effect=fake_render):
            for _ in range(4):
                session.request(runner)
                if runner.battle_video_request:
                    requested.append(runner.battle_video_request["count"])
                runner.run(test_mode=True)
                session.consume(runner)
            session.finish()
        assert requested == [8, 2]
        assert runner.t_env == 1003210
        assert session.collected == session.rendered == 10
        assert len(logger.stats["test_return_mean"]) == 1
        assert logger.stats["test_return_mean"][-1][1] == 6.5
        assert logger.stats["test_battle_won_mean"][-1][1] == .5
        assert all(conn.commands.count("reset") == 4 for conn in runner.parent_conns)
        assert [ep for _, ep in calls] == list(range(1, 11))
        for trace in captured:
            assert trace["frames"][0]["t"] == 0
            assert trace["frames"][-1]["actions"] is None
            assert trace["frames"][-1]["t"] == trace["episode_length"]
            assert len(trace["frames"]) == trace["episode_length"]+1
            assert trace["frames"][0]["snapshot"]["allies"][0]["health"] == 45
            assert trace["frames"][1]["snapshot"]["allies"][0]["health"] == 42
            assert focus_groups(trace["frames"][0]) == {0: [0, 1, 2, 3], 1: [4]}
        assert any(t["battle_won"] for t in captured) and any(not t["battle_won"] for t in captured)
        session.begin(1999999)
        assert not session.active
        session.begin(2000010)
        assert session.active
        runner.request_battle_videos(2, 2000010)
        runner.worker_run_retries = 1
        runner.worker_run_retry_delay = 0
        real_run = runner._run_once
        attempt = [0]
        def fail_once(test_mode=False):
            attempt[0] += 1
            if attempt[0] == 1:
                runner.battle_video_request = None
                raise EnvironmentWorkerError("synthetic worker timeout")
            return real_run(test_mode)
        with patch.object(runner, "_run_once", side_effect=fail_once), patch.object(runner, "_restart_workers"):
            runner.run(test_mode=True)
        assert len(runner.pop_battle_videos()) == 2
        with patch.dict(REGISTRY, {"sc2": FakeEnv}):
            single_args = SimpleNamespace(**vars(args))
            single_args.batch_size_run = 1
            single = EpisodeRunner(single_args, logger)
            setup(single)
            single.request_battle_videos(1, 1000000, 9)
            single.run(test_mode=True)
            single_trace = single.pop_battle_videos()[0]
            assert single_trace["episode_index"] == 9
            assert [f["t"] for f in single_trace["frames"]] == [0, 1, 2, 3]
            assert single_trace["frames"][-1]["actions"] is None
        real_logger = Logger(logging.getLogger("video-wandb-stub"))
        real_logger.use_wandb = True
        real_logger.wandb_current_t, real_logger.wandb_current_data = 1000000, {}
        real_logger.wandb_module = SimpleNamespace(Video=lambda path, **kwargs: (path, kwargs))
        for episode in range(1, 11):
            real_logger.log_test_battle_video("not_uploaded.mp4", 1000000, episode, captured[0])
        assert set(real_logger.wandb_current_data) == {"test_battle_video/episode_{:02d}".format(ep) for ep in range(1, 11)}
        assert real_logger._wandb_metric_allowed("test_battle_video/failed")
        assert action_label(0) == "no-op" and action_label(5) == "west"
        assert action_label(7, can_heal=True) == "heal A1"
        remote = SimpleNamespace()
        commands = iter([("reset", None), ("step_snapshot", [6]*5), ("close", None)])
        responses = []
        remote.recv = lambda: next(commands)
        remote.send = responses.append
        remote.close = lambda: None
        with patch.object(FakeEnv, "get_render_frame", create=True, side_effect=AssertionError("native render not requested")):
            env_worker(remote, SimpleNamespace(x=FakeEnv))
        assert len(responses) == 2
        assert responses[1]["snapshot"]["allies"][0]["health"] == 42
        assert "render_frame" not in responses[1]
        failed_session = BattleVideoSession(args, logger)
        failed_session.begin(3000010)
        runner.last_battle_videos = [captured[0]]
        with patch("utils.battle_video.render_battle_video", side_effect=RuntimeError("synthetic encode failure")):
            failed_session.consume(runner)
        failed_session.finish()
        assert failed_session.collected == 1 and failed_session.rendered == 0
        assert "error" in failed_session.inventory[0]
        trace = captured[0]
        trace["synthetic"] = True
        output = args_cli.output_dir or Path(temp)/"preview"
        output.mkdir(parents=True, exist_ok=True)
        numpy_rng, torch_rng = np.random.get_state(), th.random.get_rng_state().clone()
        video = render_battle_video(trace, output/"synthetic_smoke_preview.mp4", fps=6)
        assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
        assert th.equal(th.random.get_rng_state(), torch_rng)
        from PIL import Image
        Image.fromarray(draw_video_frame(trace["frames"][0], trace, video_bounds(trace["frames"]))).save(output/"synthetic_smoke_preview.png")
        import imageio.v2 as imageio
        with imageio.get_reader(video) as reader:
            assert reader.get_data(0).shape == (720, 1120, 3)
            assert reader.count_frames() == len(trace["frames"])
        assert not list(Path(temp).rglob("*_frames"))
        inventory = json.loads((session.output_root/"step_001000000/inventory.json").read_text())
        assert len(inventory) == 10
        print("PASS: 8+2 episodes from 32 tests, wins/losses unselected, aligned snapshots/actions/terminal, retry, single/parallel runners, 10 W&B keys, real MP4")
        if args_cli.output_dir:
            print("Synthetic visual QA only: " + str(video))


if __name__ == "__main__":
    main()
