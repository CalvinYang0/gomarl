from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from utils.battle_trace import model_diagnostics
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -1000000
        self.battle_trace_request = None
        self.last_battle_trace = None

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def request_battle_trace(self, prefix, t_env):
        self.battle_trace_request = {"prefix": prefix, "t_env": int(t_env)}

    def pop_battle_trace(self):
        trace = self.last_battle_trace
        self.last_battle_trace = None
        return trace

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset(test_mode=test_mode)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        snapshot_fn = getattr(self.env, "get_battle_snapshot", None)

        def safe_battle_snapshot():
            if snapshot_fn is None:
                return None
            try:
                return snapshot_fn()
            except Exception:
                return None

        trace_request = self.battle_trace_request if test_mode and snapshot_fn is not None else None
        self.battle_trace_request = None
        trace_frames = []
        snapshot = safe_battle_snapshot() if trace_request is not None else None

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            trajectory_projection = getattr(
                self.mac, "latest_trajectory_parameter_projection", None
            )
            if trajectory_projection is not None and not test_mode:
                self.batch.update(
                    {
                        "trajectory_parameter_projection": trajectory_projection
                        .detach()
                        .to(device="cpu", dtype=trajectory_projection.dtype)
                    },
                    ts=self.t,
                    mark_filled=False,
                )

            if trace_request is not None:
                trace_frames.append(
                    {
                        "t": int(self.t),
                        "t_env": int(trace_request["t_env"]),
                        "snapshot": snapshot,
                        "actions": cpu_actions[0].astype(int).tolist(),
                        "diagnostics": model_diagnostics(self.mac, batch_index=0),
                    }
                )

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            if trace_request is not None:
                snapshot = safe_battle_snapshot()

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        if trace_request is not None:
            trace_frames.append(
                {
                    "t": int(self.t),
                    "t_env": int(trace_request["t_env"]),
                    "snapshot": snapshot,
                    "actions": None,
                    "diagnostics": model_diagnostics(self.mac, batch_index=0),
                }
            )
            self.last_battle_trace = {
                "prefix": trace_request["prefix"],
                "t_env": int(trace_request["t_env"]),
                "map_name": getattr(self.env, "map_name", None),
                "frames": trace_frames,
            }
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if prefix == "test_" and hasattr(
            self.mac, "pop_test_gate_probability_summary"
        ):
            summary = self.mac.pop_test_gate_probability_summary()
            if summary is not None:
                for branch_name in ("linear", "attention"):
                    values = ", ".join(
                        "{}={:.4f}".format(slot_name, probability)
                        for slot_name, probability in zip(
                            summary["slot_names"], summary[branch_name]
                        )
                    )
                    self.logger.console_logger.info(
                        "Dynamic gate TEST {} slot probabilities | t_env={} | n={} | {}".format(
                            branch_name.upper(),
                            self.t_env,
                            summary["sample_count"],
                            values,
                        )
                    )
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
