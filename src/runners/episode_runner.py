from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import copy
import numpy as np


def _extract_role_snapshot(mac):
    role_features = getattr(mac, "group_struct_features", None)
    role_probs = getattr(mac, "group_probs", None)
    role_prototypes = getattr(mac, "group_role_prototypes", None)
    if role_features is None or role_probs is None or role_prototypes is None:
        return None
    return {
        "role_features": copy.deepcopy(role_features[0].detach().cpu().numpy()),
        "role_probs": copy.deepcopy(role_probs[0].detach().cpu().numpy()),
        "role_prototypes": copy.deepcopy(role_prototypes.detach().cpu().numpy()),
    }


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
        self.last_test_viz_trace = None
        self.last_test_group = None

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.current_test_viz_trace = None
        if test_mode and getattr(self.args, "visualize_group_graph", False):
            self.current_test_viz_trace = []
            viz_info = self.env.get_group_viz_info()
            if viz_info is not None:
                self.current_test_viz_trace.append({"viz_info": viz_info, "group": None})

    def run(self, test_mode=False):
        self.reset(test_mode=test_mode)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            if self.current_test_viz_trace is not None:
                viz_info = self.env.get_group_viz_info()
                if viz_info is not None:
                    current_groups = getattr(self.mac, "current_groups", None)
                    current_group = copy.deepcopy(current_groups[0]) if current_groups is not None else None
                    role_snapshot = _extract_role_snapshot(self.mac)
                    frame = {"viz_info": viz_info, "group": current_group}
                    if role_snapshot is not None:
                        frame.update(role_snapshot)
                    self.current_test_viz_trace.append(frame)

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

        if test_mode:
            self.last_test_viz_trace = self.current_test_viz_trace
            current_groups = getattr(self.mac, "current_groups", None)
            self.last_test_group = copy.deepcopy(current_groups[0]) if current_groups is not None else None

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
