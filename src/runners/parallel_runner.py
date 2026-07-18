from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
from utils.battle_trace import model_diagnostics
import numpy as np
import torch as th
import time
import traceback

class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.logger.console_logger.info(
            "ParallelRunner starting {} {} environment workers".format(
                self.batch_size, self.args.env
            )
        )
        self.worker_startup_stagger = float(
            getattr(self.args, "env_worker_startup_stagger", 0.0)
        )
        self.worker_reset_retries = int(
            getattr(self.args, "env_worker_reset_retries", 0)
        )
        self.worker_reset_retry_delay = float(
            getattr(self.args, "env_worker_reset_retry_delay", 1.0)
        )
        self.worker_response_timeout = float(
            getattr(self.args, "env_worker_response_timeout", 180.0)
        )
        self._initial_reset = True
        self._closed = False

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **self.args.env_args)),
                    self.worker_reset_retries,
                    self.worker_reset_retry_delay,
                ),
            )
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self._send_worker(0, "get_env_info", None)
        self.env_info = self._recv_worker(0, "get_env_info")
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.battle_trace_request = None
        self.last_battle_trace = None
        self.latest_snapshots = [None for _ in range(self.batch_size)]

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def request_battle_trace(self, prefix, t_env):
        self.battle_trace_request = {"prefix": prefix, "t_env": int(t_env)}

    def pop_battle_trace(self):
        trace = self.last_battle_trace
        self.last_battle_trace = None
        return trace

    def _send_worker(self, env_idx, command, data):
        process = self.ps[env_idx]
        if not process.is_alive():
            raise RuntimeError(
                "Environment worker {} exited before command {!r} "
                "(exitcode={}).".format(env_idx, command, process.exitcode)
            )
        try:
            self.parent_conns[env_idx].send((command, data))
        except (BrokenPipeError, EOFError, OSError) as err:
            raise RuntimeError(
                "Failed to send command {!r} to environment worker {} "
                "(alive={}, exitcode={}).".format(
                    command, env_idx, process.is_alive(), process.exitcode
                )
            ) from err

    def _recv_worker(self, env_idx, command):
        parent_conn = self.parent_conns[env_idx]
        process = self.ps[env_idx]
        if not parent_conn.poll(self.worker_response_timeout):
            raise RuntimeError(
                "Timed out after {:.1f}s waiting for environment worker {} "
                "to answer {!r} (alive={}, exitcode={}).".format(
                    self.worker_response_timeout,
                    env_idx,
                    command,
                    process.is_alive(),
                    process.exitcode,
                )
            )
        try:
            response = parent_conn.recv()
        except (EOFError, OSError) as err:
            raise RuntimeError(
                "Environment worker {} closed its pipe while handling {!r} "
                "(exitcode={}).".format(env_idx, command, process.exitcode)
            ) from err

        if isinstance(response, dict) and response.get("__worker_error__"):
            raise RuntimeError(
                "Environment worker {} failed while handling {!r}: {}\n{}".format(
                    env_idx,
                    response.get("command", command),
                    response.get("error", "unknown worker error"),
                    response.get("traceback", ""),
                )
            )
        return response

    def close_env(self):
        if self._closed:
            return
        self._closed = True
        for env_idx, process in enumerate(self.ps):
            if process.is_alive():
                try:
                    self.parent_conns[env_idx].send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self.ps:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        for env_idx in range(self.batch_size):
            self._send_worker(env_idx, "reset", None)
            if (
                self._initial_reset
                and self.worker_startup_stagger > 0
                and env_idx + 1 < self.batch_size
            ):
                time.sleep(self.worker_startup_stagger)

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        for env_idx in range(self.batch_size):
            data = self._recv_worker(env_idx, "reset")
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            self.latest_snapshots[env_idx] = data.get("snapshot")

        self._initial_reset = False

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset(test_mode=test_mode)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        trace_request = self.battle_trace_request if test_mode else None
        self.battle_trace_request = None
        trace_env_idx = 0
        trace_frames = []

        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []
        
        save_probs = getattr(self.args, "save_probs", False)
        while True:
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                
            cpu_actions = actions.to("cpu").numpy()
            action_by_env = {
                env_idx: cpu_actions[action_idx]
                for action_idx, env_idx in enumerate(envs_not_terminated)
            }

            if trace_request is not None and trace_env_idx in action_by_env:
                trace_frames.append(
                    {
                        "t": int(self.t),
                        "t_env": int(trace_request["t_env"]),
                        "snapshot": self.latest_snapshots[trace_env_idx],
                        "actions": action_by_env[trace_env_idx].astype(int).tolist(),
                        "diagnostics": model_diagnostics(self.mac, batch_index=trace_env_idx),
                    }
                )

            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        cmd = "step_trace" if trace_request is not None and idx == trace_env_idx else "step"
                        self._send_worker(idx, cmd, cpu_actions[action_idx])
                    action_idx += 1

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            post_transition_data = {
                "reward": [],
                "terminated": []
            }

            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            for idx in range(self.batch_size):
                if not terminated[idx]:
                    data = self._recv_worker(idx, "step")
                    post_transition_data["reward"].append((data["reward"],))
                    if "snapshot" in data:
                        self.latest_snapshots[idx] = data["snapshot"]

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            self.t += 1

            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run
        elif trace_request is not None:
            trace_frames.append(
                {
                    "t": int(self.t),
                    "t_env": int(trace_request["t_env"]),
                    "snapshot": self.latest_snapshots[trace_env_idx],
                    "actions": None,
                    "diagnostics": None,
                }
            )
            self.last_battle_trace = {
                "prefix": trace_request["prefix"],
                "t_env": int(trace_request["t_env"]),
                "map_name": getattr(self.args, "env_args", {}).get("map_name", None),
                "frames": trace_frames,
            }

        for env_idx in range(self.batch_size):
            self._send_worker(env_idx, "get_stats", None)

        env_stats = []
        for env_idx in range(self.batch_size):
            env_stat = self._recv_worker(env_idx, "get_stats")
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn, reset_retries=0, reset_retry_delay=1.0):
    env = None

    def close_current_env():
        nonlocal env
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            env = None

    def create_env():
        nonlocal env
        env = env_fn.x()

    def reset_with_retry():
        last_error = None
        for attempt in range(reset_retries + 1):
            try:
                if env is None:
                    create_env()
                env.reset()
                return
            except Exception as err:
                last_error = err
                close_current_env()
                if attempt < reset_retries:
                    time.sleep(reset_retry_delay * (attempt + 1))
        raise RuntimeError(
            "Environment reset failed after {} attempt(s).".format(
                reset_retries + 1
            )
        ) from last_error

    def safe_battle_snapshot():
        snapshot_fn = getattr(env, "get_battle_snapshot", None)
        if snapshot_fn is None:
            return None
        try:
            return snapshot_fn()
        except Exception:
            return None

    try:
        create_env()
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break

            try:
                if cmd in {"step", "step_trace"}:
                    actions = data
                    reward, terminated, env_info = env.step(actions)
                    state = env.get_state()
                    avail_actions = env.get_avail_actions()
                    obs = env.get_obs()
                    response = {
                        "state": state,
                        "avail_actions": avail_actions,
                        "obs": obs,
                        "reward": reward,
                        "terminated": terminated,
                        "info": env_info
                    }
                    if cmd == "step_trace":
                        snapshot = safe_battle_snapshot()
                        if snapshot is not None:
                            response["snapshot"] = snapshot
                    remote.send(response)
                elif cmd == "reset":
                    reset_with_retry()
                    response = {
                        "state": env.get_state(),
                        "avail_actions": env.get_avail_actions(),
                        "obs": env.get_obs(),
                    }
                    snapshot = safe_battle_snapshot()
                    if snapshot is not None:
                        response["snapshot"] = snapshot
                    remote.send(response)
                elif cmd == "close":
                    break
                elif cmd == "get_env_info":
                    remote.send(env.get_env_info())
                elif cmd == "get_stats":
                    remote.send(env.get_stats())
                else:
                    raise NotImplementedError
            except Exception as err:
                try:
                    remote.send(
                        {
                            "__worker_error__": True,
                            "command": cmd,
                            "error": repr(err),
                            "traceback": traceback.format_exc(),
                        }
                    )
                except Exception:
                    pass
                break
    except Exception as err:
        try:
            remote.send(
                {
                    "__worker_error__": True,
                    "command": "initialise",
                    "error": repr(err),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        close_current_env()
        remote.close()


class CloudpickleWrapper():
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
