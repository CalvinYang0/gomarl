from collections import defaultdict
from hashlib import sha256
import json
import logging
import numpy as np
import torch as th

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_wandb(self, config, team_name, project_name, mode):
        try:
            import wandb
        except ImportError as err:
            raise ImportError("W&B is enabled but package `wandb` is not installed.") from err

        if project_name in [None, ""]:
            raise ValueError("W&B logging requires `wandb_project`.")
        if mode not in ["offline", "online"]:
            raise ValueError(
                "Invalid `wandb_mode`: {}. Use 'offline' or 'online'.".format(mode)
            )

        self.use_wandb = True

        alg_name = config.get("name", "unknown_alg")
        env_name = config.get("env", "unknown_env")
        env_args = config.get("env_args", {})
        if "map_name" in env_args:
            env_name += "_" + str(env_args["map_name"])
        elif "env_name" in env_args:
            env_name += "_" + str(env_args["env_name"])
        elif "key" in env_args:
            env_name += "_" + str(env_args["key"])

        non_hash_keys = ["seed"]
        config_hash = sha256(
            json.dumps(
                {k: v for k, v in config.items() if k not in non_hash_keys},
                sort_keys=True,
                default=str,
            ).encode("utf8")
        ).hexdigest()[-10:]
        group_name = "_".join([alg_name, env_name, config_hash])

        entity = team_name if team_name not in [None, ""] else None
        self.wandb = wandb.init(
            entity=entity,
            project=project_name,
            config=config,
            group=group_name,
            mode=mode,
        )

        self.wandb_current_t = -1
        self.wandb_current_data = {}

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_wandb:
            wb_value = value.item() if hasattr(value, "item") else value
            if self.wandb_current_t != t and self.wandb_current_data:
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
                self.wandb_current_data = {}
            self.wandb_current_t = t
            self.wandb_current_data[key] = wb_value

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]
            self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(th.mean(th.tensor([float(x[1]) for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
        # Reset stats to avoid accumulating logs in memory
        self.stats = defaultdict(lambda: [])

    def finish(self):
        if self.use_wandb:
            if self.wandb_current_data:
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
            self.wandb.finish()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
