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
        # Keep local/TensorBoard/Sacred logging unchanged, but make W&B runs
        # compact by default: only the requested performance, loss, and gate
        # probability summaries are uploaded.
        self.wandb_minimal_logging = True

    @staticmethod
    def _wandb_metric_allowed(key):
        key = str(key)
        if key in {
            "game_win_mean",
            "test_game_win_mean",
            "battle_won_mean",
            "test_battle_won_mean",
            "ep_length_mean",
            "test_ep_length_mean",
            "dynamic_gate_probability_min",
            "dynamic_gate_probability_max",
            "dynamic_gate_linear_probability_mean",
            "dynamic_gate_attention_probability_mean",
        }:
            return True
        return (
            key.startswith("loss")
            or key.startswith("weighted_loss")
            or key.endswith("_to_td_ratio")
        )

    def _update_wandb_buffer(self, key, value, t):
        if not self.use_wandb:
            return
        if self.wandb_current_t != t and self.wandb_current_data:
            self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
            self.wandb_current_data = {}
        self.wandb_current_t = t
        self.wandb_current_data[key] = value

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
        self.wandb_module = wandb
        self.wandb_minimal_logging = bool(
            config.get("wandb_minimal_logging", True)
        )

        alg_name = config.get("name", "unknown_alg")
        run_name = config.get("wandb_run_name", None)
        if run_name in [None, ""]:
            run_name = alg_name
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
        if len(group_name) > 128:
            prefix_budget = 128 - len(config_hash) - 1
            group_name = "{}_{}".format(group_name[:prefix_budget], config_hash)

        entity = team_name if team_name not in [None, ""] else None
        self.wandb = wandb.init(
            entity=entity,
            project=project_name,
            config=config,
            name=run_name,
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
            if (
                not self.wandb_minimal_logging
                or self._wandb_metric_allowed(key)
            ):
                self._update_wandb_buffer(key, wb_value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]
            self._run_obj.log_scalar(key, value, t)

    def log_misc(self, key, value, t, to_sacred=True):
        if self.use_wandb and (
            not self.wandb_minimal_logging
            or self._wandb_metric_allowed(key)
        ):
            self._update_wandb_buffer(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def log_test_gate_probability_trajectory(self, trajectory, t):
        """Log one test trajectory as a compact W&B image.

        The y-axis is drop probability (one minus keep probability).  The
        dashed horizontal line is the deterministic test threshold converted
        to drop space; slots above it are dropped by the test gate.
        """
        if not self.use_wandb or trajectory is None:
            return
        # Keep this diagnostic independent from the gate-probability figure:
        # either plot should still be uploaded if the other one fails.
        self._log_test_generated_parameter_pca(trajectory, t)
        try:
            import matplotlib.pyplot as plt

            timesteps = trajectory["timesteps"]
            slot_names = trajectory["slot_names"]
            threshold = float(trajectory["threshold"])
            branches = trajectory["branches"]
            fig, axes = plt.subplots(
                1,
                len(branches),
                figsize=(max(8.0, 3.8 * len(branches)), 4.2),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            axes = axes[0]
            for axis, (branch_name, values) in zip(axes, branches.items()):
                values = np.asarray(values, dtype=float)
                for slot_index, slot_name in enumerate(slot_names):
                    axis.plot(
                        timesteps,
                        1.0 - values[:, slot_index],
                        linewidth=1.2,
                        label=str(slot_name),
                    )
                axis.axhline(
                    1.0 - threshold,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="drop threshold",
                )
                axis.set_title(str(branch_name).capitalize())
                axis.set_xlabel("timestep")
                axis.set_ylabel("drop probability")
                axis.set_ylim(0.0, 1.0)
                axis.grid(alpha=0.25)
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(4, max(1, len(labels))),
                fontsize=7,
            )
            fig.suptitle(
                "Test dynamic-gate trajectory "
                "(dashed: drop threshold={:.2f})".format(threshold),
                y=1.08,
            )
            fig.tight_layout()
            self._update_wandb_buffer(
                "test_dynamic_gate_trajectory",
                self.wandb_module.Image(fig),
                t,
            )
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - diagnostics must not stop training
            self.console_logger.warning(
                "Failed to create test dynamic-gate trajectory: %s", exc
            )

    def _log_test_generated_parameter_pca(self, trajectory, t):
        """Plot exact generated hypernetwork parameters from one test episode.

        Each timestep is one point. PCA is fitted only for visualization and
        is never part of the model or its training objective.
        """
        parameter_vectors = trajectory.get("generated_parameter_vectors")
        if parameter_vectors is None:
            return
        try:
            import matplotlib.pyplot as plt

            if th.is_tensor(parameter_vectors):
                parameters = parameter_vectors.detach().cpu().numpy()
            else:
                parameters = np.asarray(parameter_vectors, dtype=np.float32)
            if parameters.ndim != 2 or parameters.shape[0] < 2:
                return

            centered = parameters.astype(np.float64, copy=False)
            centered = centered - centered.mean(axis=0, keepdims=True)
            u, singular_values, _ = np.linalg.svd(
                centered, full_matrices=False
            )
            component_count = min(2, singular_values.size)
            coordinates = np.zeros((centered.shape[0], 2), dtype=np.float64)
            coordinates[:, :component_count] = (
                u[:, :component_count] * singular_values[:component_count]
            )
            variance = singular_values ** 2
            variance_total = float(variance.sum())
            explained = np.zeros(2, dtype=np.float64)
            if variance_total > 0.0:
                explained[:component_count] = (
                    variance[:component_count] / variance_total
                )

            timesteps = np.asarray(trajectory["timesteps"], dtype=int)
            fig, axis = plt.subplots(figsize=(6.8, 5.6))
            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="0.72",
                linewidth=1.0,
                zorder=1,
            )
            points = axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=timesteps,
                cmap="viridis",
                s=34,
                edgecolors="none",
                zorder=2,
            )
            axis.scatter(
                coordinates[0, 0], coordinates[0, 1],
                marker="o", s=80, facecolors="none", edgecolors="black",
                linewidths=1.3, label="start",
            )
            axis.scatter(
                coordinates[-1, 0], coordinates[-1, 1],
                marker="X", s=70, color="black", label="end",
            )
            axis.set_xlabel(
                "PC1 ({:.1f}% variance)".format(100.0 * explained[0])
            )
            axis.set_ylabel(
                "PC2 ({:.1f}% variance)".format(100.0 * explained[1])
            )
            axis.set_title(
                "Generated hypernetwork parameters across one test episode"
            )
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
            colorbar = fig.colorbar(points, ax=axis)
            colorbar.set_label("timestep")
            fig.tight_layout()
            self._update_wandb_buffer(
                "test_generated_parameter_pca_trajectory",
                self.wandb_module.Image(fig),
                t,
            )
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - diagnostics must not stop training
            self.console_logger.warning(
                "Failed to create test parameter PCA trajectory: %s", exc
            )

    def log_battle_trace_media(self, paths, t, fps=6):
        if not self.use_wandb:
            return

        if self.wandb_minimal_logging:
            video_path = paths.get("video")
            if video_path:
                video_format = "gif" if video_path.endswith(".gif") else "mp4"
                self._update_wandb_buffer(
                    "test_trajectory_video",
                    self.wandb_module.Video(
                        video_path, fps=fps, format=video_format
                    ),
                    t,
                )
            return

        if "battle_overview" in paths:
            self._update_wandb_buffer(
                "battle_trace/overview",
                self.wandb_module.Image(paths["battle_overview"]),
                t,
            )
        if "similarity" in paths:
            self._update_wandb_buffer(
                "battle_trace/relation_head_similarity",
                self.wandb_module.Image(paths["similarity"]),
                t,
            )
        if "alignment" in paths:
            self._update_wandb_buffer(
                "battle_trace/relation_head_alignment",
                self.wandb_module.Image(paths["alignment"]),
                t,
            )
        if "relation_dynamics_video" in paths:
            video_path = paths["relation_dynamics_video"]
            video_format = "gif" if video_path.endswith(".gif") else "mp4"
            self._update_wandb_buffer(
                "battle_trace/relation_head_dynamics",
                self.wandb_module.Video(video_path, fps=fps, format=video_format),
                t,
            )
        if "video" in paths:
            video_path = paths["video"]
            video_format = "gif" if video_path.endswith(".gif") else "mp4"
            self._update_wandb_buffer(
                "battle_trace/video",
                self.wandb_module.Video(video_path, fps=fps, format=video_format),
                t,
            )

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
