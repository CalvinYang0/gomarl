"""Historical one-fill TD update, opt-in complete-QMIX output composition."""
import json
from pathlib import Path

import torch as th

from .onefill_gate_learner import OneFillGateLearner
from modules.mixers.qtot_composition import QTotCompositionMixer


class QTotCompositionLearner(OneFillGateLearner):
    mixer_class = QTotCompositionMixer

    def __init__(self, mac, scheme, logger, args):
        if args.mixer_onefill_objective != "td":
            raise ValueError("Q_tot composition is TD-only, without KL/MC auxiliaries")
        super().__init__(mac, scheme, logger, args)
        self.contract.update(version=3, architecture="two_complete_qmix_outputs",
            output_combination=self.mixer.output_combination,
            secondary_initialization="independent_nonzero_default",
            alpha_initial=float(args.mixer_qtot_alpha_initial))

    def train(self, batch, t_env, episode_num):
        previous = self.log_stats_t
        super().train(batch, t_env, episode_num)
        if self.log_stats_t != previous:
            terminated = batch["terminated"][:, :-1].float()
            valid = batch["filled"][:, :-1].float() * batch["filled"][:, 1:].float()
            valid[:, 1:] *= (1 - terminated[:, :-1]).cumprod(1)
            for key, value in self.mixer.latest_output_components.items():
                samples = value[valid.bool()]
                if samples.numel():
                    self.logger.log_stat("mixer_branches/qtot_" + key + "_mean", samples.mean().item(), t_env)

    def _snapshot(self, t_env, states=None, valid=None):
        super()._snapshot(t_env, states, valid)
        parts = self.mixer.latest_output_components
        if parts is None:
            return
        import matplotlib.pyplot as plt
        record = dict(t_env=int(t_env), contract=self.contract,
            input_mode="actual_last_stochastic_training_forward",
            note="Complete Q_tot values, not summed weights or extra TD targets",
            alpha_main=None if self.mixer.alpha_logit is None else self.mixer.alpha().detach().item(),
            branches={})
        labels, means, errors = [], [], []
        for key, value in parts.items():
            values = value[valid.bool()] if valid is not None else value.reshape(-1)
            if not values.numel():
                continue
            stats = dict(mean=values.mean().item(), std=values.std(unbiased=False).item(), count=values.numel())
            record["branches"][key] = stats
            labels.append(key)
            means.append(stats["mean"])
            errors.append(stats["std"])
        directory = Path(self.args.local_results_path) / "mixer_onefill_gate" / self.args.unique_token
        stem = "step_{:09d}_update_{:07d}_qtot".format(int(t_env), self.train_steps)
        path = directory / (stem + ".json")
        path.write_text(json.dumps(record, indent=2, allow_nan=False))
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        image_path = directory / (stem + ".png")
        try:
            ax.bar(labels, means, yerr=errors)
            ax.axhline(0, color="black", linewidth=.6)
            ax.set_ylabel("Q_tot mean ± transition std (NOT across seeds)")
            ax.tick_params(axis="x", rotation=15)
            title = "{} / {} / same agent Q / step={}".format(
                self.args.env_args.get("map_name"), self.mixer.combination, t_env)
            if self.mixer.alpha_logit is not None:
                title += " / main alpha={:.4f}".format(self.mixer.alpha().detach().item())
            if self.args.wandb_run_name.startswith("SYNTHETIC"):
                title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
            ax.set_title(title)
            fig.savefig(image_path, dpi=120)
        finally:
            plt.close(fig)
        if self.logger.use_wandb:
            self.logger._update_wandb_buffer("mixer_branches/qtot_components",
                self.logger.wandb_module.Image(str(image_path)), t_env)
            self.logger.wandb_module.save(str(path), base_path=str(directory))
