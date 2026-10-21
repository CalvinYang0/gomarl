"""Historical joint TD recipe with deterministic soft one-fill instead of ST."""
import json
from pathlib import Path

from .onefill_gate_learner import OneFillGateLearner
from modules.mixers.soft_onefill_gate import SoftOneFillGateMixer


class SoftOneFillGateLearner(OneFillGateLearner):
    mixer_class = SoftOneFillGateMixer

    def __init__(self, mac, scheme, logger, args):
        if args.mixer_onefill_objective != "td":
            raise ValueError("Soft one-fill uses joint TD only; no KL/MC or gate-only optimizer")
        super().__init__(mac, scheme, logger, args)
        self.contract.update(version=4, architecture="single_qmix_soft_onefill",
            sampling="none_deterministic_sigmoid", target="deterministic_sigmoid",
            evaluation="deterministic_sigmoid", gate_gradient="exact_autograd",
            gate_scope="static_per_state_scalar")
        # These settings are unused, not hidden hyperparameters of this model.
        for key in ("temperature", "residual_scale", "kl_prior", "kl_ratio", "kl_decay", "kl_eps", "kl_max", "mc_coef"):
            self.contract.pop(key)

    def _gate_metrics(self, values):
        return {"mixer_input/soft_gate_value_mean": values.mean().item(),
                "mixer_input/soft_gate_value_sum": values.sum().item()}

    def _snapshot(self, t_env, states=None, valid=None):
        import matplotlib.pyplot as plt
        values = self.mixer.gate_logits.sigmoid().detach().cpu()
        directory = Path(self.args.local_results_path) / "mixer_soft_onefill_gate" / self.args.unique_token
        directory.mkdir(parents=True, exist_ok=True)
        stem = "step_{:09d}_update_{:07d}".format(int(t_env), self.train_steps)
        path, image_path = directory / (stem + ".json"), directory / (stem + ".png")
        record = dict(t_env=int(t_env), train_updates=self.train_steps,
            run_name=self.args.wandb_run_name, state_slot_names=self.names,
            gate_value=values.tolist(), contract=self.contract,
            note="Static sigmoid interpolation coefficients; not sampled masks or field importance; x=p*state+(1-p)")
        path.write_text(json.dumps(record, indent=2, allow_nan=False))
        fig, ax = plt.subplots(figsize=(max(14, len(self.names) * .15), 3), constrained_layout=True)
        try:
            pixel = ax.imshow(values.unsqueeze(0).numpy(), aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax.set_yticks([0], labels=["soft gate value"])
            ax.set_xticks(range(len(self.names)), labels=self.names, rotation=90, fontsize=6)
            title = "{} / soft one-fill / seed={} / step={} / x=p*state+(1-p); no sampling".format(
                self.args.env_args.get("map_name"), self.args.seed, t_env)
            if self.args.wandb_run_name.startswith("SYNTHETIC"):
                title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
            ax.set_title(title)
            fig.colorbar(pixel, ax=ax)
            fig.savefig(image_path, dpi=130)
        finally:
            plt.close(fig)
        if self.logger.use_wandb:
            self.logger._update_wandb_buffer("mixer_input/soft_onefill_gate_heatmap",
                self.logger.wandb_module.Image(str(image_path)), t_env)
            self.logger.wandb_module.save(str(path), base_path=str(directory))
