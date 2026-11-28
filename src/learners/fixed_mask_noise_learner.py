"""Fixed learned-mask retraining and matched noisy state-only mixer controls."""
from copy import copy
from hashlib import sha256
from pathlib import Path
import torch as th

from .soft_onefill_gate_learner import SoftOneFillGateLearner
from .soft_onefill_warmup_learner import (ProjectedOneFillRefreshLearner,
    ProjectedFreezeOneFillRefreshLearner)
from modules.mixers.fixed_onefill_mask import FixedOneFillMaskMixer


class UniqueMaskDiagnostics:
    def _snapshot(self, t_env, states=None, valid=None):
        # The historical token contains only group and wall-clock seconds.
        # Add exact run identity so simultaneous seeds cannot overwrite JSONs.
        original = self.args
        self.args = copy(original)
        self.args.unique_token = original.unique_token+'__'+sha256(
            original.wandb_run_name.encode()).hexdigest()[:12]
        try:
            return super()._snapshot(t_env, states, valid)
        finally:
            self.args = original

    def _gate_metrics(self, values):
        result = super()._gate_metrics(values)
        if self.args.env_args.get('mixer_state_noise', False):
            half = values.numel()//2
            result.update({'mixer_input/real_gate_mean': values[:half].mean().item(),
                'mixer_input/noise_gate_mean': values[half:].mean().item(),
                'mixer_input/noise_gate_below_01_fraction': (values[half:] < .1).float().mean().item()})
        return result


class FixedOneFillMaskLearner(UniqueMaskDiagnostics, SoftOneFillGateLearner):
    mixer_class = FixedOneFillMaskMixer

    def __init__(self, mac, scheme, logger, args):
        if getattr(args, 'mixer_state_refresh_enabled', False):
            raise ValueError('Fixed masks and noisy QMIX control use live state from step zero')
        super().__init__(mac, scheme, logger, args)
        self.contract.update(version=8, architecture='qmix_fixed_onefill_mask_from_initialization',
            sampling='none_fixed_binary', target='same_fixed_binary', evaluation='same_fixed_binary',
            gate_gradient='none; registered buffer', initial_probability=None,
            fixed_mask=self.mixer.coefficients().tolist(), fixed_mask_source=self.mixer.source,
            noise_state=bool(args.env_args.get('mixer_state_noise', False)))

    def load_models(self, path):
        saved = th.load(str(Path(path)/'onefill_training.th'), map_location='cpu', weights_only=False)
        expected = self.mixer.fixed_mask.detach().cpu()
        for key in ('mixer', 'target_mixer'):
            if not th.equal(saved[key]['fixed_mask'].cpu(), expected):
                raise ValueError('Checkpoint fixed mask differs from the requested heatmap')
        super().load_models(path)


class NoiseProjectedRefreshLearner(UniqueMaskDiagnostics, ProjectedOneFillRefreshLearner):
    def __init__(self, mac, scheme, logger, args):
        if not args.env_args.get('mixer_state_noise', False):
            raise ValueError('Noisy gate learner requires D real + D collected noise fields')
        super().__init__(mac, scheme, logger, args)
        self.contract.update(noise_distribution='uniform_0_1', noise_ratio=1,
            noise_scope='global mixer state only; private per-worker train/eval RNG; stored in replay',
            noise_refresh='full concatenated state uses the historical 8M curriculum')


class NoiseProjectedFreezeRefreshLearner(UniqueMaskDiagnostics, ProjectedFreezeOneFillRefreshLearner):
    """Same collected-noise/TD recipe, with one explicit binary gate freeze."""
    def __init__(self, mac, scheme, logger, args):
        if not args.env_args.get('mixer_state_noise', False):
            raise ValueError('Noise freeze learner requires the collected D+D state')
        super().__init__(mac, scheme, logger, args)
        # Keep freeze time explicit, rather than implicitly coupling it to the
        # refresh endpoint. The submitter pins both for the requested experiment.
        steps = args.mixer_soft_gate_freeze_steps
        if type(steps) is not int or steps <= 0:
            raise ValueError('Noise gate freeze step must be a positive integer')
        self.freeze_steps = steps
        self.contract.update(freeze_steps=steps, noise_distribution='uniform_0_1', noise_ratio=1,
            noise_scope='global mixer state only; private per-worker train/eval RNG; stored in replay',
            noise_refresh='full concatenated state uses the pinned curriculum',
            noise_refresh_full_steps=self.refresh_contract['full_steps'])

    def _gate_metrics(self, values):
        metrics = super()._gate_metrics(values)
        half = values.numel()//2
        metrics.update({'mixer_input/real_exact_closed_slots': int((values[:half] == 0).sum()),
                        'mixer_input/noise_exact_closed_slots': int((values[half:] == 0).sum())})
        return metrics

    def _before_training_update(self, t_env):
        was_frozen = self.mixer.freeze_t_env.item() >= 0
        super()._before_training_update(t_env)
        if not was_frozen and self.mixer.freeze_t_env.item() >= 0 and not self.mixer.coefficients().any():
            self.logger.console_logger.warning(
                'All real/noise gates closed at freeze: effective mixer input is all ones; '
                'training continues without overriding the requested threshold.')
