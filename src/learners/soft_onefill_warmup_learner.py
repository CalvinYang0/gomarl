"""Direct projected gates and opt-in warmup followed by hard freeze."""
import math

import torch as th

from .soft_onefill_gate_learner import SoftOneFillGateLearner
from .mixer_state_refresh_learner import MixerStateRefreshAuditMixin
from modules.mixers.soft_onefill_warmup import (ProjectedOneFillGateMixer,
    ProjectedFreezeOneFillGateMixer, WarmupFreezeOneFillGateMixer)


class ProjectedOneFillGateLearner(SoftOneFillGateLearner):
    mixer_class = ProjectedOneFillGateMixer

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        architecture = 'single_qmix_projected_onefill'
        if self.contract['initial_probability'] == 1:
            architecture += '_initial1'  # Keep the historical checkpoint contract.
        self.contract.update(version=5, architecture=architecture,
            sampling='none_direct_coefficient', target='projected_coefficient',
            evaluation='projected_coefficient', gate_gradient='direct_autograd_then_project_[0,1]')

    def _after_main_optimizer_step(self):
        self.mixer.project()

    def load_models(self, path):
        super().load_models(path)
        for mixer in (self.mixer, self.target_mixer):
            p = mixer.coefficients()
            if not th.isfinite(p).all() or not th.all((p >= 0) & (p <= 1)):
                raise ValueError('Projected gate checkpoint is outside [0,1]')


class ProjectedOneFillRefreshLearner(MixerStateRefreshAuditMixin, ProjectedOneFillGateLearner):
    """Same projected gate/TD, after collection-time raw-state refresh."""
    refresh_architecture = 'historical_qmix_minimal_projected_onefill_initial1'

    def __init__(self, mac, scheme, logger, args):
        if float(args.mixer_input_initial_probability) != 1:
            self.refresh_architecture = 'historical_qmix_minimal_projected_onefill'
        super().__init__(mac, scheme, logger, args)
        self.contract.update(architecture=self.contract['architecture']+'_with_refresh',
            mixer_state_refresh=self.refresh_contract,
            state_selection='collected held raw state BEFORE direct gate interpolation')


class ProjectedFreezeOneFillRefreshLearner(ProjectedOneFillRefreshLearner):
    """Binarize once, at the first learner update after refresh becomes live."""
    mixer_class = ProjectedFreezeOneFillGateMixer

    def __init__(self, mac, scheme, logger, args):
        if getattr(args, 'mixer_state_refresh_schedule', None) != 'faster_stages':
            raise ValueError('Projected gate freeze requires the faster_stages schedule')
        super().__init__(mac, scheme, logger, args)
        self.freeze_steps = self.refresh_contract['full_steps']
        self.freeze_threshold = float(args.mixer_soft_gate_freeze_threshold)
        if not math.isfinite(self.freeze_threshold) or not 0 < self.freeze_threshold < 1:
            raise ValueError('Gate freeze threshold must be between 0 and 1')
        self.contract.update(version=7, architecture=self.contract['architecture']+'_binary_freeze',
            evaluation='projected_until_freeze_then_binary', target='same_online_mask_after_freeze',
            freeze_steps=self.freeze_steps, freeze_threshold=self.freeze_threshold,
            freeze_rule='p > threshold -> 1; equality and below -> 0',
            freeze_scope='online and target gates only; other parameters keep training')

    def _gate_metrics(self, values):
        metrics = super()._gate_metrics(values)
        metrics['mixer_input/gate_frozen'] = int(self.mixer.freeze_t_env.item() >= 0)
        return metrics

    def _before_training_update(self, t_env):
        if t_env < self.freeze_steps or self.mixer.freeze_t_env.item() >= 0:
            return
        p = self.mixer.coefficients().detach()
        if not th.isfinite(p).all():
            raise RuntimeError('Cannot freeze non-finite projected gate coefficients')
        mask = (p > self.freeze_threshold).to(p)
        self.mixer.freeze(mask, t_env)
        self.target_mixer.freeze(mask, t_env)
        self.optimiser.state.pop(self.mixer.gate_parameter, None)
        self.last_image_t = min(self.last_image_t, t_env-self.image_interval)
        for key, value in dict(frozen=1, open_slots=mask.sum().item(), t_env=t_env).items():
            self.logger.log_stat('mixer_input/gate_freeze_'+key, value, t_env)

    def load_models(self, path):
        super().load_models(path)
        a, b = self.mixer, self.target_mixer
        step = a.freeze_t_env.item()
        if step != b.freeze_t_env.item() or step < -1:
            raise ValueError('Invalid online/target projected freeze timestamp')
        if step >= 0:
            p = a.coefficients()
            if (step < self.freeze_steps or not th.equal(p, b.coefficients()) or
                    not th.all((p == 0) | (p == 1))):
                raise ValueError('Invalid frozen projected mask checkpoint')
            self.optimiser.state.pop(a.gate_parameter, None)
        a.restore_trainability()
        b.restore_trainability()


class WarmupFreezeOneFillLearner(SoftOneFillGateLearner):
    mixer_class = WarmupFreezeOneFillGateMixer

    def __init__(self, mac, scheme, logger, args):
        self.freeze_steps = getattr(args, 'mixer_soft_gate_freeze_steps', 5000000)
        self.freeze_threshold = float(getattr(args, 'mixer_soft_gate_freeze_threshold', 0.9))
        if type(self.freeze_steps) is not int or self.freeze_steps <= 0:
            raise ValueError('Gate warmup steps must be a positive integer')
        if not math.isfinite(self.freeze_threshold) or not 0 < self.freeze_threshold < 1:
            raise ValueError('Gate freeze threshold must be between 0 and 1')
        super().__init__(mac, scheme, logger, args)
        self.contract.update(version=6, architecture='single_qmix_sigmoid_warmup_binary_freeze',
            evaluation='sigmoid_until_freeze_then_binary', target='same_online_mask_after_freeze',
            freeze_steps=self.freeze_steps, freeze_threshold=self.freeze_threshold,
            freeze_rule='p > threshold -> 1; equality and below -> 0',
            freeze_scope='online and target gates only; other parameters keep training')

    def _gate_metrics(self, values):
        metrics = super()._gate_metrics(values)
        metrics['mixer_input/gate_frozen'] = int(self.mixer.gate_frozen.item())
        return metrics

    def _before_training_update(self, t_env):
        if t_env < self.freeze_steps or self.mixer.gate_frozen.item():
            return
        p = self.mixer.coefficients().detach()
        if not th.isfinite(p).all():
            raise RuntimeError('Cannot freeze non-finite gate coefficients')
        mask = (p > self.freeze_threshold).to(self.mixer.frozen_mask)
        # Use ONE mask from the online warmup gate for both branches. Do not
        # synchronize target actor/mixer weights early or threshold lagging gates.
        self.mixer.freeze(mask, t_env)
        self.target_mixer.freeze(mask, t_env)
        self.optimiser.state.pop(self.mixer.gate_logits, None)
        self.last_image_t = min(self.last_image_t, t_env-self.image_interval)
        for key, value in dict(frozen=1, open_slots=mask.sum().item(), t_env=t_env).items():
            self.logger.log_stat('mixer_input/gate_freeze_'+key, value, t_env)

    def load_models(self, path):
        super().load_models(path)
        a, b = self.mixer, self.target_mixer
        if a.gate_frozen.item() != b.gate_frozen.item():
            raise ValueError('Online/target frozen-gate state differs')
        if a.gate_frozen.item():
            if (not th.equal(a.frozen_mask, b.frozen_mask) or
                    a.freeze_t_env.item() != b.freeze_t_env.item() or
                    a.freeze_t_env.item() < self.freeze_steps or
                    not th.all((a.frozen_mask == 0) | (a.frozen_mask == 1))):
                raise ValueError('Invalid frozen mask/timestamp checkpoint')
            self.optimiser.state.pop(a.gate_logits, None)
        a.restore_trainability()
        b.restore_trainability()


class WarmupFreezeOneFillRefreshLearner(MixerStateRefreshAuditMixin, WarmupFreezeOneFillLearner):
    refresh_architecture = 'historical_qmix_minimal_soft_onefill_warmup_freeze'

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.contract.update(mixer_state_refresh=self.refresh_contract,
            state_selection='collected held raw state BEFORE gate interpolation')
