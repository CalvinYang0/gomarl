"""Same deterministic soft one-fill TD, after collection-time state refresh.

Only the raw mixer state is held. Sigmoid field gates are recomputed in every
online/target forward and learned by the original joint soft one-fill TD.
"""
from .mixer_state_refresh_learner import MixerStateRefreshAuditMixin
from .soft_onefill_gate_learner import SoftOneFillGateLearner


class SoftOneFillRefreshLearner(MixerStateRefreshAuditMixin, SoftOneFillGateLearner):
    refresh_architecture = 'historical_qmix_minimal_soft_onefill'

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        # Gate snapshots and checkpoints must identify the combined experiment,
        # not silently present it as ordinary timestep-wise soft one-fill.
        self.contract.update(architecture='single_qmix_soft_onefill_with_state_refresh',
            mixer_state_refresh=self.refresh_contract,
            state_selection='collected held raw state BEFORE sigmoid interpolation')
