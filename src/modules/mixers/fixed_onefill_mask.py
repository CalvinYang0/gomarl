"""Parameter-free binary input mask; base QMIX weights are still trainable."""
from copy import copy

import torch as th

from .soft_onefill_gate import SoftOneFillGateMixer
from utils.fixed_heatmap_mask import load_heatmap_mask


class FixedOneFillMaskMixer(SoftOneFillGateMixer):
    def __init__(self, args):
        local = copy(args)
        local.mixer_input_initial_probability = .5
        super().__init__(local)
        del self.gate_logits
        self.base.args = args
        if args.mixer_fixed_mask_kind == 'heatmap':
            if args.env_args.get('mixer_state_noise', False):
                raise ValueError('Historical heatmap mask must use the original D-dimensional state')
            mask, self.source = load_heatmap_mask(args.mixer_fixed_mask_bundle,
                args.env_args['map_name'], self.state_dim)
            if self.source['bundle_sha256'] != args.mixer_fixed_mask_sha256:
                raise ValueError('Fixed mask bundle changed since submission')
        elif args.mixer_fixed_mask_kind == 'all_ones':
            if not args.env_args.get('mixer_state_noise', False):
                raise ValueError('This all-ones gate is reserved for the matched noisy QMIX control')
            mask, self.source = [1] * self.state_dim, dict(kind='ordinary_qmix_identity_input')
        else:
            raise ValueError('Unknown fixed-mask kind')
        self.register_buffer('fixed_mask', th.tensor(mask, dtype=th.float32))

    def coefficients(self):
        return self.fixed_mask

    def freeze_metadata(self):
        return dict(frozen=True, t_env=0, open_slots=int(self.fixed_mask.sum()),
                    scope='fixed from initialization; base QMIX remains trainable', source=self.source)
