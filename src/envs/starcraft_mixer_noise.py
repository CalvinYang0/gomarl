"""Opt-in D real state + D independent U[0,1) nuisance fields; obs unchanged.

Noise is sampled once per observed simulator timestep and stored in the normal
replay state. Repeated get_state calls, online/target forwards and diagnostics
therefore reuse the same values. Private worker-specific train/eval RNGs never
consume NumPy's global exploration/environment RNG.
"""
import numpy as np

from .starcraft import StarCraft2Env


def configure_noise_env(args):
    if not getattr(args, 'mixer_state_noise', False):
        return
    if (args.env != 'sc2' or args.clean_model_type != 'qmix_minimal' or args.mixer != 'qmix' or
            args.learner not in {'fixed_onefill_mask_learner', 'noise_projected_refresh_learner'}):
        raise ValueError('State noise is opt-in to the audited ordinary-agent mixer experiments only')
    args.env_args = dict(args.env_args, mixer_state_noise=True,
                        mixer_state_noise_seed=args.mixer_state_noise_seed, mixer_state_noise_stream=0)


class MixerNoiseStarCraft2Env(StarCraft2Env):
    def __init__(self, noise_seed=271828, noise_stream=0, **kwargs):
        if type(noise_seed) is not int or noise_seed < 0 or type(noise_stream) is not int or noise_stream < 0:
            raise ValueError('Noise seed and worker stream must be nonnegative integers')
        super().__init__(**kwargs)
        if self.obs_instead_of_state:
            raise ValueError('State-noise experiment requires ordinary global state, not concatenated observations')
        self.raw_state_dim = int(super().get_state_size())
        seed = 0 if self._seed is None else int(self._seed)
        self._noise_rng = [np.random.default_rng(np.random.SeedSequence(
            [seed % 2**32, noise_seed, noise_stream, mode])) for mode in (0, 1)]
        self._noise_test = False
        self._noise_cache = None
        self._noise_cache_step = None

    def set_noise_test_mode(self, test_mode):
        if type(test_mode) is not bool:
            raise ValueError('Noise test mode must be boolean')
        self._noise_test = test_mode

    def reset(self):
        self._noise_cache = self._noise_cache_step = None
        return super().reset()

    def get_state_size(self):
        return 2 * int(super().get_state_size())

    def get_state(self):
        raw = np.asarray(super().get_state(), dtype=np.float32)
        if raw.shape != (self.raw_state_dim,):
            raise RuntimeError('Raw SMAC state dimension changed')
        key = (self._noise_test, int(self._episode_steps))
        if key != self._noise_cache_step:
            self._noise_cache = self._noise_rng[int(self._noise_test)].random(
                self.raw_state_dim, dtype=np.float32)
            self._noise_cache_step = key
        return np.concatenate((raw, self._noise_cache)).astype(np.float32, copy=False)

    def get_env_info(self):
        info = super().get_env_info()
        info.update(mixer_noise_raw_state_dim=self.raw_state_dim,
                    mixer_noise_dim=self.raw_state_dim, mixer_noise_distribution='uniform_0_1')
        return info
