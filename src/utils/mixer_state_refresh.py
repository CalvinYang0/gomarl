"""Opt-in episode-limit-partitioned mixer-state sample-and-hold curriculum.

EpisodeBatch['state'] is the ACTUAL mixer condition. For this experiment only,
the unmodified environment state is additionally retained as mixer_state_raw.
Actor observations/actions/rewards are never altered. No RNG or model weights.
"""
import numpy as np
import torch as th


def enabled(args):
    return bool(getattr(args, 'mixer_state_refresh_enabled', False))


def validate(args):
    if not enabled(args):
        return
    if (getattr(args, 'clean_model_type', None) != 'qmix_minimal'
            or getattr(args, 'mixer', None) != 'qmix'
            or getattr(args, 'learner', None) != 'mixer_state_refresh_learner'):
        raise ValueError('Mixer-state refresh requires historical qmix_minimal + qmix + mixer_state_refresh_learner')
    interval = getattr(args, 'mixer_state_refresh_stage_steps', None)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError('mixer_state_refresh_stage_steps must be a positive integer environment-step count')


def segments(t_env, stage_steps, episode_limit):
    if t_env < 0 or stage_steps <= 0 or episode_limit <= 0:
        raise ValueError('Nonnegative t_env and positive stage/episode limits required')
    return min(int(episode_limit), 1 + int(t_env) // int(stage_steps))


def boundaries(episode_limit, count):
    if not 1 <= count <= episode_limit:
        raise ValueError('Segment count must lie in [1, episode_limit]')
    return tuple(i * episode_limit // count for i in range(count))


def source_times(episode_limit, count, length):
    """Terminal observation at t=L holds the last segment; no extra refresh."""
    starts = boundaries(episode_limit, count)
    return tuple(starts[min(count-1, np.searchsorted(starts, t, side='right')-1)]
                 for t in range(length))


def extra_scheme(args, state_shape, episode_limit):
    validate(args)
    if not enabled(args):
        return {}
    args.mixer_state_refresh_episode_limit = int(episode_limit)
    return {
        'mixer_state_raw': dict(vshape=state_shape),
        'mixer_state_source_t': dict(vshape=(1,), dtype=th.long),
        'mixer_state_segments': dict(vshape=(1,), dtype=th.long, episode_const=True),
        'mixer_state_collection_t_env': dict(vshape=(1,), dtype=th.long, episode_const=True),
    }


class MixerStateCache:
    def __init__(self, args, episode_limit, batch_size, t_env):
        validate(args)
        self.episode_limit, self.batch_size = int(episode_limit), int(batch_size)
        self.collection_t_env = int(t_env)
        self.count = segments(t_env, args.mixer_state_refresh_stage_steps, episode_limit)
        self.starts = set(boundaries(self.episode_limit, self.count))
        self.values = [None] * self.batch_size
        self.sources = [None] * self.batch_size
        self.last_t = [-1] * self.batch_size

    def apply(self, data, t, env_indices=None):
        indices = list(range(self.batch_size)) if env_indices is None else list(env_indices)
        if not 0 <= t <= self.episode_limit or len(indices) != len(data['state']):
            raise ValueError('Invalid mixer cache timestep/worker batch')
        if len(set(indices)) != len(indices):
            raise ValueError('Duplicate mixer cache worker index')
        raw, held, sources = [], [], []
        for idx, value in zip(indices, data['state']):
            if not 0 <= idx < self.batch_size or t != self.last_t[idx] + 1:
                raise ValueError('Mixer cache requires consecutive observations for each worker, starting at reset')
            current = np.asarray(value).copy()
            if not np.isfinite(current).all():
                raise ValueError('Nonfinite environment state')
            if t in self.starts:
                self.values[idx], self.sources[idx] = current.copy(), int(t)
            if self.values[idx] is None:
                raise RuntimeError('Mixer cache used before initial observation')
            raw.append(current)
            held.append(self.values[idx].copy())
            sources.append((self.sources[idx],))
            self.last_t[idx] = int(t)
        result = dict(data, state=held, mixer_state_raw=raw, mixer_state_source_t=sources)
        if t == 0:
            result.update(mixer_state_segments=[(self.count,)] * len(indices),
                          mixer_state_collection_t_env=[(self.collection_t_env,)] * len(indices))
        return result


def begin(args, episode_limit, batch_size, t_env):
    validate(args)
    return MixerStateCache(args, episode_limit, batch_size, t_env) if enabled(args) else None
