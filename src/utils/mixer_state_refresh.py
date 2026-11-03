"""Opt-in episode-limit-partitioned mixer-state sample-and-hold curriculum.

EpisodeBatch['state'] is the ACTUAL mixer condition. For this experiment only,
the unmodified environment state is additionally retained as mixer_state_raw.
Actor observations/actions/rewards are never altered. No RNG or model weights.
"""
import math
import numpy as np
import torch as th


def enabled(args):
    return bool(getattr(args, 'mixer_state_refresh_enabled', False))


def validate(args):
    if not enabled(args):
        return
    if (getattr(args, 'clean_model_type', None) != 'qmix_minimal'
            or getattr(args, 'mixer', None) != 'qmix'
            or getattr(args, 'learner', None) not in {
                'mixer_state_refresh_learner', 'soft_onefill_refresh_learner',
                'soft_onefill_freeze_refresh_learner', 'projected_onefill_refresh_learner'}):
        raise ValueError('Mixer-state refresh requires historical qmix_minimal + qmix + an audited refresh learner')
    interval = getattr(args, 'mixer_state_refresh_stage_steps', None)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError('mixer_state_refresh_stage_steps must be a positive integer environment-step count')
    mode = getattr(args, 'mixer_state_refresh_schedule', 'linear')
    if mode not in {'linear', 'faster_stages', 'larger_jumps'}:
        raise ValueError('Unknown mixer-state refresh schedule')
    if type(getattr(args, 'mixer_state_refresh_eval_live', False)) is not bool:
        raise ValueError('mixer_state_refresh_eval_live must be boolean')
    if mode != 'linear':
        full = getattr(args, 'mixer_state_refresh_full_steps', 8000000)
        if type(full) is not int or full <= 0:
            raise ValueError('mixer_state_refresh_full_steps must be a positive integer')
        if mode == 'larger_jumps' and full % interval:
            raise ValueError('larger_jumps requires full_steps divisible by stage_steps')


def scheduled_segments(args, t_env, episode_limit, test_mode=False):
    """Legacy defaults are unchanged; new schedules have an explicit live endpoint.

    faster_stages: K=1+floor((L-1)*(t/H)^2), i.e. transition j occurs
    at ceil(H*sqrt(j/(L-1))). Each increment is +1, waits become shorter.
    larger_jumps: fixed stage duration, K=ceil(L**(stage/num_stages)).
    """
    if t_env < 0 or episode_limit <= 0:
        raise ValueError('Nonnegative t_env and positive episode limit required')
    if test_mode and getattr(args, 'mixer_state_refresh_eval_live', False):
        return int(episode_limit)
    mode = getattr(args, 'mixer_state_refresh_schedule', 'linear')
    if mode == 'linear':
        return segments(t_env, args.mixer_state_refresh_stage_steps, episode_limit)
    full = getattr(args, 'mixer_state_refresh_full_steps', 8000000)
    if t_env >= full:
        return int(episode_limit)
    if mode == 'faster_stages':
        # Integer arithmetic prevents floating rounding at curriculum boundaries.
        return 1 + (int(episode_limit)-1)*int(t_env)**2 // full**2
    if mode == 'larger_jumps':
        stage = int(t_env) // args.mixer_state_refresh_stage_steps
        total = full // args.mixer_state_refresh_stage_steps
        return min(int(episode_limit), int(math.ceil(episode_limit**(stage/total))))
    raise ValueError('Unknown mixer-state refresh schedule')


def segments(t_env, stage_steps, episode_limit):
    if t_env < 0 or stage_steps <= 0 or episode_limit <= 0:
        raise ValueError('Nonnegative t_env and positive stage/episode limits required')
    return min(int(episode_limit), 1 + int(t_env) // int(stage_steps))


def boundaries(episode_limit, count):
    if not 1 <= count <= episode_limit:
        raise ValueError('Segment count must lie in [1, episode_limit]')
    return tuple(i * episode_limit // count for i in range(count))


def source_times(episode_limit, count, length, live_terminal=False):
    """Legacy terminal holds; opt-in live stages also refresh t=L."""
    if live_terminal:
        if count != episode_limit:
            raise ValueError('Live terminal requires timestep-wise refresh')
        return tuple(range(length))
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
    def __init__(self, args, episode_limit, batch_size, t_env, test_mode=False):
        validate(args)
        self.episode_limit, self.batch_size = int(episode_limit), int(batch_size)
        self.collection_t_env = int(t_env)
        self.count = scheduled_segments(args, t_env, episode_limit, test_mode)
        self.live_terminal = bool((test_mode and getattr(args, 'mixer_state_refresh_eval_live', False)) or
            (getattr(args, 'mixer_state_refresh_schedule', 'linear') != 'linear' and self.count == self.episode_limit))
        self.starts = set(boundaries(self.episode_limit, self.count))
        if self.live_terminal:
            self.starts.add(self.episode_limit)
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


def begin(args, episode_limit, batch_size, t_env, test_mode=False):
    validate(args)
    return MixerStateCache(args, episode_limit, batch_size, t_env, test_mode) if enabled(args) else None
