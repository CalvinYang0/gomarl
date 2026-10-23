"""Unmodified historical joint TD, with audited collected sample-and-hold state."""
import json
from pathlib import Path

import torch as th

from .clean_learner import CleanLearner
from utils.mixer_state_refresh import validate, segments, source_times


class MixerStateRefreshLearner(CleanLearner):
    def __init__(self, mac, scheme, logger, args):
        validate(args)
        if not getattr(args, 'mixer_state_refresh_enabled', False):
            raise ValueError('The refresh learner requires mixer_state_refresh_enabled=True')
        required = {'mixer_state_raw', 'mixer_state_source_t', 'mixer_state_segments', 'mixer_state_collection_t_env'}
        if not required.issubset(scheme):
            raise ValueError('Missing collected mixer-state refresh metadata')
        super().__init__(mac, scheme, logger, args)
        self.refresh_limit = int(args.mixer_state_refresh_episode_limit)
        self.refresh_log_t = -int(args.learner_log_interval)-1
        self.refresh_contract = dict(version=1, architecture='historical_qmix_minimal',
            stage_steps=args.mixer_state_refresh_stage_steps, episode_limit=self.refresh_limit,
            schedule='K=min(L,1+floor(collection_t_env/stage_steps)); boundaries=floor(j*L/K)',
            condition='state is collected held state; mixer_state_raw is original environment state',
            target='same collected state and its recorded next snapshot; no current-stage reinterpretation',
            terminal='t=L holds last segment', stage_scope='fixed per episode/reset rollout batch')

    def _check_batch(self, batch):
        valid = batch['filled'].squeeze(-1).bool()
        sources = batch['mixer_state_source_t'].squeeze(-1)
        counts = batch['mixer_state_segments'].squeeze(-1)
        collected = batch['mixer_state_collection_t_env'].squeeze(-1)
        if batch.max_seq_length > self.refresh_limit+1:
            raise ValueError('Replay exceeds configured maximum episode length')
        expected = []
        for count, t_env in zip(counts.tolist(), collected.tolist()):
            if count != segments(t_env, self.args.mixer_state_refresh_stage_steps, self.refresh_limit):
                raise ValueError('Replay segment count differs from its collection-time stage')
            expected.append(source_times(self.refresh_limit, count, batch.max_seq_length))
        expected = sources.new_tensor(expected)
        if not th.equal(sources[valid], expected[valid]):
            raise ValueError('Replay source timestamps differ from maximum-length partition')
        raw = batch['mixer_state_raw']
        # The recorded source index is episode-relative, including sliced/padded
        # replay whose left edge must still be the episode reset.
        index = expected.unsqueeze(-1).expand(-1, -1, raw.shape[-1])
        gathered = raw.gather(1, index)
        if not th.equal(batch['state'][valid], gathered[valid]):
            raise ValueError('Mixer state is not the recorded held environment snapshot')
        return valid, sources, counts

    def train(self, batch, t_env, episode_num):
        valid, sources, counts = self._check_batch(batch)
        result = super().train(batch, t_env, episode_num)
        if t_env-self.refresh_log_t >= self.args.learner_log_interval:
            current = segments(t_env, self.args.mixer_state_refresh_stage_steps, self.refresh_limit)
            self.logger.log_stat('mixer_input/refresh_current_segments', current, t_env)
            self.logger.log_stat('mixer_input/refresh_replay_segments_mean', counts.float().mean().item(), t_env)
            self.logger.log_stat('mixer_input/refresh_replay_segments_min', counts.min().item(), t_env)
            self.logger.log_stat('mixer_input/refresh_replay_segments_max', counts.max().item(), t_env)
            if valid.any():
                times = th.arange(batch.max_seq_length, device=sources.device).expand_as(sources)
                self.logger.log_stat('mixer_input/refresh_state_age_mean', (times-sources)[valid].float().mean().item(), t_env)
            self.refresh_log_t = t_env
        return result

    def save_models(self, path):
        super().save_models(path)
        (Path(path)/'mixer_state_refresh_contract.json').write_text(json.dumps(self.refresh_contract, indent=2))

    def load_models(self, path):
        contract = Path(path)/'mixer_state_refresh_contract.json'
        if not contract.is_file() or json.loads(contract.read_text()) != self.refresh_contract:
            raise ValueError('Mixer-state refresh checkpoint contract differs or is missing; use a fresh run')
        super().load_models(path)
