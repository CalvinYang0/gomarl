"""One hierarchical TD loss or two independent TD objectives, shared GRU.

All behavior/test actions and all Double-Q action choices use head zero.
Both heads evaluate those SAME actions; there is no independently maximized
unimportant target or additive/averaged behavior head.
"""
import copy
from pathlib import Path

import torch as th
from torch.optim import Adam, RMSprop

from modules.mixers.qmix import QMixer
from modules.mixers.slot_group import SlotGroupMixer
from utils.rl_utils import build_td_lambda_targets
from utils.slot_input_control import log_slot_snapshot, score_slots


class SlotGroupLearner:
    def __init__(self, mac, scheme, logger, args):
        self.mac, self.logger, self.args = mac, logger, args
        if not getattr(args, "double_q", True):
            raise ValueError("Slot controls require important-head Double-Q")
        self.target_mac = copy.deepcopy(mac)
        self.target_mac.agent.target_gate_soft = True
        self.mixer = SlotGroupMixer(args) if mac.agent.is_grouped else QMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        if args.optimizer == "adam":
            self.optimiser = Adam(self.params, lr=args.lr)
        elif args.optimizer == "rmsprop":
            self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            raise ValueError("Unsupported slot-control optimizer")
        self.train_steps = self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1
        self.last_image_t = -int(getattr(args, "slot_heatmap_interval", 100000))
        self.interval = int(getattr(args, "slot_group_interval", 1000))
        self.budget = int(getattr(args, "slot_importance_samples", 256))
        if self.interval <= 0 or self.budget < args.n_agents:
            raise ValueError("Invalid slot grouping schedule/probe budget")
        self.last_losses = self.last_double_actions = None

    def _rollout(self, mac, batch):
        mac.init_hidden(batch.batch_size)
        values, group_values, group_obs = [], [], []
        for t in range(batch.max_seq_length):
            values.append(mac.forward(batch, t))
            if mac.agent.is_grouped:
                group_values.append(mac.latest_group_q)
                group_obs.append(mac.latest_group_observations)
        return (th.stack(values, 1), th.stack(group_values, 1) if group_values else None,
                th.stack(group_obs, 1) if group_obs else None)

    def _joint(self, mixer, agent_qs, states, observations, masks):
        if isinstance(mixer, SlotGroupMixer):
            local = mixer.group_values(agent_qs, observations, masks)
            return mixer.big(local, states) if mixer.hierarchical else local
        return mixer(agent_qs, states)

    def train(self, batch, t_env, episode_num):
        self.mac.agent.train()
        rewards, actions = batch["reward"][:, :-1], batch["actions"][:, :-1].long()
        terminated = batch["terminated"][:, :-1].float()
        # Both ends of the transition must be filled; all post-terminal
        # states are excluded, not just the immediately following state.
        mask = batch["filled"][:, :-1].float() * batch["filled"][:, 1:].float()
        if mask.shape[1] > 1:
            mask[:, 1:] *= (1 - terminated[:, :-1]).cumprod(1)
        important, groups, observations = self._rollout(self.mac, batch)
        masks = self.mac.agent.group_masks(important) if groups is not None else None
        if groups is not None:
            chosen = groups[:, :-1].gather(-1, actions.unsqueeze(2).expand(-1, -1, 2, -1, -1)).squeeze(-1)
        else:
            chosen = important[:, :-1].gather(-1, actions).squeeze(-1)
        joint = self._joint(self.mixer, chosen, batch["state"][:, :-1],
                            observations[:, :-1] if observations is not None else None, masks)
        with th.no_grad():
            target_important, target_groups, target_obs = self._rollout(self.target_mac, batch)
            # Select with ONLINE important head for every state (including
            # bootstrap t+1); evaluate using lagged target heads and mixers.
            selected = important.detach().masked_fill(batch["avail_actions"] == 0, -th.inf).argmax(-1, keepdim=True)
            self.last_double_actions = selected.detach().clone()
            if groups is not None:
                target_q = target_groups.gather(-1, selected.unsqueeze(2).expand(-1, -1, 2, -1, -1)).squeeze(-1)
                target_masks = self.target_mac.agent.group_masks(important)
            else:
                target_q, target_masks = target_important.gather(-1, selected).squeeze(-1), None
            target_joint = self._joint(self.target_mixer, target_q, batch["state"], target_obs, target_masks)
            targets = build_td_lambda_targets(rewards, terminated, mask, target_joint,
                                             self.args.n_agents, self.args.gamma, self.args.td_lambda)
        error = joint - targets
        expanded = mask.expand_as(error)
        # Dual TD uses two equal-weight losses, NOT a single TD of a summed
        # or averaged pair of Q_tot values. Each routes to its own generator.
        losses = (error.square() * expanded).sum((0, 1)) / expanded.sum((0, 1)).clamp(min=1)
        loss = losses.sum()
        if not th.isfinite(loss):
            raise RuntimeError("Non-finite slot-control TD loss")
        self.optimiser.zero_grad()
        loss.backward()
        norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        if not th.isfinite(norm):
            raise RuntimeError("Non-finite slot-control gradient norm")
        self.optimiser.step()
        self.train_steps += 1
        self.last_losses = losses.detach().clone()
        migration = None
        regroup = self.mac.agent.is_grouped and self.train_steps % self.interval == 0
        image_due = t_env - self.last_image_t >= int(getattr(self.args, "slot_heatmap_interval", 100000))
        if regroup or image_due:
            measured = score_slots(self.mac, batch, self.budget)
            if measured is not None:
                scores, flips, count = measured
                self.logger.log_stat("slot_input/probe_count", count, t_env)
                if regroup:
                    migration = self.mac.agent.apply_slot_scores(scores, flips)
                    # Masks are structural assignments, not fitted target
                    # parameters. Synchronize them immediately on a regroup.
                    self._sync_partition()
                elif not self.mac.agent.is_grouped:
                    self.mac.agent.slot_score_ema.copy_(scores)
                    self.mac.agent.slot_flip_rate.copy_(flips)
            else:
                self.logger.log_stat("slot_input/no_valid_probes", 1, t_env)
            log_slot_snapshot(self.mac.agent, self.args, self.logger, t_env, self.train_steps,
                              migration, images=image_due or regroup)
            if image_due or regroup:
                self.last_image_t = t_env
        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            if len(losses) == 2:
                self.logger.log_stat("loss_td_important", losses[0].item(), t_env)
                self.logger.log_stat("loss_td_unimportant", losses[1].item(), t_env)
            self.logger.log_stat("grad_norm", norm.item(), t_env)
            self.logger.log_stat("td_error_abs", (error.abs() * expanded).sum().item() / expanded.sum().clamp(min=1).item(), t_env)
            self.log_stats_t = t_env

    def _sync_partition(self):
        for name in ("important_slots", "slot_score_ema", "slot_flip_rate", "slot_score_initialized", "grouping_updates"):
            getattr(self.target_mac.agent, name).copy_(getattr(self.mac.agent, name))

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mac.agent.target_gate_soft = True
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), str(Path(path) / "mixer.th"))
        th.save(self.optimiser.state_dict(), str(Path(path) / "opt.th"))
        th.save(dict(train_steps=self.train_steps, last_target_update_episode=self.last_target_update_episode,
                     last_image_t=self.last_image_t), str(Path(path) / "slot_schedule.th"))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mixer.load_state_dict(th.load(str(Path(path) / "mixer.th"), map_location="cpu"))
        self.optimiser.load_state_dict(th.load(str(Path(path) / "opt.th"), map_location="cpu"))
        schedule = th.load(str(Path(path) / "slot_schedule.th"), map_location="cpu")
        for k, v in schedule.items():
            setattr(self, k, v)
        self._update_targets()
