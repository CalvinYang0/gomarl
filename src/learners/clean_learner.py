import copy

import torch as th
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets


class CleanLearner:
    def __init__(self, mac, scheme, logger, args):
        del scheme
        self.args = args
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.logger = logger
        self.params = list(self.mac.parameters())

        mixer_name = getattr(args, "mixer", "qmix")
        if mixer_name == "qmix":
            self.mixer = QMixer(args)
            self.target_mixer = copy.deepcopy(self.mixer)
            self.params += list(self.mixer.parameters())
        elif mixer_name == "vdn":
            self.mixer = VDNMixer()
            self.target_mixer = copy.deepcopy(self.mixer)
        elif mixer_name in {"none", None}:
            self.mixer = None
            self.target_mixer = None
        else:
            raise ValueError("Unsupported mixer={} for CleanLearner.".format(mixer_name))

        if getattr(args, "optimizer", "adam") == "adam":
            self.optimiser = Adam(self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(
                params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
            )

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_out.append(self.mac.forward(batch, t=t))
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_mac_out.append(self.target_mac.forward(batch, t=t))
            target_mac_out = th.stack(target_mac_out, dim=1)

            mac_out_detach = mac_out.detach().clone()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_agent_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            if self.target_mixer is not None:
                target_max_qvals = self.target_mixer(target_max_agent_qvals, batch["state"])
            else:
                target_max_qvals = target_max_agent_qvals

            targets = build_td_lambda_targets(
                rewards,
                terminated,
                mask,
                target_max_qvals,
                self.args.n_agents,
                self.args.gamma,
                self.args.td_lambda,
            )

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_mask = mask.expand_as(td_error)
        masked_td_error = td_error * td_mask
        loss = (masked_td_error.pow(2).sum()) / td_mask.sum().clamp(min=1.0)

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = td_mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                masked_td_error.abs().sum().item() / max(mask_elems, 1.0),
                t_env,
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * td_mask).sum().item()
                / max(mask_elems, 1.0),
                t_env,
            )
            self.logger.log_stat(
                "target_mean", (targets * td_mask).sum().item() / max(mask_elems, 1.0),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
