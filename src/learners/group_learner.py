import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.group import Mixer as GroupMixer
from modules.mixers.group_vdn import Mixer as GroupVDNMixer
from utils.rl_utils import build_td_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import torch.nn as nn
import numpy as np
from utils.th_utils import get_parameters_num
from utils.graph_grouping import (
    adjacency_to_groups,
    local_subgraph_fusion_graph,
    local_subgraph_similarity_graph,
    pseudo_attention_graph,
    sparsify_graph,
)


class GROUPLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.params = list(self.mac.parameters())

        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')

        if args.mixer == "group":
            self.mixer = GroupMixer(args)
        elif args.mixer == "group_vdn":
            self.mixer = GroupVDNMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0

    def _zero(self, ref):
        return ref.new_tensor(0.0)

    def _compute_group_repulsion_loss(self, group_states, group_probs, mask):
        group_head_mode = getattr(self.args, "group_head_mode", "latent").replace("-", "_")
        if group_head_mode not in ["fixed_group", "graph_better_struct"]:
            return self._zero(group_states)

        norm_states = F.normalize(group_states, p=2, dim=-1)
        similarity = th.matmul(norm_states, norm_states.transpose(-1, -2))
        same_prob = th.matmul(group_probs, group_probs.transpose(-1, -2))

        off_diag = 1.0 - th.eye(self.args.n_agents, device=group_states.device).view(1, 1, self.args.n_agents, self.args.n_agents)
        valid = mask.unsqueeze(-1).expand_as(similarity) * off_diag
        if valid.sum() <= 0:
            return self._zero(group_states)

        loss_terms = -same_prob * similarity + (1.0 - same_prob) * similarity
        return (loss_terms * valid).sum() / valid.sum()

    def _compute_struct_group_regularizers(self, group_probs, group_graphs, mask):
        zero = self._zero(group_probs)
        group_head_mode = getattr(self.args, "group_head_mode", "latent").replace("-", "_")
        if group_head_mode != "graph_better_struct":
            return zero, zero, zero

        valid = mask.unsqueeze(-1).expand_as(group_probs[..., :1]).squeeze(-1)
        if valid.sum() <= 0:
            return zero, zero, zero

        mean_probs = (group_probs * valid.unsqueeze(-1)).sum(dim=(0, 1)) / valid.sum()
        balance_loss = -th.log(mean_probs.clamp(min=1e-8)).mean()

        entropy = -(group_probs.clamp(min=1e-8) * group_probs.clamp(min=1e-8).log()).sum(dim=-1)
        conf_loss = (entropy * valid).sum() / valid.sum()

        off_diag = 1.0 - th.eye(self.args.n_agents, device=group_graphs.device).view(1, 1, self.args.n_agents, self.args.n_agents)
        sparse_graph = group_graphs * off_diag
        row_sum = sparse_graph.sum(dim=-1, keepdim=True)
        valid_rows = valid * (row_sum.squeeze(-1) > 0).float()
        if valid_rows.sum() <= 0:
            sparse_loss = zero
        else:
            row_probs = sparse_graph / row_sum.clamp(min=1e-8)
            row_entropy = -(row_probs.clamp(min=1e-8) * row_probs.clamp(min=1e-8).log()).sum(dim=-1)
            sparse_loss = (row_entropy * valid_rows).sum() / valid_rows.sum()

        return balance_loss, conf_loss, sparse_loss

    def _apply_group_update(self, group_nxt):
        if group_nxt == self.mixer.group:
            return

        target_group_num = len(group_nxt)
        while len(self.mixer.hyper_b1) < target_group_num:
            self.mixer.add_new_net()
            self.target_mixer.add_new_net()
        while len(self.mixer.hyper_b1) > target_group_num:
            self.mixer.del_net(len(self.mixer.hyper_b1) - 1)
            self.target_mixer.del_net(len(self.target_mixer.hyper_b1) - 1)

        self.mixer.update_group(group_nxt)
        self.target_mixer.update_group(group_nxt)
        self._update_targets()

    def _change_group_contribution(self, batch: EpisodeBatch, change_group_i: int):
        if self.args.mixer != "group":
            raise RuntimeError("contribution group adjustment requires a mixer that exposes get_w1_avg().")
        if change_group_i == 0:
            self.agent_w1_avg = 0

        mac_hidden = []

        with th.no_grad():
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                self.mac.forward(batch, t=t)
                mac_hidden.append(self.mac.hidden_states)
            mac_hidden = th.stack(mac_hidden, dim=1)

            w1_avg = self.mixer.get_w1_avg(mac_hidden[:, :-1])
            self.agent_w1_avg += w1_avg

        if change_group_i == self.args.change_group_batch_num - 1:

            self.agent_w1_avg /= self.args.change_group_batch_num
            group_now = copy.deepcopy(self.mixer.group)
            group_nxt = copy.deepcopy(self.mixer.group)
            for group_index, group_i in enumerate(group_now):
                group_w1_avg = self.agent_w1_avg[group_i]

                group_avg = th.mean(group_w1_avg)
                relative_lasso_threshold = group_avg * self.args.change_group_value
                indices = th.where(group_w1_avg < relative_lasso_threshold)[0]

                if len(group_i) < 3:
                    continue

                if group_index+1 == len(group_now) and len(indices) != 0:
                    tmp = []
                    group_nxt.append(tmp)
                    self.mixer.add_new_net()
                    self.target_mixer.add_new_net()

                for i in range(len(indices)-1, -1, -1):
                    idx = group_now[group_index][indices[i]]
                    group_nxt[group_index+1].append(idx)
                    del group_nxt[group_index][indices[i]]
                    for m in self.mixer.hyper_w1[idx]:
                        if type(m) != nn.ReLU:
                            m.reset_parameters()

            whether_group_changed = True if group_now != group_nxt else False

            if not whether_group_changed:
                return

            for i in range(len(group_nxt)-1, -1, -1):
                if group_nxt[i] == []:
                    del group_nxt[i]
                    self.mixer.del_net(i)
                    self.target_mixer.del_net(i)

            self.mixer.update_group(group_nxt)
            self.target_mixer.update_group(group_nxt)
            self._update_targets()

    def _change_group_graph_pseudo_attn(self, batch: EpisodeBatch, change_group_i: int):
        if change_group_i == 0:
            self.graph_adj_avg = None

        with th.no_grad():
            obs = batch["obs"][:, :-1]
            filled = batch["filled"][:, :-1].squeeze(-1).long()
            valid_lengths = filled.sum(dim=1).clamp(min=1)
            last_indices = valid_lengths - 1
            batch_indices = th.arange(batch.batch_size, device=obs.device)
            last_obs = obs[batch_indices, last_indices]
            graph = pseudo_attention_graph(last_obs)

        if self.graph_adj_avg is None:
            self.graph_adj_avg = graph
        else:
            self.graph_adj_avg += graph

        graph_batch_num = getattr(self.args, "graph_change_group_batch_num", 1)
        if change_group_i != graph_batch_num - 1:
            return

        graph = self.graph_adj_avg / float(graph_batch_num)
        topk = getattr(self.args, "graph_topk", None)
        threshold = getattr(self.args, "graph_edge_threshold", 0.0)
        graph = sparsify_graph(graph, topk=topk, threshold=threshold)
        group_nxt = adjacency_to_groups(graph)
        self._apply_group_update(group_nxt)

    def _change_group_graph_local_subgraph(self, batch: EpisodeBatch, change_group_i: int):
        if change_group_i == 0:
            self.graph_adj_avg = None

        with th.no_grad():
            obs = batch["obs"][:, :-1]
            filled = batch["filled"][:, :-1].squeeze(-1).long()
            valid_lengths = filled.sum(dim=1).clamp(min=1)
            last_indices = valid_lengths - 1
            batch_indices = th.arange(batch.batch_size, device=obs.device)
            last_obs = obs[batch_indices, last_indices]
            neighbor_topk = getattr(self.args, "graph_local_neighbor_topk", None)
            graph = local_subgraph_similarity_graph(last_obs, neighbor_topk=neighbor_topk)

        if self.graph_adj_avg is None:
            self.graph_adj_avg = graph
        else:
            self.graph_adj_avg += graph

        graph_batch_num = getattr(self.args, "graph_change_group_batch_num", 1)
        if change_group_i != graph_batch_num - 1:
            return

        graph = self.graph_adj_avg / float(graph_batch_num)
        topk = getattr(self.args, "graph_topk", None)
        threshold = getattr(self.args, "graph_edge_threshold", 0.0)
        graph = sparsify_graph(graph, topk=topk, threshold=threshold)
        group_nxt = adjacency_to_groups(graph)
        self._apply_group_update(group_nxt)

    def _change_group_graph_local_fusion(self, batch: EpisodeBatch, change_group_i: int):
        if change_group_i == 0:
            self.graph_adj_avg = None

        with th.no_grad():
            obs = batch["obs"][:, :-1]
            filled = batch["filled"][:, :-1].squeeze(-1).long()
            valid_lengths = filled.sum(dim=1).clamp(min=1)
            last_indices = valid_lengths - 1
            batch_indices = th.arange(batch.batch_size, device=obs.device)
            last_obs = obs[batch_indices, last_indices]
            neighbor_topk = getattr(self.args, "graph_local_neighbor_topk", None)
            graph = local_subgraph_fusion_graph(last_obs, neighbor_topk=neighbor_topk)

        if self.graph_adj_avg is None:
            self.graph_adj_avg = graph
        else:
            self.graph_adj_avg += graph

        graph_batch_num = getattr(self.args, "graph_change_group_batch_num", 1)
        if change_group_i != graph_batch_num - 1:
            return

        graph = self.graph_adj_avg / float(graph_batch_num)
        topk = getattr(self.args, "graph_topk", None)
        threshold = getattr(self.args, "graph_edge_threshold", 0.0)
        graph = sparsify_graph(graph, topk=topk, threshold=threshold)
        group_nxt = adjacency_to_groups(graph)
        self._apply_group_update(group_nxt)

    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden = []
        mac_group_state = []
        mac_group_probs = []
        mac_group_graphs = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_hidden.append(self.mac.hidden_states)
            mac_group_state.append(self.mac.group_states)
            mac_group_probs.append(self.mac.group_probs)
            mac_group_graphs.append(self.mac.group_graphs)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)
        mac_hidden = th.stack(mac_hidden, dim=1)
        mac_group_state = th.stack(mac_group_state, dim=1)
        mac_group_probs = th.stack(mac_group_probs, dim=1)
        mac_group_graphs = th.stack(mac_group_graphs, dim=1)
        mac_hidden = mac_hidden.detach()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Mixer
        chosen_action_qvals, w1_avg_list, sd_loss = self.mixer(chosen_action_qvals, batch["state"][:, :-1], mac_hidden[:, :-1], mac_group_state[:, :-1], "eval")

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            target_mac_hidden = []
            target_mac_group_state = []

            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_hidden.append(self.target_mac.hidden_states)
                target_mac_group_state.append(self.target_mac.group_states)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_mac_hidden = th.stack(target_mac_hidden, dim=1)
            target_mac_group_state = th.stack(target_mac_group_state, dim=1)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"], target_mac_hidden, target_mac_group_state, "target")

            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)
        
        # lasso_alpha
        lasso_alpha = []
        for i in range(len(w1_avg_list)):
            lasso_alpha_time = self.args.lasso_alpha_start * (self.args.lasso_alpha_anneal ** (t_env//self.args.lasso_alpha_anneal_time))
            lasso_alpha.append(lasso_alpha_time)

        # lasso loss
        lasso_loss = th.tensor(0.0, device=chosen_action_qvals.device)
        for i in range(len(w1_avg_list)):
            group_w1_sum = th.sum(w1_avg_list[i])
            lasso_loss += group_w1_sum * lasso_alpha[i]

        if self.args.mixer == "group":
            sd_loss = sd_loss * mask
            sd_loss = self.args.sd_alpha * sd_loss.sum() / mask.sum()
            balance_loss = self._zero(chosen_action_qvals)
            conf_loss = self._zero(chosen_action_qvals)
            sparse_loss = self._zero(chosen_action_qvals)
        else:
            sd_loss = self.args.sd_alpha * self._compute_group_repulsion_loss(
                mac_group_state[:, :-1], mac_group_probs[:, :-1], mask
            )
            balance_loss, conf_loss, sparse_loss = self._compute_struct_group_regularizers(
                mac_group_probs[:, :-1], mac_group_graphs[:, :-1], mask
            )
            balance_loss = getattr(self.args, "group_balance_alpha", 0.0) * balance_loss
            conf_loss = getattr(self.args, "group_conf_alpha", 0.0) * conf_loss
            sparse_loss = getattr(self.args, "group_sparse_alpha", 0.0) * sparse_loss

        td_error = (chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = masked_td_error.sum() / mask.sum()

        loss = td_loss + lasso_loss + sd_loss + balance_loss + conf_loss + sparse_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self.logger.log_stat("total_loss", loss.item(), t_env)
            self.logger.log_stat("lasso_loss", lasso_loss.item(), t_env)
            self.logger.log_stat("sd_loss", sd_loss.item(), t_env)
            if self.args.mixer == "group_vdn":
                self.logger.log_stat("group_balance_loss", balance_loss.item(), t_env)
                self.logger.log_stat("group_conf_loss", conf_loss.item(), t_env)
                self.logger.log_stat("group_sparse_loss", sparse_loss.item(), t_env)
            
            self.log_stats_t = t_env
    
    def change_group(self, batch: EpisodeBatch, change_group_i: int):
        mode = getattr(self.args, "group_adjustment_mode", "contribution")
        if mode == "graph_pseudo_attn":
            return self._change_group_graph_pseudo_attn(batch, change_group_i)
        if mode == "graph_local_subgraph":
            return self._change_group_graph_local_subgraph(batch, change_group_i)
        if mode == "graph_local_fusion":
            return self._change_group_graph_local_fusion(batch, change_group_i)
        return self._change_group_contribution(batch, change_group_i)

    def log_group_stats(self, t_env: int, prefix="test_", group_trace=None, current_group=None, map_name="unknown_map"):
        group = copy.deepcopy(self.mixer.group)
        if current_group is not None:
            group = copy.deepcopy(current_group)
        if group_trace:
            last_group = group_trace[-1].get("group") if isinstance(group_trace[-1], dict) else None
            if last_group is not None:
                group = copy.deepcopy(last_group)
        self.logger.log_group(group, t_env, prefix=prefix)
        if getattr(self.args, "visualize_group_graph", False):
            self.logger.log_group_viz(
                group_trace,
                group,
                t_env,
                map_name,
                max_frames=getattr(self.args, "visualize_group_graph_max_frames", 24),
                fps=getattr(self.args, "visualize_group_graph_fps", 4),
                prefix=prefix,
            )

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
            np.save("{}/group.npy".format(path), self.mixer.group)
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.group = np.load("{}/group.npy".format(path))
            for i in range(len(self.mixer.group)-1):
                self.mixer.add_new_net()
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
