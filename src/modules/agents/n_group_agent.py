import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GroupAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GroupAgent, self).__init__()
        self.args = args
        self.a_h_dim = args.rnn_hidden_dim
        self.action_dim = args.n_actions
        self.group_head_mode = getattr(args, "group_head_mode", "latent").replace("-", "_")
        self.group_num = getattr(args, "group_num", 3)
        self.group_assignment_tau = getattr(args, "group_assignment_tau", 1.0)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.group_embeddings = None

        if self.group_head_mode == "plain":
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        elif self.group_head_mode == "latent":
            self.hyper_group = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                nn.ReLU(inplace=True),
                nn.Linear(args.hypernet_embed, args.hypernet_embed),
                nn.Tanh(),
            )
            self.hyper_b = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))
            self.hyper_w = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))
        else:
            if self.group_head_mode == "fixed_group":
                if args.group is None:
                    raise ValueError("`group_head_mode=fixed_group` requires resolved `args.group`.")
                self.group_num = len(args.group)
                fixed_group_ids = th.full((args.n_agents,), -1, dtype=th.long)
                for group_id, group_i in enumerate(args.group):
                    for agent_id in group_i:
                        fixed_group_ids[agent_id] = group_id
                if (fixed_group_ids < 0).any():
                    raise ValueError("Every agent must appear in `args.group` for fixed_group mode.")
                self.register_buffer("fixed_group_ids", fixed_group_ids)
            elif self.group_head_mode == "graph_better_struct":
                self.attn_q = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                self.attn_k = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                self.struct_encoder = nn.Sequential(
                    nn.Linear(args.n_agents + 2, args.hypernet_embed),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hypernet_embed, args.hypernet_embed),
                    nn.Tanh(),
                )
                self.group_assign = nn.Linear(args.hypernet_embed, self.group_num)
            else:
                raise ValueError("Unknown `group_head_mode`: {}".format(self.group_head_mode))

            self.group_embeddings = nn.Parameter(th.randn(self.group_num, args.hypernet_embed) * 0.1)
            self.group_decoder = nn.Sequential(
                nn.Linear(args.hypernet_embed, args.hypernet_embed),
                nn.ReLU(inplace=True),
                nn.Linear(args.hypernet_embed, args.hypernet_embed),
                nn.Tanh(),
            )
            self.hyper_b = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))
            self.hyper_w = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))

        self.group_probs = None
        self.group_graphs = None
        self.current_groups = None
        self.group_struct_features = None
        self.group_role_prototypes = None

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def _groups_from_assignment(self, assignments):
        groups = [[] for _ in range(self.group_num)]
        for agent_id, group_id in enumerate(assignments.tolist()):
            if 0 <= group_id < self.group_num:
                groups[group_id].append(agent_id)
        return groups

    def _assignment_lists(self, probs):
        hard_assign = probs.argmax(dim=-1)
        return [self._groups_from_assignment(assignments) for assignments in hard_assign]

    def _build_graph_better_struct(self, h):
        b, a, _ = h.size()
        q = self.attn_q(h)
        k = self.attn_k(h)
        scores = th.matmul(q, k.transpose(1, 2)) / math.sqrt(self.a_h_dim)
        eye = th.eye(a, device=h.device, dtype=th.bool).unsqueeze(0)
        scores = scores.masked_fill(eye, -1e9)
        attn = th.softmax(scores, dim=-1)
        attn = 0.5 * (attn + attn.transpose(1, 2))
        attn = attn.masked_fill(eye, 0.0)

        degree = attn.sum(dim=-1, keepdim=True)
        entropy = -(attn.clamp(min=1e-8) * attn.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)
        struct_input = th.cat([attn, degree, entropy], dim=-1)
        struct_feat = self.struct_encoder(struct_input.reshape(b * a, -1)).view(b, a, -1)
        logits = self.group_assign(struct_feat) / max(self.group_assignment_tau, 1e-6)
        probs = th.softmax(logits, dim=-1)
        return probs, struct_feat, attn

    def _build_fixed_group(self, h):
        b, a, _ = h.size()
        probs = F.one_hot(self.fixed_group_ids, num_classes=self.group_num).float()
        probs = probs.unsqueeze(0).expand(b, -1, -1)
        zero_graph = h.new_zeros(b, a, a)
        zero_struct = h.new_zeros(b, a, self.args.hypernet_embed)
        return probs, zero_struct, zero_graph

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in).view(b, a, -1)

        if self.group_head_mode == "plain":
            q = self.fc2(h)
            group_state = h.new_zeros(b, a, self.args.hypernet_embed)
            group_probs = h.new_zeros(b, a, self.group_num)
            group_graphs = h.new_zeros(b, a, a)
            struct_feat = h.new_zeros(b, a, self.args.hypernet_embed)
        elif self.group_head_mode == "latent":
            h_detach = h.detach().reshape(b * a, -1)
            group_state = self.hyper_group(h_detach).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.detach().reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.detach().reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
            q = q.view(b, a, -1)
            group_probs = h.new_zeros(b, a, self.group_num)
            group_graphs = h.new_zeros(b, a, a)
            struct_feat = group_state
        elif self.group_head_mode == "fixed_group":
            group_probs, _, group_graphs = self._build_fixed_group(h)
            group_emb = th.matmul(group_probs, self.group_embeddings)
            group_state = self.group_decoder(group_emb.reshape(b * a, -1)).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
            q = q.view(b, a, -1)
            struct_feat = group_state
        else:
            group_probs, struct_feat, group_graphs = self._build_graph_better_struct(h)
            group_emb = th.matmul(group_probs, self.group_embeddings)
            group_state = self.group_decoder((struct_feat + group_emb).reshape(b * a, -1)).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
            q = q.view(b, a, -1)

        self.group_probs = group_probs.detach()
        self.group_graphs = group_graphs.detach()
        self.group_struct_features = struct_feat.detach()
        self.group_role_prototypes = None if self.group_embeddings is None else self.group_embeddings.detach()
        self.current_groups = self._assignment_lists(self.group_probs)

        return q, h, group_state, group_probs, group_graphs
