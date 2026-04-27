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
        self.no_group_head_compare_modes = {
            "graph_input_fusion_hidden_head",
            "graph_input_fusion_node_embed_head",
            "graph_input_fusion_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_head",
        }
        self.group_num = getattr(args, "group_num", 3)
        if self.group_head_mode == "graph_input_fusion_node_embed_struct_only" or self.group_head_mode in self.no_group_head_compare_modes:
            self.group_num = 1
        self.group_assignment_tau = getattr(args, "group_assignment_tau", 1.0)
        self.base_struct_stat_dim = 10
        self.struct_stat_dim = self.base_struct_stat_dim
        self.group_direct_topk = max(1, min(args.n_agents - 1, getattr(args, "group_direct_topk", 3)))
        self.group_ema_alpha = getattr(args, "group_ema_alpha", 0.8)
        self.graph_attention_tau = getattr(args, "graph_attention_tau", 0.5)
        self.graph_obs_dim = input_shape
        if getattr(args, "obs_last_action", False):
            self.graph_obs_dim -= args.n_actions
        if getattr(args, "obs_agent_id", False):
            self.graph_obs_dim -= args.n_agents
        self.graph_obs_dim = max(0, self.graph_obs_dim)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.group_embeddings = None
        self.role_prototypes = None
        self.struct_repr = None
        self.cached_group_probs = None
        self.cached_group_graphs = None
        self.cached_struct_features = None

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
            if self.group_head_mode in ["fixed_group", "graph_input_fusion_fixed_group"]:
                if args.group is None:
                    raise ValueError("`group_head_mode={}` requires resolved `args.group`.".format(self.group_head_mode))
                self.group_num = len(args.group)
                fixed_group_ids = th.full((args.n_agents,), -1, dtype=th.long)
                for group_id, group_i in enumerate(args.group):
                    for agent_id in group_i:
                        fixed_group_ids[agent_id] = group_id
                if (fixed_group_ids < 0).any():
                    raise ValueError("Every agent must appear in `args.group` for fixed_group mode.")
                self.register_buffer("fixed_group_ids", fixed_group_ids)
                if self.group_head_mode == "graph_input_fusion_fixed_group":
                    self.struct_stat_dim = args.n_agents + 2
                    self.attn_q = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                    self.attn_k = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                    self.graph_obs_proj = nn.Sequential(
                        nn.Linear(self.graph_obs_dim, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                    self.graph_action_proj = nn.Sequential(
                        nn.Linear(args.n_actions, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                    self.graph_input_fuse = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim * 3, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                    self.struct_encoder = nn.Sequential(
                        nn.Linear(self.struct_stat_dim, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
            elif self.group_head_mode in [
                "graph_better_struct",
                "graph_better_struct_proto",
                "graph_better_struct_repr",
                "graph_better_struct_slow",
                "graph_better_struct_sparse",
                "graph_better_struct_hybrid",
                "graph_better_struct_row_sparse",
                "graph_better_struct_topk_signature",
                "graph_better_struct_ego_subgraph",
                "graph_input_fusion",
                "graph_input_fusion_node_embed",
                "graph_input_fusion_node_embed_no_groupemb",
                "graph_input_fusion_node_embed_struct_only",
                "graph_input_fusion_node_embed_sharp",
                "graph_input_fusion_node_embed_threshold_group",
                "graph_input_fusion_hidden_head",
                "graph_input_fusion_node_embed_head",
                "graph_input_fusion_struct_feat_head",
                "graph_input_fusion_node_embed_struct_feat_head",
                "graph_input_fusion_fixed_group",
                "graph_input_fusion_group_only",
                "graph_input_fusion_head_only",
            ]:
                use_proto_assignment = self.group_head_mode == "graph_better_struct_proto"
                if self.group_head_mode == "graph_better_struct_hybrid":
                    self.struct_stat_dim = self.base_struct_stat_dim + args.n_agents
                elif self.group_head_mode == "graph_better_struct_row_sparse":
                    self.struct_stat_dim = args.n_agents + 2
                elif self.group_head_mode == "graph_better_struct_topk_signature":
                    self.struct_stat_dim = self.group_direct_topk + self.group_direct_topk * self.group_direct_topk + 2
                elif self.group_head_mode == "graph_better_struct_ego_subgraph":
                    self.struct_stat_dim = (self.group_direct_topk + 1) * (self.group_direct_topk + 1) + 2
                elif self.group_head_mode in [
                    "graph_input_fusion",
                    "graph_input_fusion_node_embed",
                    "graph_input_fusion_node_embed_no_groupemb",
                    "graph_input_fusion_node_embed_struct_only",
                    "graph_input_fusion_node_embed_sharp",
                    "graph_input_fusion_node_embed_threshold_group",
                    "graph_input_fusion_hidden_head",
                    "graph_input_fusion_node_embed_head",
                    "graph_input_fusion_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_head",
                    "graph_input_fusion_fixed_group",
                    "graph_input_fusion_group_only",
                    "graph_input_fusion_head_only",
                ]:
                    self.struct_stat_dim = args.n_agents + 2
                    if self.group_head_mode in [
                        "graph_input_fusion_node_embed",
                        "graph_input_fusion_node_embed_no_groupemb",
                        "graph_input_fusion_node_embed_struct_only",
                        "graph_input_fusion_node_embed_sharp",
                        "graph_input_fusion_node_embed_threshold_group",
                    ]:
                        self.struct_stat_dim += args.rnn_hidden_dim
                self.attn_q = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                self.attn_k = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                if self.group_head_mode == "graph_better_struct_repr":
                    self.struct_repr = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                elif self.group_head_mode in [
                    "graph_input_fusion",
                    "graph_input_fusion_node_embed",
                    "graph_input_fusion_node_embed_no_groupemb",
                    "graph_input_fusion_node_embed_struct_only",
                    "graph_input_fusion_node_embed_sharp",
                    "graph_input_fusion_node_embed_threshold_group",
                    "graph_input_fusion_hidden_head",
                    "graph_input_fusion_node_embed_head",
                    "graph_input_fusion_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_head",
                    "graph_input_fusion_fixed_group",
                    "graph_input_fusion_group_only",
                    "graph_input_fusion_head_only",
                ]:
                    self.graph_obs_proj = nn.Sequential(
                        nn.Linear(self.graph_obs_dim, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                    self.graph_action_proj = nn.Sequential(
                        nn.Linear(args.n_actions, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                    self.graph_input_fuse = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim * 3, args.rnn_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                    )
                if self.group_head_mode == "graph_input_fusion_hidden_head":
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                elif self.group_head_mode == "graph_input_fusion_node_embed_head":
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                elif self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_head":
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim + args.hypernet_embed, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                self.struct_encoder = nn.Sequential(
                    nn.Linear(self.struct_stat_dim, args.hypernet_embed),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hypernet_embed, args.hypernet_embed),
                    nn.Tanh(),
                )
                if not use_proto_assignment and self.group_head_mode != "graph_input_fusion_node_embed_struct_only" and self.group_head_mode not in self.no_group_head_compare_modes:
                    self.group_assign = nn.Linear(args.hypernet_embed, self.group_num)
                elif use_proto_assignment:
                    self.role_prototypes = nn.Parameter(th.randn(self.group_num, args.hypernet_embed) * 0.1)
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
        self.group_node_embeddings = None

    def init_hidden(self):
        self.cached_group_probs = None
        self.cached_group_graphs = None
        self.cached_struct_features = None
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

    def _build_attention_graph(self, h, graph_source=None):
        source = graph_source if graph_source is not None else (self.struct_repr(h) if self.struct_repr is not None else h)
        sharp_mode = self.group_head_mode == "graph_input_fusion_node_embed_sharp"
        if sharp_mode:
            source = F.normalize(source, p=2, dim=-1)
        b, a, _ = h.size()
        q = self.attn_q(source)
        k = self.attn_k(source)
        if sharp_mode:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            scores = th.matmul(q, k.transpose(1, 2))
        else:
            scores = th.matmul(q, k.transpose(1, 2)) / math.sqrt(self.a_h_dim)
        eye = th.eye(a, device=h.device, dtype=th.bool).unsqueeze(0)
        scores = scores.masked_fill(eye, -1e9)
        if sharp_mode:
            valid_mask = (~eye).float()
            score_no_diag = scores.masked_fill(eye, 0.0)
            row_mean = (score_no_diag * valid_mask).sum(dim=-1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            row_var = ((score_no_diag - row_mean).pow(2) * valid_mask).sum(dim=-1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            scores = (scores - row_mean) / row_var.sqrt().clamp(min=1e-6)
            scores = scores.masked_fill(eye, -1e9)
            attn = th.softmax(scores / max(self.graph_attention_tau, 1e-6), dim=-1)
        else:
            attn = th.softmax(scores, dim=-1)
        attn = 0.5 * (attn + attn.transpose(1, 2))
        attn = attn.masked_fill(eye, 0.0)
        if self.group_head_mode == "graph_better_struct_sparse":
            topk = getattr(self.args, "group_sparse_topk", None)
            topk = max(1, min(a - 1, topk if topk is not None else max(1, a // 2)))
            vals, idx = th.topk(attn, k=topk, dim=-1)
            mask = th.zeros_like(attn, dtype=th.bool)
            mask.scatter_(-1, idx, True)
            mask = mask & mask.transpose(1, 2)
            attn = attn * mask.float()
            attn = 0.5 * (attn + attn.transpose(1, 2))
        return attn

    def _build_graph_input_fusion_source(self, h, graph_context):
        b, a, _ = h.size()
        if graph_context is None:
            obs = h.new_zeros(b, a, self.graph_obs_dim)
            prev_action = h.new_zeros(b, a, self.action_dim)
        else:
            obs = graph_context["obs"]
            prev_action = graph_context["prev_action"]
        obs_feat = self.graph_obs_proj(obs.reshape(b * a, -1)).view(b, a, -1)
        action_feat = self.graph_action_proj(prev_action.reshape(b * a, -1)).view(b, a, -1)
        fused = th.cat([h, obs_feat, action_feat], dim=-1)
        return self.graph_input_fuse(fused.reshape(b * a, -1)).view(b, a, -1)

    def _build_topk_signature_input(self, attn, row_probs, degree, entropy):
        b, a, _ = attn.size()
        topk_vals, topk_idx = th.topk(row_probs, k=self.group_direct_topk, dim=-1)
        signatures = []
        for bi in range(b):
            local_signatures = []
            graph_b = attn[bi]
            for ai in range(a):
                nbrs = topk_idx[bi, ai]
                local_adj = graph_b.index_select(0, nbrs).index_select(1, nbrs)
                local_signatures.append(
                    th.cat(
                        [
                            topk_vals[bi, ai],
                            local_adj.reshape(-1),
                            degree[bi, ai],
                            entropy[bi, ai],
                        ],
                        dim=-1,
                    )
                )
            signatures.append(th.stack(local_signatures, dim=0))
        return th.stack(signatures, dim=0)

    def _build_ego_subgraph_input(self, attn, degree, entropy):
        b, a, _ = attn.size()
        _, topk_idx = th.topk(attn, k=self.group_direct_topk, dim=-1)
        signatures = []
        for bi in range(b):
            local_signatures = []
            graph_b = attn[bi]
            for ai in range(a):
                nbrs = topk_idx[bi, ai]
                center = nbrs.new_tensor([ai])
                nodes = th.cat([center, nbrs], dim=0)
                local_adj = graph_b.index_select(0, nodes).index_select(1, nodes)
                local_signatures.append(
                    th.cat(
                        [
                            local_adj.reshape(-1),
                            degree[bi, ai],
                            entropy[bi, ai],
                        ],
                        dim=-1,
                    )
                )
            signatures.append(th.stack(local_signatures, dim=0))
        return th.stack(signatures, dim=0)

    def _build_graph_struct_features(self, attn, node_embed=None):
        b, a, _ = attn.size()
        degree = attn.sum(dim=-1, keepdim=True)
        row_probs = attn / degree.clamp(min=1e-8)
        entropy = -(row_probs.clamp(min=1e-8) * row_probs.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)

        topk_vals = th.topk(attn, k=min(3, a), dim=-1).values
        top1 = topk_vals[..., 0:1]
        top2_mean = topk_vals[..., : min(2, topk_vals.size(-1))].mean(dim=-1, keepdim=True)
        top3_mean = topk_vals.mean(dim=-1, keepdim=True)
        second = topk_vals[..., 1:2] if topk_vals.size(-1) > 1 else top1
        max_second_gap = top1 - second

        a2 = th.matmul(attn, attn)
        two_hop_mass = a2.sum(dim=-1, keepdim=True)
        triangle_mass = th.diagonal(th.matmul(a2, attn), dim1=-2, dim2=-1).unsqueeze(-1)
        local_density = triangle_mass / degree.pow(2).clamp(min=1e-8)
        neighbor_degree = th.matmul(attn, degree) / degree.clamp(min=1e-8)
        peak_ratio = top1 / degree.clamp(min=1e-8)

        struct_stats = th.cat(
            [
                degree,
                entropy,
                top1,
                top2_mean,
                top3_mean,
                max_second_gap,
                two_hop_mass,
                triangle_mass,
                local_density,
                peak_ratio + neighbor_degree,
            ],
            dim=-1,
        )
        if self.group_head_mode == "graph_better_struct_hybrid":
            struct_input = th.cat([row_probs, struct_stats], dim=-1)
        elif self.group_head_mode == "graph_better_struct_row_sparse":
            struct_input = th.cat([row_probs, degree, entropy], dim=-1)
        elif self.group_head_mode in [
            "graph_input_fusion",
            "graph_input_fusion_node_embed",
            "graph_input_fusion_node_embed_no_groupemb",
            "graph_input_fusion_node_embed_struct_only",
            "graph_input_fusion_node_embed_sharp",
            "graph_input_fusion_node_embed_threshold_group",
            "graph_input_fusion_hidden_head",
            "graph_input_fusion_node_embed_head",
            "graph_input_fusion_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_head",
            "graph_input_fusion_fixed_group",
            "graph_input_fusion_group_only",
            "graph_input_fusion_head_only",
        ]:
            struct_input = th.cat([row_probs, degree, entropy], dim=-1)
            if self.group_head_mode in [
                "graph_input_fusion_node_embed",
                "graph_input_fusion_node_embed_no_groupemb",
                "graph_input_fusion_node_embed_struct_only",
                "graph_input_fusion_node_embed_sharp",
                "graph_input_fusion_node_embed_threshold_group",
            ]:
                struct_input = th.cat([node_embed, struct_input], dim=-1)
        elif self.group_head_mode == "graph_better_struct_topk_signature":
            struct_input = self._build_topk_signature_input(attn, row_probs, degree, entropy)
        elif self.group_head_mode == "graph_better_struct_ego_subgraph":
            struct_input = self._build_ego_subgraph_input(attn, degree, entropy)
        else:
            struct_input = struct_stats
        struct_feat = self.struct_encoder(struct_input.reshape(b * a, -1)).view(b, a, -1)
        return struct_feat

    def _build_graph_better_struct(self, h, graph_source=None, node_embed=None):
        attn = self._build_attention_graph(h, graph_source=graph_source)
        struct_feat = self._build_graph_struct_features(attn, node_embed=node_embed)
        logits = self.group_assign(struct_feat) / max(self.group_assignment_tau, 1e-6)
        probs = th.softmax(logits, dim=-1)
        if self.group_head_mode == "graph_better_struct_slow":
            if self.cached_group_probs is not None:
                probs = self.group_ema_alpha * self.cached_group_probs + (1.0 - self.group_ema_alpha) * probs
                struct_feat = self.group_ema_alpha * self.cached_struct_features + (1.0 - self.group_ema_alpha) * struct_feat
                attn = self.group_ema_alpha * self.cached_group_graphs + (1.0 - self.group_ema_alpha) * attn
            self.cached_group_probs = probs.detach()
            self.cached_struct_features = struct_feat.detach()
            self.cached_group_graphs = attn.detach()
        return probs, struct_feat, attn

    def _build_graph_better_struct_proto(self, h):
        attn = self._build_attention_graph(h)
        struct_feat = self._build_graph_struct_features(attn)
        proto = F.normalize(self.role_prototypes, p=2, dim=-1)
        feat = F.normalize(struct_feat, p=2, dim=-1)
        logits = th.einsum("bad,kd->bak", feat, proto) / max(self.group_assignment_tau, 1e-6)
        probs = th.softmax(logits, dim=-1)
        return probs, struct_feat, attn

    def _build_fixed_group(self, h):
        b, a, _ = h.size()
        probs = F.one_hot(self.fixed_group_ids, num_classes=self.group_num).float()
        probs = probs.unsqueeze(0).expand(b, -1, -1)
        zero_graph = h.new_zeros(b, a, a)
        zero_struct = h.new_zeros(b, a, self.args.hypernet_embed)
        return probs, zero_struct, zero_graph

    def _build_graph_input_fusion_fixed_group(self, h, graph_context):
        group_probs, _, _ = self._build_fixed_group(h)
        graph_source = self._build_graph_input_fusion_source(h, graph_context)
        group_graphs = self._build_attention_graph(h, graph_source=graph_source)
        struct_feat = self._build_graph_struct_features(group_graphs)
        return group_probs, struct_feat, group_graphs

    def _build_graph_input_fusion_node_embed_struct_only(self, h, graph_context):
        b, a, _ = h.size()
        node_embed = self._build_graph_input_fusion_source(h, graph_context)
        group_graphs = self._build_attention_graph(h, graph_source=node_embed)
        struct_feat = self._build_graph_struct_features(group_graphs, node_embed=node_embed)
        group_probs = h.new_ones(b, a, 1)
        return group_probs, struct_feat, group_graphs

    def _build_graph_input_fusion_no_group_head(self, h, graph_context):
        b, a, _ = h.size()
        group_probs = h.new_ones(b, a, 1)
        zero_graph = h.new_zeros(b, a, a)
        zero_struct = h.new_zeros(b, a, self.args.hypernet_embed)
        node_embed = None

        if self.group_head_mode == "graph_input_fusion_hidden_head":
            head_feat = self.head_input_encoder(h.reshape(b * a, -1)).view(b, a, -1)
            return group_probs, zero_struct, zero_graph, head_feat, node_embed

        node_embed = self._build_graph_input_fusion_source(h, graph_context)
        if self.group_head_mode == "graph_input_fusion_node_embed_head":
            head_feat = self.head_input_encoder(node_embed.reshape(b * a, -1)).view(b, a, -1)
            return group_probs, zero_struct, zero_graph, head_feat, node_embed

        group_graphs = self._build_attention_graph(h, graph_source=node_embed)
        struct_feat = self._build_graph_struct_features(group_graphs)
        if self.group_head_mode == "graph_input_fusion_struct_feat_head":
            head_feat = struct_feat
        else:
            fused = th.cat([node_embed, struct_feat], dim=-1)
            head_feat = self.head_input_encoder(fused.reshape(b * a, -1)).view(b, a, -1)
        return group_probs, struct_feat, group_graphs, head_feat, node_embed

    def forward(self, inputs, hidden_state, graph_context=None):
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
        elif self.group_head_mode == "graph_input_fusion_fixed_group":
            group_probs, struct_feat, group_graphs = self._build_graph_input_fusion_fixed_group(h, graph_context)
            group_emb = th.matmul(group_probs, self.group_embeddings)
            group_state = self.group_decoder((struct_feat + group_emb).reshape(b * a, -1)).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
            q = q.view(b, a, -1)
        elif self.group_head_mode in [
            "graph_better_struct",
            "graph_better_struct_repr",
            "graph_better_struct_slow",
            "graph_better_struct_sparse",
            "graph_better_struct_hybrid",
            "graph_better_struct_row_sparse",
            "graph_better_struct_topk_signature",
            "graph_better_struct_ego_subgraph",
            "graph_input_fusion",
            "graph_input_fusion_node_embed",
            "graph_input_fusion_node_embed_no_groupemb",
            "graph_input_fusion_node_embed_struct_only",
            "graph_input_fusion_node_embed_sharp",
            "graph_input_fusion_node_embed_threshold_group",
            "graph_input_fusion_hidden_head",
            "graph_input_fusion_node_embed_head",
            "graph_input_fusion_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_head",
            "graph_input_fusion_group_only",
            "graph_input_fusion_head_only",
        ]:
            if self.group_head_mode == "graph_input_fusion_node_embed_struct_only":
                group_probs, struct_feat, group_graphs = self._build_graph_input_fusion_node_embed_struct_only(h, graph_context)
                group_state = self.group_decoder(struct_feat.reshape(b * a, -1)).view(b, a, -1)
            elif self.group_head_mode in self.no_group_head_compare_modes:
                group_probs, struct_feat, group_graphs, head_feat, node_embed = self._build_graph_input_fusion_no_group_head(h, graph_context)
                group_state = self.group_decoder(head_feat.reshape(b * a, -1)).view(b, a, -1)
            else:
                graph_source = None
                node_embed = None
                if self.group_head_mode in [
                    "graph_input_fusion",
                    "graph_input_fusion_node_embed",
                    "graph_input_fusion_node_embed_no_groupemb",
                    "graph_input_fusion_node_embed_sharp",
                    "graph_input_fusion_node_embed_threshold_group",
                    "graph_input_fusion_hidden_head",
                    "graph_input_fusion_node_embed_head",
                    "graph_input_fusion_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_head",
                    "graph_input_fusion_group_only",
                    "graph_input_fusion_head_only",
                ]:
                    graph_source = self._build_graph_input_fusion_source(h, graph_context)
                    node_embed = (
                        graph_source
                        if self.group_head_mode in [
                            "graph_input_fusion_node_embed",
                            "graph_input_fusion_node_embed_no_groupemb",
                            "graph_input_fusion_node_embed_sharp",
                            "graph_input_fusion_node_embed_threshold_group",
                        ]
                        else None
                    )
                group_probs, struct_feat, group_graphs = self._build_graph_better_struct(
                    h, graph_source=graph_source, node_embed=node_embed
                )
                if self.group_head_mode == "graph_input_fusion_head_only":
                    group_probs = th.zeros_like(group_probs)
                    group_probs[..., 0] = 1.0
                group_emb = th.matmul(group_probs, self.group_embeddings)
                if self.group_head_mode == "graph_input_fusion_group_only":
                    group_state = self.group_decoder(group_emb.reshape(b * a, -1)).view(b, a, -1)
                elif self.group_head_mode == "graph_input_fusion_node_embed_no_groupemb":
                    group_state = self.group_decoder(struct_feat.reshape(b * a, -1)).view(b, a, -1)
                else:
                    group_state = self.group_decoder((struct_feat + group_emb).reshape(b * a, -1)).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
            q = q.view(b, a, -1)
        else:
            group_probs, struct_feat, group_graphs = self._build_graph_better_struct_proto(h)
            group_emb = th.matmul(group_probs, self.group_embeddings)
            group_state = self.group_decoder((struct_feat + group_emb).reshape(b * a, -1)).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
            q = q.view(b, a, -1)

        self.group_probs = group_probs
        self.group_graphs = group_graphs
        self.group_struct_features = struct_feat
        self.group_role_prototypes = self.role_prototypes
        self.group_node_embeddings = node_embed if "node_embed" in locals() and node_embed is not None else h.new_zeros(b, a, self.args.rnn_hidden_dim)
        self.current_groups = self._assignment_lists(self.group_probs.detach())

        return q, h, group_state, group_probs, group_graphs
