import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from envs.starcraft.smac_maps import get_map_params


class MLPHyperParameterGenerator(nn.Module):
    def __init__(self, embed_dim, output_dims, hyper_hidden_dim):
        super().__init__()
        self.output_dims = output_dims
        self.weight_mlps = nn.ModuleList()
        self.bias_mlps = nn.ModuleList()

        for layer_idx, (input_dim, output_dim) in enumerate(self.output_dims):
            is_final = layer_idx == len(self.output_dims) - 1
            gain = 1.0 if is_final else math.sqrt(2.0)

            weight_mlp = nn.Sequential(
                nn.Linear(embed_dim, hyper_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hyper_hidden_dim, input_dim * output_dim),
            )
            bias_mlp = nn.Sequential(
                nn.Linear(embed_dim, hyper_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hyper_hidden_dim, output_dim),
            )

            nn.init.orthogonal_(weight_mlp[0].weight, gain=math.sqrt(2.0))
            nn.init.zeros_(weight_mlp[0].bias)
            nn.init.orthogonal_(weight_mlp[2].weight, gain=gain)
            nn.init.zeros_(weight_mlp[2].bias)

            nn.init.orthogonal_(bias_mlp[0].weight, gain=math.sqrt(2.0))
            nn.init.zeros_(bias_mlp[0].bias)
            nn.init.zeros_(bias_mlp[2].weight)
            nn.init.zeros_(bias_mlp[2].bias)

            self.weight_mlps.append(weight_mlp)
            self.bias_mlps.append(bias_mlp)

    def forward(self, embeddings):
        weights = []
        biases = []
        for weight_mlp, bias_mlp, (input_dim, output_dim) in zip(
            self.weight_mlps, self.bias_mlps, self.output_dims
        ):
            weight = weight_mlp(embeddings).view(embeddings.size(0), input_dim, output_dim)
            bias = bias_mlp(embeddings).view(embeddings.size(0), 1, output_dim)
            weights.append(weight)
            biases.append(bias)
        return weights, biases


class StandardGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node_feat, adj):
        bsz, n_agents, _ = node_feat.shape
        identity = th.eye(n_agents, device=node_feat.device).unsqueeze(0).expand(bsz, -1, -1)
        adj_hat = adj + identity
        degree = adj_hat.sum(dim=-1).clamp(min=1e-6)
        inv_sqrt = degree.pow(-0.5)
        norm_adj = inv_sqrt.unsqueeze(-1) * adj_hat * inv_sqrt.unsqueeze(-2)
        mixed = th.bmm(norm_adj, node_feat)
        return self.linear(mixed)


class ObsGraphEncoder(nn.Module):
    def __init__(self, obs_dim, node_dim, gcn_layers, graph_topk):
        super().__init__()
        self.node_dim = node_dim
        self.graph_topk = graph_topk
        self.node_encoder = nn.Sequential(
            nn.Linear(obs_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )
        self.query = nn.Linear(node_dim, node_dim)
        self.key = nn.Linear(node_dim, node_dim)
        self.gcn_layers = nn.ModuleList(
            StandardGraphConv(node_dim, node_dim) for _ in range(max(1, gcn_layers))
        )

    def _apply_topk(self, adj):
        if self.graph_topk is None:
            return adj

        topk = max(1, min(self.graph_topk, adj.size(-1)))
        values, indices = th.topk(adj, k=topk, dim=-1)
        masked = th.zeros_like(adj)
        masked.scatter_(-1, indices, values)
        denom = masked.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return masked / denom

    def _build_adj(self, node_tokens):
        scale = math.sqrt(float(self.node_dim))
        query = self.query(node_tokens)
        key = self.key(node_tokens)
        score = th.matmul(query, key.transpose(-1, -2)) / scale
        adj = F.softmax(score, dim=-1)
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        return self._apply_topk(adj)

    def forward(self, obs):
        node_tokens = self.node_encoder(obs)
        adj = self._build_adj(node_tokens)
        graph_feat = node_tokens
        for layer_idx, layer in enumerate(self.gcn_layers):
            graph_feat = layer(graph_feat, adj)
            if layer_idx != len(self.gcn_layers) - 1:
                graph_feat = F.relu(graph_feat, inplace=True)
        return graph_feat, adj, node_tokens


class RPGInspiredRelationCapturer(nn.Module):
    # RPG-inspired single-task adaptation:
    # we borrow observation splitting, first-person relation capture, and a
    # temporal relation state, but we do not reproduce RPG's continual-learning
    # regularizers, task embedding, or structured ego/interaction decision heads.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
    ):
        super().__init__()
        self.move_dim = move_dim
        self.own_dim = own_dim
        self.ally_feat_dim = ally_feat_dim
        self.enemy_feat_dim = enemy_feat_dim
        self.relation_dim = relation_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(move_dim + own_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.ally_encoder = nn.Sequential(
            nn.Linear(ally_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )

        self.self_query = nn.Linear(relation_dim, relation_dim)
        self.ally_key = nn.Linear(relation_dim, relation_dim)
        self.ally_value = nn.Linear(relation_dim, relation_dim)
        self.enemy_key = nn.Linear(relation_dim, relation_dim)
        self.enemy_value = nn.Linear(relation_dim, relation_dim)

        self.instant_pattern = nn.Sequential(
            nn.Linear(relation_dim * 3, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.temporal_gru = nn.GRUCell(relation_dim * 2, relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def _masked_cross_attention(self, query, tokens, mask, key_proj, value_proj):
        scale = math.sqrt(float(self.relation_dim))
        key = key_proj(tokens)
        value = value_proj(tokens)
        logits = th.matmul(self.self_query(query).unsqueeze(2), key.transpose(-1, -2)).squeeze(2) / scale
        valid_mask = mask.bool()
        valid_any = valid_mask.any(dim=-1, keepdim=True)
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        attn = F.softmax(masked_logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))
        context = th.matmul(attn.unsqueeze(2), value).squeeze(2)
        return context, attn

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        self_token = self.self_encoder(self_feat)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        ally_tokens = self.ally_encoder(ally_feat) * ally_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        ally_context, ally_attn = self._masked_cross_attention(
            self_token, ally_tokens, ally_mask, self.ally_key, self.ally_value
        )
        enemy_context, enemy_attn = self._masked_cross_attention(
            self_token, enemy_tokens, enemy_mask, self.enemy_key, self.enemy_value
        )

        instant = self.instant_pattern(th.cat([self_token, ally_context, enemy_context], dim=-1))
        temporal_input = th.cat([self_token, instant], dim=-1)

        batch_size, n_agents, _ = temporal_input.shape
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class EgoGATLayer(nn.Module):
    def __init__(self, relation_dim):
        super().__init__()
        self.relation_dim = relation_dim
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.out = nn.Linear(relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, self_token, entity_tokens, entity_mask):
        batch_size, n_agents, _ = self_token.shape
        self_node = self_token.unsqueeze(2)
        nodes = th.cat([self_node, entity_tokens], dim=2)
        self_mask = th.ones(batch_size, n_agents, 1, device=self_token.device, dtype=th.bool)
        node_mask = th.cat([self_mask, entity_mask.bool()], dim=-1)

        query = self.query(self_token).unsqueeze(2)
        key = self.key(nodes)
        value = self.value(nodes)
        logits = (query * key).sum(dim=-1) / math.sqrt(float(self.relation_dim))
        logits = logits.masked_fill(~node_mask, -1e9)
        attn = F.softmax(logits, dim=-1)
        context = (attn.unsqueeze(-1) * value).sum(dim=2)
        updated = self.norm(self_token + F.elu(self.out(context), inplace=True))
        return updated, attn[:, :, 1:]


class TwoGraphGATRelationCapturer(nn.Module):
    # Explicit ego graph variant of RPG's relation capturer. It builds two
    # local graphs per agent, self+allies and self+enemies, and reads the
    # updated self node as the relation pattern source.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
    ):
        super().__init__()
        self.relation_dim = relation_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(move_dim + own_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.ally_encoder = nn.Sequential(
            nn.Linear(ally_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )

        self.ally_graph = EgoGATLayer(relation_dim)
        self.enemy_graph = EgoGATLayer(relation_dim)
        self.instant_pattern = nn.Sequential(
            nn.Linear(relation_dim * 3, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.temporal_gru = nn.GRUCell(relation_dim * 2, relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        self_token = self.self_encoder(self_feat)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        ally_tokens = self.ally_encoder(ally_feat) * ally_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        ally_self, ally_attn = self.ally_graph(self_token, ally_tokens, ally_mask)
        enemy_self, enemy_attn = self.enemy_graph(self_token, enemy_tokens, enemy_mask)
        instant = self.instant_pattern(th.cat([self_token, ally_self, enemy_self], dim=-1))
        temporal_input = th.cat([self_token, instant], dim=-1)

        batch_size, n_agents, _ = temporal_input.shape
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class TypedEgoGATMessage(nn.Module):
    def __init__(self, relation_dim):
        super().__init__()
        self.relation_dim = relation_dim
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.out = nn.Linear(relation_dim, relation_dim)

    def forward(self, self_token, entity_tokens, entity_mask):
        valid_mask = entity_mask.bool()
        valid_any = valid_mask.any(dim=-1, keepdim=True)
        query = self.query(self_token).unsqueeze(2)
        key = self.key(entity_tokens)
        value = self.value(entity_tokens)
        logits = (query * key).sum(dim=-1) / math.sqrt(float(self.relation_dim))
        logits = logits.masked_fill(~valid_mask, -1e9)
        attn = F.softmax(logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))
        message = (attn.unsqueeze(-1) * value).sum(dim=2)
        return F.elu(self.out(message), inplace=True), attn


class HeteroGATRelationCapturer(nn.Module):
    # Ego-centric heterogeneous graph variant. It keeps separate message
    # parameters for self-loop, ally->self, and enemy->self relation types,
    # then uses type-level attention to fuse the typed messages.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
    ):
        super().__init__()
        self.relation_dim = relation_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(move_dim + own_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.ally_encoder = nn.Sequential(
            nn.Linear(ally_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )

        self.self_loop = nn.Linear(relation_dim, relation_dim)
        self.ally_to_self = TypedEgoGATMessage(relation_dim)
        self.enemy_to_self = TypedEgoGATMessage(relation_dim)
        self.type_score = nn.Linear(relation_dim, 1)
        self.type_norm = nn.LayerNorm(relation_dim)
        self.instant_pattern = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.temporal_gru = nn.GRUCell(relation_dim * 2, relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        self_token = self.self_encoder(self_feat)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        ally_tokens = self.ally_encoder(ally_feat) * ally_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        self_msg = F.elu(self.self_loop(self_token), inplace=True)
        ally_msg, ally_attn = self.ally_to_self(self_token, ally_tokens, ally_mask)
        enemy_msg, enemy_attn = self.enemy_to_self(self_token, enemy_tokens, enemy_mask)

        typed_messages = th.stack([self_msg, ally_msg, enemy_msg], dim=2)
        type_mask = th.stack(
            [
                th.ones_like(ally_mask[..., :1], dtype=th.bool),
                ally_mask.any(dim=-1, keepdim=True),
                enemy_mask.any(dim=-1, keepdim=True),
            ],
            dim=2,
        ).squeeze(-1)
        type_logits = self.type_score(typed_messages).squeeze(-1).masked_fill(~type_mask, -1e9)
        type_attn = F.softmax(type_logits, dim=-1)
        hetero_context = (type_attn.unsqueeze(-1) * typed_messages).sum(dim=2)
        hetero_context = self.type_norm(self_token + hetero_context)

        instant = self.instant_pattern(th.cat([self_token, hetero_context], dim=-1))
        temporal_input = th.cat([self_token, instant], dim=-1)

        batch_size, n_agents, _ = temporal_input.shape
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class GlobalGraphGATLayer(nn.Module):
    def __init__(self, relation_dim):
        super().__init__()
        self.relation_dim = relation_dim
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.out = nn.Linear(relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, nodes, node_mask):
        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)
        logits = th.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.relation_dim))

        source_mask = node_mask.bool().unsqueeze(1)
        valid_any = node_mask.bool().any(dim=-1, keepdim=True).unsqueeze(-1)
        logits = logits.masked_fill(~source_mask, -1e9)
        attn = F.softmax(logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))

        context = th.matmul(attn, value)
        updated = self.norm(nodes + F.elu(self.out(context), inplace=True))
        updated = updated * node_mask.unsqueeze(-1).float()
        return updated, attn


class GlobalCrossAttention(nn.Module):
    def __init__(self, relation_dim):
        super().__init__()
        self.relation_dim = relation_dim
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.out = nn.Linear(relation_dim, relation_dim)

    def forward(self, query_nodes, key_nodes, key_mask):
        query = self.query(query_nodes)
        key = self.key(key_nodes)
        value = self.value(key_nodes)
        logits = th.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.relation_dim))

        valid_mask = key_mask.bool().unsqueeze(1)
        valid_any = key_mask.bool().any(dim=-1, keepdim=True).unsqueeze(-1)
        logits = logits.masked_fill(~valid_mask, -1e9)
        attn = F.softmax(logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))

        context = th.matmul(attn, value)
        return F.elu(self.out(context), inplace=True), attn


class GlobalTwoGraphGATRelationEncoder(nn.Module):
    # CTCE upper-bound graph encoder. It builds one global friendly graph and
    # one global enemy graph per timestep, then bridges enemy information back
    # into each friendly node through cross-graph attention.
    def __init__(
        self,
        move_dim,
        own_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
    ):
        super().__init__()
        self.relation_dim = relation_dim
        self.friend_encoder = nn.Sequential(
            nn.Linear(move_dim + own_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.friend_graph = GlobalGraphGATLayer(relation_dim)
        self.enemy_graph = GlobalGraphGATLayer(relation_dim)
        self.enemy_to_friend = GlobalCrossAttention(relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def _pool_global_enemy_features(self, enemy_feat):
        visible = enemy_feat.abs().sum(dim=-1) > 0
        count = visible.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (enemy_feat * visible.unsqueeze(-1).float()).sum(dim=1) / count
        return pooled, visible.any(dim=1)

    def forward(self, self_feat, enemy_feat):
        batch_size, n_agents, _ = self_feat.shape
        friend_mask = th.ones(batch_size, n_agents, device=self_feat.device, dtype=th.bool)
        friend_tokens = self.friend_encoder(self_feat)

        pooled_enemy, enemy_mask = self._pool_global_enemy_features(enemy_feat)
        enemy_tokens = self.enemy_encoder(pooled_enemy) * enemy_mask.unsqueeze(-1).float()

        friend_graph_tokens, _ = self.friend_graph(friend_tokens, friend_mask)
        enemy_graph_tokens, _ = self.enemy_graph(enemy_tokens, enemy_mask)
        enemy_context, _ = self.enemy_to_friend(friend_graph_tokens, enemy_graph_tokens, enemy_mask)
        return self.output_encoder(th.cat([friend_graph_tokens, enemy_context], dim=-1))


class GlobalHeteroGATRelationEncoder(nn.Module):
    # CTCE upper-bound heterogeneous graph encoder. Friendly and enemy nodes
    # share one global graph, while node-type embeddings and edge-type
    # embeddings tell attention which semantic relation each message represents.
    def __init__(
        self,
        move_dim,
        own_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
    ):
        super().__init__()
        self.relation_dim = relation_dim
        self.friend_encoder = nn.Sequential(
            nn.Linear(move_dim + own_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.node_type_embed = nn.Embedding(2, relation_dim)
        self.edge_type_bias = nn.Embedding(4, 1)
        self.edge_type_value = nn.Embedding(4, relation_dim)
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.out = nn.Linear(relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def _pool_global_enemy_features(self, enemy_feat):
        visible = enemy_feat.abs().sum(dim=-1) > 0
        count = visible.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (enemy_feat * visible.unsqueeze(-1).float()).sum(dim=1) / count
        return pooled, visible.any(dim=1)

    def forward(self, self_feat, enemy_feat):
        batch_size, n_agents, _ = self_feat.shape
        pooled_enemy, enemy_mask = self._pool_global_enemy_features(enemy_feat)
        n_enemies = pooled_enemy.size(1)

        friend_tokens = self.friend_encoder(self_feat)
        enemy_tokens = self.enemy_encoder(pooled_enemy) * enemy_mask.unsqueeze(-1).float()
        friend_type = th.zeros(n_agents, device=self_feat.device, dtype=th.long)
        enemy_type = th.ones(n_enemies, device=self_feat.device, dtype=th.long)
        node_types = th.cat([friend_type, enemy_type], dim=0)
        nodes = th.cat([friend_tokens, enemy_tokens], dim=1) + self.node_type_embed(node_types).unsqueeze(0)

        friend_mask = th.ones(batch_size, n_agents, device=self_feat.device, dtype=th.bool)
        node_mask = th.cat([friend_mask, enemy_mask], dim=1)
        edge_types = node_types.unsqueeze(1) * 2 + node_types.unsqueeze(0)

        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)
        logits = th.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.relation_dim))
        logits = logits + self.edge_type_bias(edge_types).squeeze(-1).unsqueeze(0)
        logits = logits.masked_fill(~node_mask.unsqueeze(1), -1e9)
        attn = F.softmax(logits, dim=-1)

        edge_value = self.edge_type_value(edge_types).unsqueeze(0)
        typed_value = value.unsqueeze(1) + edge_value
        context = (attn.unsqueeze(-1) * typed_value).sum(dim=2)
        updated = self.norm(nodes + F.elu(self.out(context), inplace=True))
        return self.output_encoder(updated[:, :n_agents])


class CleanHyperAgent(nn.Module):
    MODEL_SPECS = {
        "baseline": {"uses_hypernet": True, "execution_scope": "ctde"},
        "hypermarl_id": {"uses_hypernet": True, "execution_scope": "ctde"},
        "hypermarl_fullnet": {"uses_hypernet": True, "execution_scope": "ctde"},
        "dynamic_route": {"uses_hypernet": True, "execution_scope": "ctde"},
        "local_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_route": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_full_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_readout_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_linear_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_residual_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_film_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_moe_interaction_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_smooth_linear_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_fixed_structured_maker": {"uses_hypernet": False, "execution_scope": "ctde"},
        "rpg_fixed_linear_structured_maker": {"uses_hypernet": False, "execution_scope": "ctde"},
        "two_graph_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "hetero_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "global_two_graph_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctce"},
        "global_hetero_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctce"},
        "graph_hypercond": {"uses_hypernet": True, "execution_scope": "ctce"},
        "graph_route": {"uses_hypernet": True, "execution_scope": "ctce"},
        "qmix_minimal": {"uses_hypernet": False, "execution_scope": "ctde"},
    }

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.model_type = getattr(args, "clean_model_type", "baseline").replace("-", "_")
        if self.model_type == "hypermarl_mlp_hyper":
            self.model_type = "hypermarl_fullnet"
        if self.model_type not in self.MODEL_SPECS:
            raise ValueError(
                "Unknown clean_model_type={}. Expected one of {}.".format(
                    self.model_type, sorted(self.MODEL_SPECS.keys())
                )
            )

        self.execution_scope = self.MODEL_SPECS[self.model_type]["execution_scope"]
        self.is_ctce_model = self.execution_scope == "ctce"
        if self.is_ctce_model:
            print(
                "[clean_hyper_agent] {} currently runs in CTCE validation mode: "
                "graph construction uses all agents' observations at execution.".format(self.model_type)
            )

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = args.rnn_hidden_dim
        self.cond_dim = int(getattr(args, "clean_condition_dim", args.hypernet_embed))
        self.route_num = int(getattr(args, "clean_route_num", 4))
        self.route_temperature = float(getattr(args, "clean_route_temperature", 1.0))
        self.id_embed_dim = int(getattr(args, "clean_id_embed_dim", self.cond_dim))
        self.graph_node_dim = int(getattr(args, "clean_graph_node_dim", self.cond_dim))
        self.graph_layers = int(getattr(args, "clean_graph_layers", 1))
        self.graph_topk = getattr(args, "clean_graph_topk", None)
        self.hyper_mlp_hidden_dim = int(getattr(args, "clean_hyper_mlp_hidden_dim", 64))
        self.apply_hypermarl_init = bool(getattr(args, "clean_apply_hypermarl_init", False))
        self.rpg_relation_dim = int(getattr(args, "clean_rpg_relation_dim", self.cond_dim))
        self.rpg_interaction_hidden_dim = int(getattr(args, "clean_rpg_interaction_hidden_dim", 16))
        self.rpg_interaction_experts = int(getattr(args, "clean_rpg_interaction_experts", 4))
        self.rpg_residual_gate_bias = float(getattr(args, "clean_rpg_residual_gate_bias", -1.0))
        self.smooth_head_loss_coef = float(getattr(args, "clean_smooth_head_loss_coef", 0.0))
        self.smooth_head_knn = int(getattr(args, "clean_smooth_head_knn", 4))
        self.smooth_head_sample_size = int(getattr(args, "clean_smooth_head_sample_size", 256))

        self.obs_dim = input_shape
        if getattr(args, "obs_last_action", False):
            self.obs_dim -= args.n_actions
        if getattr(args, "obs_agent_id", False):
            self.obs_dim -= args.n_agents
        self.obs_dim = max(0, self.obs_dim)

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        local_source_dim = self.obs_dim + args.n_actions + self.hidden_dim
        self.local_condition_encoder = nn.Sequential(
            nn.Linear(local_source_dim, self.cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        if self.model_type in {
            "local_structured_hypercond",
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
        }:
            self._init_rpg_relation_capturer()
        else:
            self.rpg_relation_capturer = None
            self.rpg_obs_layout = None

        if self.model_type in {"hypermarl_id", "hypermarl_fullnet"}:
            self.id_embeddings = nn.Embedding(self.n_agents, self.id_embed_dim)
            nn.init.orthogonal_(self.id_embeddings.weight)
            if self.model_type == "hypermarl_id":
                self.id_condition_encoder = nn.Sequential(
                    nn.Linear(self.id_embed_dim, self.cond_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.cond_dim, self.cond_dim),
                )
            else:
                self.id_condition_encoder = None
        else:
            self.id_embeddings = None
            self.id_condition_encoder = None

        if self.model_type in {"dynamic_route", "graph_route", "rpg_relation_route"}:
            self.route_logits_head = nn.Sequential(
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.route_num),
            )
            self.route_codebook = nn.Parameter(th.empty(self.route_num, self.cond_dim))
            nn.init.xavier_uniform_(self.route_codebook)
        else:
            self.route_logits_head = None
            self.route_codebook = None

        if self.model_type in {"graph_hypercond", "graph_route"}:
            self.graph_encoder = ObsGraphEncoder(
                obs_dim=self.obs_dim,
                node_dim=self.graph_node_dim,
                gcn_layers=self.graph_layers,
                graph_topk=self.graph_topk,
            )
            self.graph_condition_encoder = nn.Sequential(
                nn.Linear(self.graph_node_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
        else:
            self.graph_encoder = None
            self.graph_condition_encoder = None

        if self.model_type in {"global_two_graph_gat_hypercond", "global_hetero_gat_hypercond"}:
            self.rpg_obs_layout = self._build_rpg_obs_layout()
            global_graph_cls = (
                GlobalTwoGraphGATRelationEncoder
                if self.model_type == "global_two_graph_gat_hypercond"
                else GlobalHeteroGATRelationEncoder
            )
            self.global_graph_relation_encoder = global_graph_cls(
                move_dim=self.rpg_obs_layout["move_dim"],
                own_dim=self.rpg_obs_layout["own_dim"],
                enemy_feat_dim=self.rpg_obs_layout["enemy_feat_dim"],
                relation_dim=self.rpg_relation_dim,
                output_dim=self.cond_dim,
            )
        else:
            self.global_graph_relation_encoder = None

        if self.model_type == "hypermarl_fullnet":
            self.full_head_hypernet = MLPHyperParameterGenerator(
                embed_dim=self.id_embed_dim,
                output_dims=[
                    (self.hidden_dim, self.hidden_dim),
                    (self.hidden_dim, self.n_actions),
                ],
                hyper_hidden_dim=self.hyper_mlp_hidden_dim,
            )
        else:
            self.full_head_hypernet = None

        if self.MODEL_SPECS[self.model_type]["uses_hypernet"] and self.model_type not in {
            "hypermarl_fullnet",
            "local_structured_hypercond",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
        }:
            self.hyper_bottleneck_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
            self.hyper_bottleneck_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.hyper_out_w = nn.Linear(self.cond_dim, self.hidden_dim * self.n_actions)
            self.hyper_out_b = nn.Linear(self.cond_dim, self.n_actions)
            if self.apply_hypermarl_init:
                self._apply_hypermarl_style_init()
        else:
            self.hyper_bottleneck_w = None
            self.hyper_bottleneck_b = None
            self.hyper_out_w = None
            self.hyper_out_b = None
            self.fixed_head = (
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ELU(inplace=True),
                    nn.Linear(self.hidden_dim, self.n_actions),
                )
                if self.model_type == "qmix_minimal"
                else None
            )

        if self.model_type in {
            "local_structured_hypercond",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
        }:
            self.rpg_n_ego_actions = self.n_actions - self.rpg_obs_layout["n_enemies"]
            if self.model_type in {"rpg_fixed_structured_maker", "rpg_fixed_linear_structured_maker"}:
                self.rpg_ego_bottleneck_w = None
                self.rpg_ego_bottleneck_b = None
                self.rpg_ego_out_w = None
                self.rpg_ego_out_b = None
                self.rpg_ego_maker = nn.Sequential(
                    nn.Linear(self.hidden_dim + self.cond_dim, self.hidden_dim),
                    nn.ELU(inplace=True),
                    nn.Linear(self.hidden_dim, self.rpg_n_ego_actions),
                )
            else:
                self.rpg_ego_bottleneck_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
                self.rpg_ego_bottleneck_b = nn.Linear(self.cond_dim, self.hidden_dim)
                self.rpg_ego_out_w = nn.Linear(self.cond_dim, self.hidden_dim * self.rpg_n_ego_actions)
                self.rpg_ego_out_b = nn.Linear(self.cond_dim, self.rpg_n_ego_actions)
                self.rpg_ego_maker = None

            if self.model_type == "rpg_full_structured_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = nn.Linear(
                    self.cond_dim, self.rpg_interaction_input_dim * self.rpg_interaction_hidden_dim
                )
                self.rpg_interaction_bottleneck_b = nn.Linear(self.cond_dim, self.rpg_interaction_hidden_dim)
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_hidden_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_readout_structured_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_hidden_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = nn.Sequential(
                    nn.Linear(self.rpg_interaction_input_dim, self.rpg_interaction_hidden_dim),
                    nn.ELU(inplace=True),
                )
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_linear_interaction_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_smooth_linear_interaction_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_residual_interaction_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = nn.Linear(
                    self.hidden_dim + self.cond_dim + self.rpg_relation_dim, 1
                )
                self.rpg_interaction_gate = nn.Sequential(
                    nn.Linear(self.cond_dim, self.cond_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.cond_dim, 1),
                )
                nn.init.zeros_(self.rpg_interaction_gate[-1].weight)
                nn.init.constant_(self.rpg_interaction_gate[-1].bias, self.rpg_residual_gate_bias)
            elif self.model_type == "rpg_film_interaction_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = None
                self.rpg_interaction_out_b = None
                self.rpg_interaction_encoder = nn.Sequential(
                    nn.Linear(self.rpg_interaction_input_dim, self.rpg_interaction_hidden_dim),
                    nn.ELU(inplace=True),
                )
                self.rpg_interaction_film_gamma = nn.Linear(self.cond_dim, self.rpg_interaction_hidden_dim)
                self.rpg_interaction_film_beta = nn.Linear(self.cond_dim, self.rpg_interaction_hidden_dim)
                self.rpg_interaction_scorer = nn.Linear(self.rpg_interaction_hidden_dim, 1)
                nn.init.zeros_(self.rpg_interaction_film_gamma.weight)
                nn.init.zeros_(self.rpg_interaction_film_gamma.bias)
                nn.init.zeros_(self.rpg_interaction_film_beta.weight)
                nn.init.zeros_(self.rpg_interaction_film_beta.bias)
            elif self.model_type == "rpg_moe_interaction_head":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = None
                self.rpg_interaction_out_b = None
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
                self.rpg_interaction_expert_gate = nn.Sequential(
                    nn.Linear(self.cond_dim, self.cond_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.cond_dim, self.rpg_interaction_experts),
                )
                self.rpg_interaction_expert_heads = nn.ModuleList(
                    [nn.Linear(self.rpg_interaction_input_dim, 1) for _ in range(self.rpg_interaction_experts)]
                )
            elif self.model_type == "rpg_fixed_linear_structured_maker":
                self.rpg_interaction_input_dim = self.hidden_dim + self.cond_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = None
                self.rpg_interaction_out_b = None
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = nn.Linear(self.rpg_interaction_input_dim, 1)
            else:
                self.rpg_interaction_input_dim = self.hidden_dim + self.cond_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = None
                self.rpg_interaction_out_b = None
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = nn.Sequential(
                    nn.Linear(self.rpg_interaction_input_dim, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, 1),
                )
            if self.apply_hypermarl_init and self.model_type not in {
                "rpg_fixed_structured_maker",
                "rpg_fixed_linear_structured_maker",
            }:
                nn.init.orthogonal_(self.rpg_ego_bottleneck_w.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(self.rpg_ego_bottleneck_w.bias)
                nn.init.zeros_(self.rpg_ego_bottleneck_b.weight)
                nn.init.zeros_(self.rpg_ego_bottleneck_b.bias)
                nn.init.orthogonal_(self.rpg_ego_out_w.weight, gain=1.0)
                nn.init.zeros_(self.rpg_ego_out_w.bias)
                nn.init.zeros_(self.rpg_ego_out_b.weight)
                nn.init.zeros_(self.rpg_ego_out_b.bias)
                if self.model_type == "rpg_full_structured_hypercond":
                    nn.init.orthogonal_(self.rpg_interaction_bottleneck_w.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(self.rpg_interaction_bottleneck_w.bias)
                    nn.init.zeros_(self.rpg_interaction_bottleneck_b.weight)
                    nn.init.zeros_(self.rpg_interaction_bottleneck_b.bias)
                    nn.init.orthogonal_(self.rpg_interaction_out_w.weight, gain=1.0)
                    nn.init.zeros_(self.rpg_interaction_out_w.bias)
                    nn.init.zeros_(self.rpg_interaction_out_b.weight)
                    nn.init.zeros_(self.rpg_interaction_out_b.bias)
                elif self.model_type == "rpg_readout_structured_hypercond":
                    nn.init.orthogonal_(self.rpg_interaction_out_w.weight, gain=1.0)
                    nn.init.zeros_(self.rpg_interaction_out_w.bias)
                    nn.init.zeros_(self.rpg_interaction_out_b.weight)
                    nn.init.zeros_(self.rpg_interaction_out_b.bias)
                elif self.model_type == "rpg_linear_interaction_hypercond":
                    nn.init.orthogonal_(self.rpg_interaction_out_w.weight, gain=1.0)
                    nn.init.zeros_(self.rpg_interaction_out_w.bias)
                    nn.init.zeros_(self.rpg_interaction_out_b.weight)
                    nn.init.zeros_(self.rpg_interaction_out_b.bias)
                elif self.model_type in {
                    "rpg_residual_interaction_hypercond",
                    "rpg_smooth_linear_interaction_hypercond",
                }:
                    nn.init.orthogonal_(self.rpg_interaction_out_w.weight, gain=1.0)
                    nn.init.zeros_(self.rpg_interaction_out_w.bias)
                    nn.init.zeros_(self.rpg_interaction_out_b.weight)
                    nn.init.zeros_(self.rpg_interaction_out_b.bias)
        else:
            self.rpg_n_ego_actions = None
            self.rpg_ego_bottleneck_w = None
            self.rpg_ego_bottleneck_b = None
            self.rpg_ego_out_w = None
            self.rpg_ego_out_b = None
            self.rpg_ego_maker = None
            self.rpg_interaction_input_dim = None
            self.rpg_interaction_bottleneck_w = None
            self.rpg_interaction_bottleneck_b = None
            self.rpg_interaction_out_w = None
            self.rpg_interaction_out_b = None
            self.rpg_interaction_encoder = None
            self.rpg_interaction_scorer = None
            self.rpg_interaction_gate = None
            self.rpg_interaction_film_gamma = None
            self.rpg_interaction_film_beta = None
            self.rpg_interaction_expert_gate = None
            self.rpg_interaction_expert_heads = None

        self.latest_condition = None
        self.latest_aux_loss = None
        self.latest_generated_interaction_head = None
        self.latest_route_logits = None
        self.latest_route_indices = None
        self.latest_graph_adj = None
        self.latest_graph_nodes = None
        self.latest_relation_ally_attn = None
        self.latest_relation_enemy_attn = None

    def init_hidden(self):
        hidden_size = self.hidden_dim
        if self.model_type in {
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
        }:
            hidden_size += self.rpg_relation_dim
        return self.fc1.weight.new_zeros(hidden_size)

    def _apply_hypermarl_style_init(self):
        nn.init.orthogonal_(self.hyper_bottleneck_w.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.hyper_bottleneck_w.bias)
        nn.init.zeros_(self.hyper_bottleneck_b.weight)
        nn.init.zeros_(self.hyper_bottleneck_b.bias)
        nn.init.orthogonal_(self.hyper_out_w.weight, gain=1.0)
        nn.init.zeros_(self.hyper_out_w.bias)
        nn.init.zeros_(self.hyper_out_b.weight)
        nn.init.zeros_(self.hyper_out_b.bias)

    def _build_local_source(self, hidden, context):
        obs = context["obs"]
        prev_action = context["prev_action"]
        return th.cat([obs, prev_action, hidden], dim=-1)

    def _build_rpg_obs_layout(self):
        env_args = getattr(self.args, "env_args", {})
        if getattr(self.args, "env", None) != "sc2":
            raise ValueError("RPG-inspired relation variants currently only support env=sc2.")

        map_params = get_map_params(env_args["map_name"])
        shield_bits_ally = 1 if map_params["a_race"] == "P" else 0
        shield_bits_enemy = 1 if map_params["b_race"] == "P" else 0
        unit_type_bits = map_params["unit_type_bits"]

        move_dim = 4
        if env_args.get("obs_pathing_grid", False):
            move_dim += 8
        if env_args.get("obs_terrain_height", False):
            move_dim += 9

        enemy_feat_dim = 4 + unit_type_bits
        if env_args.get("obs_all_health", True):
            enemy_feat_dim += 1 + shield_bits_enemy

        ally_feat_dim = 4 + unit_type_bits
        if env_args.get("obs_all_health", True):
            ally_feat_dim += 1 + shield_bits_ally
        if env_args.get("obs_last_action", False):
            ally_feat_dim += self.n_actions

        own_dim = unit_type_bits
        if env_args.get("obs_own_health", True):
            own_dim += 1 + shield_bits_ally
        if env_args.get("obs_timestep_number", False):
            own_dim += 1

        return {
            "move_dim": move_dim,
            "enemy_feat_dim": enemy_feat_dim,
            "ally_feat_dim": ally_feat_dim,
            "own_dim": own_dim,
            "n_enemies": map_params["n_enemies"],
            "n_allies": self.n_agents - 1,
        }

    def _init_rpg_relation_capturer(self):
        self.rpg_obs_layout = self._build_rpg_obs_layout()
        capturer_cls = RPGInspiredRelationCapturer
        if self.model_type == "two_graph_gat_hypercond":
            capturer_cls = TwoGraphGATRelationCapturer
        elif self.model_type == "hetero_gat_hypercond":
            capturer_cls = HeteroGATRelationCapturer

        self.rpg_relation_capturer = capturer_cls(
            move_dim=self.rpg_obs_layout["move_dim"],
            own_dim=self.rpg_obs_layout["own_dim"],
            ally_feat_dim=self.rpg_obs_layout["ally_feat_dim"],
            enemy_feat_dim=self.rpg_obs_layout["enemy_feat_dim"],
            relation_dim=self.rpg_relation_dim,
            output_dim=self.cond_dim,
        )

    def _split_rpg_obs(self, obs):
        layout = self.rpg_obs_layout
        batch_size, n_agents, _ = obs.shape
        idx = 0

        move = obs[:, :, idx : idx + layout["move_dim"]]
        idx += layout["move_dim"]

        enemy_total = layout["n_enemies"] * layout["enemy_feat_dim"]
        enemy = obs[:, :, idx : idx + enemy_total].view(
            batch_size, n_agents, layout["n_enemies"], layout["enemy_feat_dim"]
        )
        idx += enemy_total

        ally_total = layout["n_allies"] * layout["ally_feat_dim"]
        ally = obs[:, :, idx : idx + ally_total].view(
            batch_size, n_agents, layout["n_allies"], layout["ally_feat_dim"]
        )
        idx += ally_total

        own = obs[:, :, idx : idx + layout["own_dim"]]
        return move, enemy, ally, own

    def _build_local_structured_condition(self, hidden, context):
        condition = self.local_condition_encoder(self._build_local_source(hidden, context))
        _, enemy_feat, _, _ = self._split_rpg_obs(context["obs"])
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        enemy_tokens = self.rpg_relation_capturer.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()
        return condition, enemy_tokens, enemy_mask

    def _build_rpg_condition(self, context, relation_hidden):
        obs = context["obs"]
        move_feat, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        (
            condition,
            new_relation_hidden,
            ally_attn,
            enemy_attn,
            enemy_tokens,
            enemy_mask,
        ) = self.rpg_relation_capturer(
            self_feat=self_feat,
            ally_feat=ally_feat,
            enemy_feat=enemy_feat,
            prev_relation_hidden=relation_hidden,
        )
        self.latest_relation_ally_attn = ally_attn.detach()
        self.latest_relation_enemy_attn = enemy_attn.detach()
        return condition, new_relation_hidden, enemy_tokens, enemy_mask

    def _build_global_graph_condition(self, context):
        move_feat, enemy_feat, _, own_feat = self._split_rpg_obs(context["obs"])
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        return self.global_graph_relation_encoder(self_feat, enemy_feat)

    def _route_from_logits(self, route_logits, test_mode):
        if test_mode:
            route_index = route_logits.argmax(dim=-1)
            route_weight = F.one_hot(route_index, num_classes=self.route_num).float()
        else:
            route_weight = F.gumbel_softmax(
                route_logits, tau=self.route_temperature, hard=True, dim=-1
            )
            route_index = route_weight.argmax(dim=-1)

        route_condition = th.matmul(route_weight, self.route_codebook)
        self.latest_route_logits = route_logits.detach()
        self.latest_route_indices = route_index.detach()
        return route_condition

    def _build_condition(self, hidden, context, test_mode):
        self.latest_route_logits = None
        self.latest_route_indices = None
        self.latest_graph_adj = None
        self.latest_graph_nodes = None
        self.latest_relation_ally_attn = None
        self.latest_relation_enemy_attn = None

        if self.model_type == "baseline":
            condition = self.local_condition_encoder(self._build_local_source(hidden, context))
        elif self.model_type == "hypermarl_id":
            agent_ids = th.arange(self.n_agents, device=hidden.device).view(1, self.n_agents).expand(hidden.size(0), -1)
            condition = self.id_condition_encoder(self.id_embeddings(agent_ids))
        elif self.model_type == "hypermarl_fullnet":
            agent_ids = th.arange(self.n_agents, device=hidden.device).view(1, self.n_agents).expand(hidden.size(0), -1)
            condition = self.id_embeddings(agent_ids)
        elif self.model_type == "dynamic_route":
            local_base = self.local_condition_encoder(self._build_local_source(hidden, context))
            route_logits = self.route_logits_head(local_base)
            condition = self._route_from_logits(route_logits, test_mode=test_mode)
        elif self.model_type in {
            "local_structured_hypercond",
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
        }:
            raise RuntimeError(
                "{} uses a dedicated condition path and should bypass _build_condition.".format(self.model_type)
            )
        elif self.model_type == "graph_hypercond":
            graph_feat, graph_adj, graph_nodes = self.graph_encoder(context["obs"])
            self.latest_graph_adj = graph_adj.detach()
            self.latest_graph_nodes = graph_nodes.detach()
            condition = self.graph_condition_encoder(graph_feat)
        elif self.model_type == "graph_route":
            graph_feat, graph_adj, graph_nodes = self.graph_encoder(context["obs"])
            self.latest_graph_adj = graph_adj.detach()
            self.latest_graph_nodes = graph_nodes.detach()
            graph_base = self.graph_condition_encoder(graph_feat)
            route_logits = self.route_logits_head(graph_base)
            condition = self._route_from_logits(route_logits, test_mode=test_mode)
        elif self.model_type in {"global_two_graph_gat_hypercond", "global_hetero_gat_hypercond"}:
            condition = self._build_global_graph_condition(context)
        elif self.model_type == "qmix_minimal":
            condition = None
        else:
            raise RuntimeError("Unhandled clean_model_type={}".format(self.model_type))

        if condition is not None:
            self.latest_condition = condition.detach()
        else:
            self.latest_condition = None
        return condition

    def _apply_dynamic_head(self, hidden, condition):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = condition.reshape(batch_size * n_agents, -1)

        bottleneck_w = self.hyper_bottleneck_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        bottleneck_b = self.hyper_bottleneck_b(flat_condition).view(batch_size * n_agents, 1, self.hidden_dim)
        out_w = self.hyper_out_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        out_b = self.hyper_out_b(flat_condition).view(batch_size * n_agents, 1, self.n_actions)

        mid = F.elu(th.bmm(flat_hidden, bottleneck_w) + bottleneck_b)
        q = th.bmm(mid, out_w) + out_b
        return q.view(batch_size, n_agents, self.n_actions)

    def _apply_full_hypermarl_head(self, hidden, id_embeddings):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_embeddings = id_embeddings.reshape(batch_size * n_agents, -1)
        weights, biases = self.full_head_hypernet(flat_embeddings)

        current = flat_hidden
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            current = th.bmm(current, weight) + bias
            if layer_idx != len(weights) - 1:
                current = F.elu(current)
        return current.view(batch_size, n_agents, self.n_actions)

    def _linear_generated_interaction(self, flat_interaction_input, flat_condition, batch_size, n_agents):
        interaction_out_w = self.rpg_interaction_out_w(flat_condition).view(
            batch_size * n_agents, self.rpg_interaction_input_dim, 1
        )
        interaction_out_b = self.rpg_interaction_out_b(flat_condition).view(batch_size * n_agents, 1, 1)
        generated_head = th.cat(
            [
                interaction_out_w.reshape(batch_size * n_agents, -1),
                interaction_out_b.reshape(batch_size * n_agents, -1),
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
        q_attack = th.bmm(flat_interaction_input, interaction_out_w) + interaction_out_b
        return q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"]), generated_head

    def _head_smoothness_loss(self, flat_condition, generated_head):
        if self.smooth_head_loss_coef <= 0.0 or generated_head.size(0) <= 1:
            return generated_head.new_zeros(())

        sample_size = min(self.smooth_head_sample_size, generated_head.size(0))
        if sample_size < 2:
            return generated_head.new_zeros(())

        if sample_size < generated_head.size(0):
            indices = th.linspace(
                0, generated_head.size(0) - 1, steps=sample_size, device=generated_head.device
            ).long()
            condition = flat_condition.index_select(0, indices)
            head = generated_head.index_select(0, indices)
        else:
            condition = flat_condition
            head = generated_head

        condition = F.normalize(condition, p=2, dim=-1)
        head = F.normalize(head, p=2, dim=-1)
        rel_dist = 1.0 - th.matmul(condition, condition.transpose(0, 1))
        rel_dist = rel_dist.masked_fill(th.eye(sample_size, device=rel_dist.device).bool(), float("inf"))
        knn = min(self.smooth_head_knn, sample_size - 1)
        neighbor_idx = rel_dist.topk(k=knn, largest=False, dim=-1).indices
        neighbor_head = head.index_select(0, neighbor_idx.reshape(-1)).view(sample_size, knn, -1)
        head_sim = (head.unsqueeze(1) * neighbor_head).sum(dim=-1)
        return (1.0 - head_sim).mean()

    def _apply_rpg_structured_maker(self, hidden, relation_condition, enemy_tokens, enemy_mask):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)

        if self.model_type in {"rpg_fixed_structured_maker", "rpg_fixed_linear_structured_maker"}:
            q_ego = self.rpg_ego_maker(th.cat([hidden, relation_condition], dim=-1))
        else:
            ego_bottleneck_w = self.rpg_ego_bottleneck_w(flat_condition).view(
                batch_size * n_agents, self.hidden_dim, self.hidden_dim
            )
            ego_bottleneck_b = self.rpg_ego_bottleneck_b(flat_condition).view(
                batch_size * n_agents, 1, self.hidden_dim
            )
            ego_out_w = self.rpg_ego_out_w(flat_condition).view(
                batch_size * n_agents, self.hidden_dim, self.rpg_n_ego_actions
            )
            ego_out_b = self.rpg_ego_out_b(flat_condition).view(
                batch_size * n_agents, 1, self.rpg_n_ego_actions
            )

            ego_mid = F.elu(th.bmm(flat_hidden, ego_bottleneck_w) + ego_bottleneck_b)
            q_ego = th.bmm(ego_mid, ego_out_w) + ego_out_b
            q_ego = q_ego.view(batch_size, n_agents, self.rpg_n_ego_actions)

        hidden_rep = hidden.unsqueeze(2).expand(-1, -1, self.rpg_obs_layout["n_enemies"], -1)
        if self.model_type == "rpg_full_structured_hypercond":
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            flat_interaction_input = interaction_input.reshape(
                batch_size * n_agents, self.rpg_obs_layout["n_enemies"], self.rpg_interaction_input_dim
            )
            interaction_bottleneck_w = self.rpg_interaction_bottleneck_w(flat_condition).view(
                batch_size * n_agents, self.rpg_interaction_input_dim, self.rpg_interaction_hidden_dim
            )
            interaction_bottleneck_b = self.rpg_interaction_bottleneck_b(flat_condition).view(
                batch_size * n_agents, 1, self.rpg_interaction_hidden_dim
            )
            interaction_out_w = self.rpg_interaction_out_w(flat_condition).view(
                batch_size * n_agents, self.rpg_interaction_hidden_dim, 1
            )
            interaction_out_b = self.rpg_interaction_out_b(flat_condition).view(
                batch_size * n_agents, 1, 1
            )

            interaction_mid = F.elu(
                th.bmm(flat_interaction_input, interaction_bottleneck_w) + interaction_bottleneck_b
            )
            q_attack = th.bmm(interaction_mid, interaction_out_w) + interaction_out_b
            q_attack = q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"])
        elif self.model_type == "rpg_readout_structured_hypercond":
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            interaction_feat = self.rpg_interaction_encoder(interaction_input)
            flat_interaction_feat = interaction_feat.reshape(
                batch_size * n_agents, self.rpg_obs_layout["n_enemies"], self.rpg_interaction_hidden_dim
            )
            interaction_out_w = self.rpg_interaction_out_w(flat_condition).view(
                batch_size * n_agents, self.rpg_interaction_hidden_dim, 1
            )
            interaction_out_b = self.rpg_interaction_out_b(flat_condition).view(
                batch_size * n_agents, 1, 1
            )
            q_attack = th.bmm(flat_interaction_feat, interaction_out_w) + interaction_out_b
            q_attack = q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"])
        elif self.model_type == "rpg_linear_interaction_hypercond":
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            flat_interaction_input = interaction_input.reshape(
                batch_size * n_agents, self.rpg_obs_layout["n_enemies"], self.rpg_interaction_input_dim
            )
            q_attack, _ = self._linear_generated_interaction(
                flat_interaction_input, flat_condition, batch_size, n_agents
            )
        elif self.model_type == "rpg_smooth_linear_interaction_hypercond":
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            flat_interaction_input = interaction_input.reshape(
                batch_size * n_agents, self.rpg_obs_layout["n_enemies"], self.rpg_interaction_input_dim
            )
            q_attack, generated_head = self._linear_generated_interaction(
                flat_interaction_input, flat_condition, batch_size, n_agents
            )
            self.latest_aux_loss = self.smooth_head_loss_coef * self._head_smoothness_loss(
                flat_condition, generated_head
            )
        elif self.model_type == "rpg_residual_interaction_hypercond":
            cond_rep = relation_condition.unsqueeze(2).expand(-1, -1, self.rpg_obs_layout["n_enemies"], -1)
            fixed_input = th.cat([hidden_rep, cond_rep, enemy_tokens], dim=-1)
            q_fixed = self.rpg_interaction_scorer(fixed_input).squeeze(-1)
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            flat_interaction_input = interaction_input.reshape(
                batch_size * n_agents, self.rpg_obs_layout["n_enemies"], self.rpg_interaction_input_dim
            )
            q_dynamic, _ = self._linear_generated_interaction(
                flat_interaction_input, flat_condition, batch_size, n_agents
            )
            gate = th.sigmoid(self.rpg_interaction_gate(relation_condition))
            q_attack = q_fixed + gate.squeeze(-1).unsqueeze(-1) * q_dynamic
        elif self.model_type == "rpg_film_interaction_hypercond":
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            interaction_feat = self.rpg_interaction_encoder(interaction_input)
            gamma = 1.0 + self.rpg_interaction_film_gamma(relation_condition).unsqueeze(2)
            beta = self.rpg_interaction_film_beta(relation_condition).unsqueeze(2)
            q_attack = self.rpg_interaction_scorer(gamma * interaction_feat + beta).squeeze(-1)
        elif self.model_type == "rpg_moe_interaction_head":
            interaction_input = th.cat([hidden_rep, enemy_tokens], dim=-1)
            expert_qs = th.stack(
                [expert(interaction_input).squeeze(-1) for expert in self.rpg_interaction_expert_heads],
                dim=-1,
            )
            expert_weight = F.softmax(self.rpg_interaction_expert_gate(relation_condition), dim=-1)
            q_attack = (expert_qs * expert_weight.unsqueeze(2)).sum(dim=-1)
        else:
            cond_rep = relation_condition.unsqueeze(2).expand(-1, -1, self.rpg_obs_layout["n_enemies"], -1)
            interaction_input = th.cat([hidden_rep, cond_rep, enemy_tokens], dim=-1)
            q_attack = self.rpg_interaction_scorer(interaction_input).squeeze(-1)
        q_attack = q_attack.masked_fill(~enemy_mask.bool(), 0.0)
        return th.cat([q_ego, q_attack], dim=-1)

    def forward(self, inputs, hidden_state, context=None, test_mode=False):
        batch_size, n_agents, _ = inputs.shape
        flat_inputs = inputs.reshape(batch_size * n_agents, -1)
        x = F.relu(self.fc1(flat_inputs), inplace=True)

        self.latest_route_logits = None
        self.latest_route_indices = None
        self.latest_graph_adj = None
        self.latest_graph_nodes = None
        self.latest_relation_ally_attn = None
        self.latest_relation_enemy_attn = None
        self.latest_condition = None
        self.latest_aux_loss = None
        self.latest_generated_interaction_head = None

        if hidden_state is None:
            hidden_state = self.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)
        if self.model_type in {
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
        }:
            policy_hidden_state = hidden_state[:, :, : self.hidden_dim]
            relation_hidden_state = hidden_state[:, :, self.hidden_dim :]
        else:
            policy_hidden_state = hidden_state
            relation_hidden_state = None

        flat_hidden = policy_hidden_state.reshape(batch_size * n_agents, -1)
        hidden = self.rnn(x, flat_hidden).view(batch_size, n_agents, self.hidden_dim)

        if self.model_type == "qmix_minimal":
            q = self.fixed_head(hidden)
            next_hidden = hidden
        else:
            if context is None:
                raise ValueError("{} requires context with obs/prev_action.".format(self.model_type))
            if self.model_type == "local_structured_hypercond":
                condition, enemy_tokens, enemy_mask = self._build_local_structured_condition(hidden, context)
                self.latest_condition = condition.detach()
                q = self._apply_rpg_structured_maker(hidden, condition, enemy_tokens, enemy_mask)
                next_hidden = hidden
            elif self.model_type in {
                "rpg_relation_hypercond",
                "rpg_relation_route",
                "rpg_structured_hypercond",
                "rpg_full_structured_hypercond",
                "rpg_readout_structured_hypercond",
                "rpg_linear_interaction_hypercond",
                "rpg_residual_interaction_hypercond",
                "rpg_film_interaction_hypercond",
                "rpg_moe_interaction_head",
                "rpg_smooth_linear_interaction_hypercond",
                "rpg_fixed_structured_maker",
                "rpg_fixed_linear_structured_maker",
                "two_graph_gat_hypercond",
                "hetero_gat_hypercond",
            }:
                relation_condition, next_relation_hidden, enemy_tokens, enemy_mask = self._build_rpg_condition(
                    context, relation_hidden_state
                )
                if self.model_type in {
                    "rpg_relation_hypercond",
                    "two_graph_gat_hypercond",
                    "hetero_gat_hypercond",
                }:
                    condition = relation_condition
                    self.latest_condition = condition.detach()
                    q = self._apply_dynamic_head(hidden, condition)
                elif self.model_type == "rpg_relation_route":
                    route_logits = self.route_logits_head(relation_condition)
                    condition = self._route_from_logits(route_logits, test_mode=test_mode)
                    self.latest_condition = condition.detach()
                    q = self._apply_dynamic_head(hidden, condition)
                else:
                    condition = relation_condition
                    self.latest_condition = condition.detach()
                    q = self._apply_rpg_structured_maker(hidden, relation_condition, enemy_tokens, enemy_mask)
                next_hidden = th.cat([hidden, next_relation_hidden], dim=-1)
            else:
                condition = self._build_condition(hidden, context, test_mode=test_mode)
                if self.model_type == "hypermarl_fullnet":
                    q = self._apply_full_hypermarl_head(hidden, condition)
                else:
                    q = self._apply_dynamic_head(hidden, condition)
                next_hidden = hidden

        return q, next_hidden
