import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


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


class CleanHyperAgent(nn.Module):
    MODEL_SPECS = {
        "baseline": {"uses_hypernet": True, "execution_scope": "ctde"},
        "hypermarl_id": {"uses_hypernet": True, "execution_scope": "ctde"},
        "hypermarl_fullnet": {"uses_hypernet": True, "execution_scope": "ctde"},
        "dynamic_route": {"uses_hypernet": True, "execution_scope": "ctde"},
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

        if self.model_type in {"dynamic_route", "graph_route"}:
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

        if self.MODEL_SPECS[self.model_type]["uses_hypernet"] and self.model_type != "hypermarl_fullnet":
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
            self.fixed_head = nn.Linear(self.hidden_dim, self.n_actions) if self.model_type == "qmix_minimal" else None

        self.latest_condition = None
        self.latest_route_logits = None
        self.latest_route_indices = None
        self.latest_graph_adj = None
        self.latest_graph_nodes = None

    def init_hidden(self):
        return self.fc1.weight.new_zeros(self.hidden_dim)

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

    def forward(self, inputs, hidden_state, context=None, test_mode=False):
        batch_size, n_agents, _ = inputs.shape
        flat_inputs = inputs.reshape(batch_size * n_agents, -1)
        x = F.relu(self.fc1(flat_inputs), inplace=True)

        if hidden_state is None:
            hidden_state = self.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)
        flat_hidden = hidden_state.reshape(batch_size * n_agents, -1)
        hidden = self.rnn(x, flat_hidden).view(batch_size, n_agents, self.hidden_dim)

        if self.model_type == "qmix_minimal":
            q = self.fixed_head(hidden)
        else:
            if context is None:
                raise ValueError("{} requires context with obs/prev_action.".format(self.model_type))
            condition = self._build_condition(hidden, context, test_mode=test_mode)
            if self.model_type == "hypermarl_fullnet":
                q = self._apply_full_hypermarl_head(hidden, condition)
            else:
                q = self._apply_dynamic_head(hidden, condition)

        return q, hidden
