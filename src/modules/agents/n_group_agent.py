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
        self.no_group_param_scope_modes = {
            "graph_input_fusion_node_embed_struct_feat_two_layer_head",
            "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
            "graph_input_fusion_node_embed_struct_feat_full_head",
            "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
            "graph_input_fusion_node_embed_struct_feat_full_head_residual",
        }
        self.full_head_local_ctde_modes = {
            "graph_input_fusion_node_embed_struct_feat_full_head",
            "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
            "graph_input_fusion_node_embed_struct_feat_full_head_residual",
        }
        self.no_group_decoupled_modes = {
            "graph_input_fusion_node_embed_struct_feat_decoupled_head",
            "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
        }
        self.no_group_gcn_modes = {
            "graph_input_fusion_node_embed_gcn_full_head",
        }
        self.no_group_subgraph_modes = {
            "graph_input_fusion_node_embed_subgraph_full_head",
        }
        self.no_group_full_model_modes = {
            "graph_input_fusion_node_embed_struct_feat_full_model",
        }
        self.grouped_full_head_modes = {
            "graph_input_fusion_node_embed_group_full_head",
        }
        self.group_num = getattr(args, "group_num", 3)
        if (
            self.group_head_mode == "graph_input_fusion_node_embed_struct_only"
            or self.group_head_mode in self.no_group_head_compare_modes
            or self.group_head_mode in self.no_group_param_scope_modes
            or self.group_head_mode in self.no_group_decoupled_modes
            or self.group_head_mode in self.no_group_gcn_modes
            or self.group_head_mode in self.no_group_subgraph_modes
            or self.group_head_mode in self.no_group_full_model_modes
        ):
            self.group_num = 1
        self.group_assignment_tau = getattr(args, "group_assignment_tau", 1.0)
        self.base_struct_stat_dim = 10
        self.struct_stat_dim = self.base_struct_stat_dim
        self.group_direct_topk = max(1, min(args.n_agents - 1, getattr(args, "group_direct_topk", 3)))
        self.group_ema_alpha = getattr(args, "group_ema_alpha", 0.8)
        self.graph_attention_tau = getattr(args, "graph_attention_tau", 0.5)
        self.full_head_local_ctde = (
            getattr(args, "full_head_local_ctde", False) and self.group_head_mode in self.full_head_local_ctde_modes
        )
        self.full_head_variant = getattr(args, "full_head_variant", "dynamic").replace("-", "_")
        legacy_variant_map = {
            "episode_mean": "ema_ep_struct_mean",
            "episode_mean_param_avg": "ema_ep_struct_mean",
            "gradient_separate": "grad_decouple",
            "grad_separate": "grad_decouple",
            "rf_strategy": "rf",
            "with_id": "id_cond",
            "tgn": "temporal_gnn",
            "temporal": "temporal_gnn",
            "edge": "edge_gnn",
            "edge_based": "edge_gnn",
            "relation": "relation_gnn",
            "relation_based": "relation_gnn",
            "hetero": "hetero_enemy",
            "hetero_graph": "hetero_enemy",
            "heterogeneous": "hetero_enemy",
        }
        self.full_head_variant = legacy_variant_map.get(self.full_head_variant, self.full_head_variant)
        self.full_head_param_ema_beta = getattr(args, "full_head_param_ema_beta", 0.99)
        self.full_head_use_ema_in_test = getattr(args, "full_head_use_ema_in_test", True)
        self.rf_fan_mode = getattr(args, "full_head_rf_fan_mode", "fan_in")
        self.full_head_relation_num = max(2, int(getattr(args, "full_head_relation_num", 4)))
        hetero_enemy_nodes = int(getattr(args, "full_head_hetero_enemy_nodes", 0))
        if hetero_enemy_nodes <= 0:
            inferred = args.n_actions - 6 if args.n_actions > 6 else args.n_agents
            hetero_enemy_nodes = max(1, min(args.n_agents, inferred))
        self.full_head_hetero_enemy_nodes = hetero_enemy_nodes
        self.register_buffer("agent_id_onehot", th.eye(args.n_agents))
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
                "graph_input_fusion_node_embed_no_reg",
                "graph_input_fusion_node_embed_no_groupemb_no_reg",
                "graph_input_fusion_node_embed_struct_only",
                "graph_input_fusion_node_embed_sharp",
                "graph_input_fusion_node_embed_threshold_group",
                "graph_input_fusion_hidden_head",
                "graph_input_fusion_node_embed_head",
                "graph_input_fusion_struct_feat_head",
                "graph_input_fusion_node_embed_struct_feat_head",
                "graph_input_fusion_node_embed_struct_feat_two_layer_head",
                "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
                "graph_input_fusion_node_embed_struct_feat_full_head",
                "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                "graph_input_fusion_node_embed_struct_feat_full_head_residual",
                "graph_input_fusion_node_embed_struct_feat_decoupled_head",
                "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
                "graph_input_fusion_node_embed_gcn_full_head",
                "graph_input_fusion_node_embed_subgraph_full_head",
                "graph_input_fusion_node_embed_struct_feat_full_model",
                "graph_input_fusion_node_embed_group_full_head",
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
                    "graph_input_fusion_node_embed_no_reg",
                    "graph_input_fusion_node_embed_no_groupemb_no_reg",
                    "graph_input_fusion_node_embed_struct_only",
                    "graph_input_fusion_node_embed_sharp",
                    "graph_input_fusion_node_embed_threshold_group",
                    "graph_input_fusion_hidden_head",
                    "graph_input_fusion_node_embed_head",
                    "graph_input_fusion_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_two_layer_head",
                    "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
                    "graph_input_fusion_node_embed_struct_feat_full_head",
                    "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                    "graph_input_fusion_node_embed_struct_feat_full_head_residual",
                    "graph_input_fusion_node_embed_struct_feat_decoupled_head",
                    "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
                    "graph_input_fusion_node_embed_gcn_full_head",
                    "graph_input_fusion_node_embed_subgraph_full_head",
                    "graph_input_fusion_node_embed_struct_feat_full_model",
                    "graph_input_fusion_node_embed_group_full_head",
                    "graph_input_fusion_fixed_group",
                    "graph_input_fusion_group_only",
                    "graph_input_fusion_head_only",
                ]:
                    self.struct_stat_dim = args.n_agents + 2
                    if self.group_head_mode in [
                        "graph_input_fusion_node_embed",
                        "graph_input_fusion_node_embed_no_groupemb",
                        "graph_input_fusion_node_embed_no_reg",
                        "graph_input_fusion_node_embed_no_groupemb_no_reg",
                        "graph_input_fusion_node_embed_struct_only",
                        "graph_input_fusion_node_embed_sharp",
                        "graph_input_fusion_node_embed_threshold_group",
                        "graph_input_fusion_node_embed_struct_feat_full_model",
                        "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                        "graph_input_fusion_node_embed_group_full_head",
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
                    "graph_input_fusion_node_embed_no_reg",
                    "graph_input_fusion_node_embed_no_groupemb_no_reg",
                    "graph_input_fusion_node_embed_struct_only",
                    "graph_input_fusion_node_embed_sharp",
                    "graph_input_fusion_node_embed_threshold_group",
                    "graph_input_fusion_hidden_head",
                    "graph_input_fusion_node_embed_head",
                    "graph_input_fusion_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_two_layer_head",
                    "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
                    "graph_input_fusion_node_embed_struct_feat_full_head",
                    "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                    "graph_input_fusion_node_embed_struct_feat_full_head_residual",
                    "graph_input_fusion_node_embed_struct_feat_decoupled_head",
                    "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
                    "graph_input_fusion_node_embed_gcn_full_head",
                    "graph_input_fusion_node_embed_subgraph_full_head",
                    "graph_input_fusion_node_embed_struct_feat_full_model",
                    "graph_input_fusion_node_embed_group_full_head",
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
                elif self.group_head_mode in self.no_group_param_scope_modes:
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim + args.hypernet_embed, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                    if self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_two_layer_head":
                        self.static_pre_head = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                            nn.ReLU(inplace=True),
                        )
                    elif self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_bottleneck_head":
                        self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                        self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                        self.static_out_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)
                    elif self.group_head_mode in {
                        "graph_input_fusion_node_embed_struct_feat_full_head",
                        "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                        "graph_input_fusion_node_embed_struct_feat_full_head_residual",
                    }:
                        self.distill_student_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                        self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                        self.struct_only_head_encoder = nn.Sequential(
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.obs_action_exec_encoder = nn.Sequential(
                            nn.Linear(self.graph_obs_dim + args.n_actions, args.rnn_hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                            nn.Tanh(),
                        )
                        self.id_cond_head_encoder = nn.Sequential(
                            nn.Linear(args.hypernet_embed + args.n_agents, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.full_head_gcn_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * 2, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.full_head_gat_q = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_gat_k = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_gat_v = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_gat_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * 2, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.full_head_tgn_msg_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                            nn.Tanh(),
                        )
                        self.full_head_tgn_gru = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
                        self.full_head_tgn_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * 2, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        edge_pair_in_dim = args.rnn_hidden_dim * 4 + 2
                        self.full_head_edge_pair_encoder = nn.Sequential(
                            nn.Linear(edge_pair_in_dim, args.rnn_hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                            nn.Tanh(),
                        )
                        self.full_head_edge_score = nn.Linear(args.rnn_hidden_dim, 1)
                        self.full_head_edge_struct_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * 2 + 2, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )

                        self.full_head_relation_pair_encoder = nn.Sequential(
                            nn.Linear(edge_pair_in_dim, args.rnn_hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                            nn.Tanh(),
                        )
                        self.full_head_relation_logits = nn.Linear(args.rnn_hidden_dim, self.full_head_relation_num)
                        self.full_head_relation_struct_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * (1 + self.full_head_relation_num), args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )

                        self.full_head_hetero_enemy_seed = nn.Sequential(
                            nn.Linear(self.graph_obs_dim, args.rnn_hidden_dim * self.full_head_hetero_enemy_nodes),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.rnn_hidden_dim * self.full_head_hetero_enemy_nodes, args.rnn_hidden_dim * self.full_head_hetero_enemy_nodes),
                        )
                        self.full_head_hetero_a_q = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_hetero_a_k = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_hetero_a_v = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_hetero_e_q = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_hetero_e_k = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_hetero_e_v = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
                        self.full_head_hetero_struct_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim * 4, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        if self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_full_head_residual":
                            self.static_out_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)
                            self.dynamic_residual_scale = getattr(args, "dynamic_residual_scale", 0.3)
                elif self.group_head_mode in self.grouped_full_head_modes:
                    self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                elif self.group_head_mode in self.no_group_decoupled_modes:
                    if self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head":
                        self.head_input_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim + args.hypernet_embed, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.node_head_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.graph_head_encoder = nn.Sequential(
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                    else:
                        self.node_head_encoder = nn.Sequential(
                            nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                        self.graph_head_encoder = nn.Sequential(
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, args.hypernet_embed),
                            nn.Tanh(),
                        )
                    self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                elif self.group_head_mode in self.no_group_gcn_modes:
                    self.gcn_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim * 2, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim + args.hypernet_embed, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                    self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                elif self.group_head_mode in self.no_group_subgraph_modes:
                    subgraph_dim = (self.group_direct_topk + 1) * (self.group_direct_topk + 1) + 2
                    self.subgraph_encoder = nn.Sequential(
                        nn.Linear(subgraph_dim, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim + args.hypernet_embed, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                    self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                elif self.group_head_mode in self.no_group_full_model_modes:
                    self.head_input_encoder = nn.Sequential(
                        nn.Linear(args.rnn_hidden_dim + args.hypernet_embed, args.hypernet_embed),
                        nn.ReLU(inplace=True),
                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                        nn.Tanh(),
                    )
                    self.hyper_fc1_w = nn.Linear(args.hypernet_embed, input_shape * args.rnn_hidden_dim)
                    self.hyper_fc1_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                    self.hyper_gru_ih_w = nn.Linear(args.hypernet_embed, 3 * args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_gru_ih_b = nn.Linear(args.hypernet_embed, 3 * args.rnn_hidden_dim)
                    self.hyper_gru_hh_w = nn.Linear(args.hypernet_embed, 3 * args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_gru_hh_b = nn.Linear(args.hypernet_embed, 3 * args.rnn_hidden_dim)
                    self.hyper_bottleneck_w = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.rnn_hidden_dim)
                    self.hyper_bottleneck_b = nn.Linear(args.hypernet_embed, args.rnn_hidden_dim)
                if self.full_head_local_ctde:
                    self.struct_stat_dim = args.rnn_hidden_dim
                self.struct_encoder = nn.Sequential(
                    nn.Linear(self.struct_stat_dim, args.hypernet_embed),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.hypernet_embed, args.hypernet_embed),
                    nn.Tanh(),
                )
                if (
                    not use_proto_assignment
                    and self.group_head_mode != "graph_input_fusion_node_embed_struct_only"
                    and self.group_head_mode not in self.no_group_head_compare_modes
                    and self.group_head_mode not in self.no_group_param_scope_modes
                    and self.group_head_mode not in self.no_group_decoupled_modes
                    and self.group_head_mode not in self.no_group_gcn_modes
                    and self.group_head_mode not in self.no_group_subgraph_modes
                    and self.group_head_mode not in self.no_group_full_model_modes
                ):
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
            if self.full_head_variant == "rf" and (
                self.group_head_mode in self.no_group_param_scope_modes
                or self.group_head_mode in self.grouped_full_head_modes
            ):
                self._reset_full_head_rf_initialization()

        self.group_probs = None
        self.group_graphs = None
        self.current_groups = None
        self.group_struct_features = None
        self.group_role_prototypes = None
        self.group_node_embeddings = None
        self.temporal_graph_state = None
        self.distill_teacher_q = None
        self.distill_student_q = None
        self.episode_head_feat_sum = None
        self.episode_head_feat_count = 0
        self.episode_param_sum_wb = None
        self.episode_param_sum_bb = None
        self.episode_param_sum_wo = None
        self.episode_param_sum_bo = None
        self.episode_param_count = 0
        self.episode_struct_sum = None
        self.episode_struct_count = 0
        self.register_buffer(
            "head_param_ema_wb",
            th.zeros(args.n_agents, args.rnn_hidden_dim, args.rnn_hidden_dim),
        )
        self.register_buffer(
            "head_param_ema_bb",
            th.zeros(args.n_agents, 1, args.rnn_hidden_dim),
        )
        self.register_buffer(
            "head_param_ema_wo",
            th.zeros(args.n_agents, args.rnn_hidden_dim, args.n_actions),
        )
        self.register_buffer(
            "head_param_ema_bo",
            th.zeros(args.n_agents, 1, args.n_actions),
        )
        self.register_buffer("head_param_ema_initialized", th.zeros(1))

    def init_hidden(self):
        self._flush_episode_ema_update()
        self.cached_group_probs = None
        self.cached_group_graphs = None
        self.cached_struct_features = None
        self.temporal_graph_state = None
        self.distill_teacher_q = None
        self.distill_student_q = None
        self.episode_head_feat_sum = None
        self.episode_head_feat_count = 0
        self.episode_param_sum_wb = None
        self.episode_param_sum_bb = None
        self.episode_param_sum_wo = None
        self.episode_param_sum_bo = None
        self.episode_param_count = 0
        self.episode_struct_sum = None
        self.episode_struct_count = 0
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
            "graph_input_fusion_node_embed_no_reg",
            "graph_input_fusion_node_embed_no_groupemb_no_reg",
            "graph_input_fusion_node_embed_struct_only",
            "graph_input_fusion_node_embed_sharp",
            "graph_input_fusion_node_embed_threshold_group",
            "graph_input_fusion_hidden_head",
            "graph_input_fusion_node_embed_head",
            "graph_input_fusion_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_two_layer_head",
            "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
            "graph_input_fusion_node_embed_struct_feat_full_head",
            "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
            "graph_input_fusion_node_embed_struct_feat_full_head_residual",
            "graph_input_fusion_node_embed_struct_feat_decoupled_head",
            "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
            "graph_input_fusion_node_embed_struct_feat_full_model",
            "graph_input_fusion_node_embed_group_full_head",
            "graph_input_fusion_fixed_group",
            "graph_input_fusion_group_only",
            "graph_input_fusion_head_only",
        ]:
            struct_input = th.cat([row_probs, degree, entropy], dim=-1)
            if self.group_head_mode in [
                "graph_input_fusion_node_embed",
                "graph_input_fusion_node_embed_no_groupemb",
                "graph_input_fusion_node_embed_no_reg",
                "graph_input_fusion_node_embed_no_groupemb_no_reg",
                "graph_input_fusion_node_embed_struct_only",
                "graph_input_fusion_node_embed_sharp",
                "graph_input_fusion_node_embed_threshold_group",
                "graph_input_fusion_node_embed_struct_feat_full_model",
                "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                "graph_input_fusion_node_embed_group_full_head",
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

    def _build_graph_input_fusion_no_group_head(self, h, graph_context, test_mode=False):
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

        if self.full_head_variant == "distill" and test_mode:
            student_head_feat = self.distill_student_encoder(node_embed.reshape(b * a, -1)).view(b, a, -1)
            return group_probs, zero_struct, zero_graph, student_head_feat, node_embed

        if self.full_head_local_ctde:
            struct_feat = self.struct_encoder(node_embed.reshape(b * a, -1)).view(b, a, -1)
            fused = th.cat([node_embed, struct_feat], dim=-1)
            head_feat = self.head_input_encoder(fused.reshape(b * a, -1)).view(b, a, -1)
            return group_probs, struct_feat, zero_graph, head_feat, node_embed

        group_graphs = self._build_attention_graph(h, graph_source=node_embed)
        struct_feat = self._build_graph_struct_features(group_graphs, node_embed=node_embed)
        if (
            self.group_head_mode in self.full_head_local_ctde_modes
            and self.full_head_variant in {"gcn", "gat", "temporal_gnn", "edge_gnn", "relation_gnn", "hetero_enemy"}
        ):
            variant_struct = self._build_full_head_graph_variant_struct(node_embed, group_graphs, graph_context=graph_context)
            if variant_struct is not None:
                struct_feat = variant_struct
        if self.group_head_mode == "graph_input_fusion_struct_feat_head":
            head_feat = struct_feat
        elif self.group_head_mode in self.no_group_param_scope_modes or self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_head":
            fused = th.cat([node_embed, struct_feat], dim=-1)
            head_feat = self.head_input_encoder(fused.reshape(b * a, -1)).view(b, a, -1)
        else:
            head_feat = struct_feat
        return group_probs, struct_feat, group_graphs, head_feat, node_embed

    def _build_graph_input_fusion_no_group_decoupled(self, h, graph_context):
        b, a, _ = h.size()
        group_probs = h.new_ones(b, a, 1)
        node_embed = self._build_graph_input_fusion_source(h, graph_context)
        group_graphs = self._build_attention_graph(h, graph_source=node_embed)
        struct_feat = self._build_graph_struct_features(group_graphs)
        node_state = self.node_head_encoder(node_embed.reshape(b * a, -1)).view(b, a, -1)
        graph_state = self.graph_head_encoder(struct_feat.reshape(b * a, -1)).view(b, a, -1)
        if self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head":
            fused = th.cat([node_embed, struct_feat], dim=-1)
            shared_state = self.head_input_encoder(fused.reshape(b * a, -1)).view(b, a, -1)
            node_state = shared_state + node_state
            graph_state = shared_state + graph_state
        return group_probs, struct_feat, group_graphs, node_state, graph_state, node_embed

    def _build_graph_input_fusion_no_group_gcn(self, h, graph_context):
        b, a, _ = h.size()
        group_probs = h.new_ones(b, a, 1)
        node_embed = self._build_graph_input_fusion_source(h, graph_context)
        group_graphs = self._build_attention_graph(h, graph_source=node_embed)
        degree = group_graphs.sum(dim=-1, keepdim=True)
        row_probs = group_graphs / degree.clamp(min=1e-8)
        neighbor_embed = th.matmul(row_probs, node_embed)
        gcn_input = th.cat([node_embed, neighbor_embed], dim=-1)
        gcn_embed = self.gcn_encoder(gcn_input.reshape(b * a, -1)).view(b, a, -1)
        head_input = th.cat([node_embed, gcn_embed], dim=-1)
        head_feat = self.head_input_encoder(head_input.reshape(b * a, -1)).view(b, a, -1)
        return group_probs, gcn_embed, group_graphs, head_feat, node_embed

    def _build_graph_input_fusion_no_group_subgraph(self, h, graph_context):
        b, a, _ = h.size()
        group_probs = h.new_ones(b, a, 1)
        node_embed = self._build_graph_input_fusion_source(h, graph_context)
        group_graphs = self._build_attention_graph(h, graph_source=node_embed)
        degree = group_graphs.sum(dim=-1, keepdim=True)
        row_probs = group_graphs / degree.clamp(min=1e-8)
        entropy = -(row_probs.clamp(min=1e-8) * row_probs.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)
        subgraph_input = self._build_ego_subgraph_input(group_graphs, degree, entropy)
        subgraph_feat = self.subgraph_encoder(subgraph_input.reshape(b * a, -1)).view(b, a, -1)
        head_input = th.cat([node_embed, subgraph_feat], dim=-1)
        head_feat = self.head_input_encoder(head_input.reshape(b * a, -1)).view(b, a, -1)
        return group_probs, subgraph_feat, group_graphs, head_feat, node_embed

    def _build_obs_action_exec_input(self, h, graph_context):
        b, a, _ = h.size()
        if graph_context is None:
            obs = h.new_zeros(b, a, self.graph_obs_dim)
            prev_action = h.new_zeros(b, a, self.action_dim)
        else:
            obs = graph_context["obs"]
            prev_action = graph_context["prev_action"]
        obs_action = th.cat([obs, prev_action], dim=-1)
        return self.obs_action_exec_encoder(obs_action.reshape(b * a, -1)).view(b, a, -1)

    def _build_full_head_graph_variant_struct(self, node_embed, group_graphs, graph_context=None):
        b, a, _ = node_embed.size()
        eye = th.eye(a, device=node_embed.device, dtype=th.bool).unsqueeze(0)
        degree = group_graphs.sum(dim=-1, keepdim=True)
        row_probs = group_graphs / degree.clamp(min=1e-8)

        if self.full_head_variant == "gcn":
            neighbor_embed = th.matmul(row_probs, node_embed)
            gcn_input = th.cat([node_embed, neighbor_embed], dim=-1)
            return self.full_head_gcn_encoder(gcn_input.reshape(b * a, -1)).view(b, a, -1)

        if self.full_head_variant == "gat":
            q = self.full_head_gat_q(node_embed)
            k = self.full_head_gat_k(node_embed)
            v = self.full_head_gat_v(node_embed)
            scores = th.matmul(q, k.transpose(1, 2)) / math.sqrt(self.a_h_dim)
            scores = scores.masked_fill(eye, -1e9)
            attn = th.softmax(scores, dim=-1)
            attn = 0.5 * (attn + attn.transpose(1, 2))
            attn = attn.masked_fill(eye, 0.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            neighbor_embed = th.matmul(attn, v)
            gat_input = th.cat([node_embed, neighbor_embed], dim=-1)
            return self.full_head_gat_encoder(gat_input.reshape(b * a, -1)).view(b, a, -1)

        if self.full_head_variant == "temporal_gnn":
            neighbor_embed = th.matmul(row_probs, node_embed)
            msg = self.full_head_tgn_msg_encoder(th.cat([node_embed, neighbor_embed], dim=-1).reshape(b * a, -1))
            if self.temporal_graph_state is None or self.temporal_graph_state.shape != node_embed.shape:
                prev_state = msg.new_zeros(b * a, self.a_h_dim)
            else:
                prev_state = self.temporal_graph_state.reshape(b * a, self.a_h_dim)
            temporal_state = self.full_head_tgn_gru(msg, prev_state).view(b, a, self.a_h_dim)
            self.temporal_graph_state = temporal_state.detach()
            temporal_input = th.cat([node_embed, temporal_state], dim=-1)
            return self.full_head_tgn_encoder(temporal_input.reshape(b * a, -1)).view(b, a, -1)

        if graph_context is None:
            prev_action = node_embed.new_zeros(b, a, self.action_dim)
            obs = node_embed.new_zeros(b, a, self.graph_obs_dim)
        else:
            prev_action = graph_context["prev_action"]
            obs = graph_context["obs"]

        act_sim = th.matmul(prev_action, prev_action.transpose(1, 2)).unsqueeze(-1)
        ni = node_embed.unsqueeze(2).expand(-1, -1, a, -1)
        nj = node_embed.unsqueeze(1).expand(-1, a, -1, -1)
        pair_feat = th.cat(
            [
                ni,
                nj,
                (ni - nj).abs(),
                ni * nj,
                group_graphs.unsqueeze(-1),
                act_sim,
            ],
            dim=-1,
        )

        if self.full_head_variant == "edge_gnn":
            pair_hidden = self.full_head_edge_pair_encoder(pair_feat.reshape(b * a * a, -1)).view(b, a, a, -1)
            edge_scores = self.full_head_edge_score(pair_hidden).squeeze(-1)
            edge_scores = edge_scores.masked_fill(eye, -1e9)
            edge_attn = th.softmax(edge_scores, dim=-1)
            edge_attn = 0.5 * (edge_attn + edge_attn.transpose(1, 2))
            edge_attn = edge_attn.masked_fill(eye, 0.0)
            edge_attn = edge_attn / edge_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            edge_msg = th.matmul(edge_attn, node_embed)
            edge_degree = edge_attn.sum(dim=-1, keepdim=True)
            edge_row = edge_attn / edge_degree.clamp(min=1e-8)
            edge_entropy = -(edge_row.clamp(min=1e-8) * edge_row.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)
            edge_struct_in = th.cat([node_embed, edge_msg, edge_degree, edge_entropy], dim=-1)
            return self.full_head_edge_struct_encoder(edge_struct_in.reshape(b * a, -1)).view(b, a, -1)

        if self.full_head_variant == "relation_gnn":
            pair_hidden = self.full_head_relation_pair_encoder(pair_feat.reshape(b * a * a, -1)).view(b, a, a, -1)
            rel_logits = self.full_head_relation_logits(pair_hidden)
            rel_logits = rel_logits.masked_fill(eye.unsqueeze(-1), -1e9)
            rel_probs = th.softmax(rel_logits, dim=-1)  # [B, A, A, R]
            rel_adj = rel_probs.permute(0, 3, 1, 2)  # [B, R, A, A]
            rel_adj = rel_adj.masked_fill(eye.unsqueeze(1), 0.0)
            rel_adj = rel_adj / rel_adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            node_expand = node_embed.unsqueeze(1).expand(-1, self.full_head_relation_num, -1, -1)
            rel_msg = th.matmul(rel_adj, node_expand)  # [B, R, A, H]
            rel_msg = rel_msg.permute(0, 2, 1, 3).reshape(b, a, self.full_head_relation_num * self.a_h_dim)
            rel_struct_in = th.cat([node_embed, rel_msg], dim=-1)
            return self.full_head_relation_struct_encoder(rel_struct_in.reshape(b * a, -1)).view(b, a, -1)

        if self.full_head_variant == "hetero_enemy":
            enemy_seed = self.full_head_hetero_enemy_seed(obs.reshape(b * a, -1)).view(
                b, a, self.full_head_hetero_enemy_nodes, self.a_h_dim
            )
            enemy_nodes = enemy_seed.mean(dim=1)  # [B, E, H]

            ally_self = th.matmul(row_probs, node_embed)

            q_a = self.full_head_hetero_a_q(node_embed)
            k_e = self.full_head_hetero_e_k(enemy_nodes)
            v_e = self.full_head_hetero_e_v(enemy_nodes)
            score_ae = th.matmul(q_a, k_e.transpose(1, 2)) / math.sqrt(self.a_h_dim)
            attn_ae = th.softmax(score_ae, dim=-1)
            ally_from_enemy = th.matmul(attn_ae, v_e)

            q_e = self.full_head_hetero_e_q(enemy_nodes)
            k_a = self.full_head_hetero_a_k(node_embed)
            v_a = self.full_head_hetero_a_v(node_embed)
            score_ea = th.matmul(q_e, k_a.transpose(1, 2)) / math.sqrt(self.a_h_dim)
            attn_ea = th.softmax(score_ea, dim=-1)
            enemy_from_ally = th.matmul(attn_ea, v_a)
            enemy_global = enemy_from_ally.mean(dim=1, keepdim=True).expand(-1, a, -1)

            hetero_struct_in = th.cat([node_embed, ally_self, ally_from_enemy, enemy_global], dim=-1)
            return self.full_head_hetero_struct_encoder(hetero_struct_in.reshape(b * a, -1)).view(b, a, -1)

        return None

    def _reset_linear_fan_init(self, linear):
        if not isinstance(linear, nn.Linear):
            return
        mode = str(self.rf_fan_mode).lower()
        with th.no_grad():
            if mode in {"fan_out", "out"}:
                nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), mode="fan_out", nonlinearity="linear")
            elif mode in {"fan_in_out", "fan_avg", "avg"}:
                nn.init.xavier_uniform_(linear.weight)
            else:
                nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="linear")
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def _reset_full_head_rf_initialization(self):
        if hasattr(self, "hyper_bottleneck_w"):
            self._reset_linear_fan_init(self.hyper_bottleneck_w)
        if hasattr(self, "hyper_bottleneck_b"):
            self._reset_linear_fan_init(self.hyper_bottleneck_b)
        if hasattr(self, "hyper_w") and isinstance(self.hyper_w, nn.Sequential) and len(self.hyper_w) > 0:
            self._reset_linear_fan_init(self.hyper_w[0])
        if hasattr(self, "hyper_b") and isinstance(self.hyper_b, nn.Sequential) and len(self.hyper_b) > 0:
            self._reset_linear_fan_init(self.hyper_b[0])

    def _update_episode_head_feat_mean(self, head_feat):
        if self.episode_head_feat_sum is None or self.episode_head_feat_sum.shape != head_feat.shape:
            self.episode_head_feat_sum = th.zeros_like(head_feat)
            self.episode_head_feat_count = 0
        cur_count = self.episode_head_feat_count + 1
        mean_feat = (self.episode_head_feat_sum + head_feat) / float(cur_count)
        self.episode_head_feat_sum = self.episode_head_feat_sum + head_feat.detach()
        self.episode_head_feat_count = cur_count
        return mean_feat

    def _build_dynamic_full_head_q(self, h, group_state, head_input=None):
        b, a, _ = h.size()
        head_source = h if head_input is None else head_input
        state_flat = group_state.reshape(b * a, -1)
        bottleneck_w = self.hyper_bottleneck_w(state_flat).reshape(b * a, self.a_h_dim, self.a_h_dim)
        bottleneck_b = self.hyper_bottleneck_b(state_flat).reshape(b * a, 1, self.a_h_dim)
        bottleneck = th.matmul(head_source.reshape(b * a, 1, self.a_h_dim), bottleneck_w) + bottleneck_b
        bottleneck = F.relu(bottleneck, inplace=True)
        fc2_w = self.hyper_w(state_flat).reshape(b * a, self.a_h_dim, self.action_dim)
        fc2_b = self.hyper_b(state_flat).reshape(b * a, 1, self.action_dim)
        q = th.matmul(bottleneck, fc2_w) + fc2_b
        return q.view(b, a, -1), bottleneck_w, bottleneck_b, fc2_w, fc2_b

    def _update_head_param_ema_from_agent_params(self, wb_agent, bb_agent, wo_agent, bo_agent):
        if self.head_param_ema_initialized.item() < 0.5:
            self.head_param_ema_wb.copy_(wb_agent)
            self.head_param_ema_bb.copy_(bb_agent)
            self.head_param_ema_wo.copy_(wo_agent)
            self.head_param_ema_bo.copy_(bo_agent)
            self.head_param_ema_initialized.fill_(1.0)
            return
        beta = self.full_head_param_ema_beta
        self.head_param_ema_wb.mul_(beta).add_(wb_agent, alpha=1.0 - beta)
        self.head_param_ema_bb.mul_(beta).add_(bb_agent, alpha=1.0 - beta)
        self.head_param_ema_wo.mul_(beta).add_(wo_agent, alpha=1.0 - beta)
        self.head_param_ema_bo.mul_(beta).add_(bo_agent, alpha=1.0 - beta)

    def _apply_full_head_with_fixed_params(self, h, wb, bb, wo, bo):
        b, a, _ = h.size()
        wb_use = wb.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * a, self.a_h_dim, self.a_h_dim)
        bb_use = bb.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * a, 1, self.a_h_dim)
        wo_use = wo.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * a, self.a_h_dim, self.action_dim)
        bo_use = bo.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * a, 1, self.action_dim)
        bottleneck = th.matmul(h.reshape(b * a, 1, self.a_h_dim), wb_use) + bb_use
        bottleneck = F.relu(bottleneck, inplace=True)
        q = th.matmul(bottleneck, wo_use) + bo_use
        return q.view(b, a, -1)

    def _update_head_param_ema(self, bottleneck_w, bottleneck_b, fc2_w, fc2_b):
        b = bottleneck_w.size(0)
        a = self.args.n_agents
        wb_mean = bottleneck_w.detach().reshape(b // a, a, self.a_h_dim, self.a_h_dim).mean(dim=0)
        bb_mean = bottleneck_b.detach().reshape(b // a, a, 1, self.a_h_dim).mean(dim=0)
        wo_mean = fc2_w.detach().reshape(b // a, a, self.a_h_dim, self.action_dim).mean(dim=0)
        bo_mean = fc2_b.detach().reshape(b // a, a, 1, self.action_dim).mean(dim=0)
        self._update_head_param_ema_from_agent_params(wb_mean, bb_mean, wo_mean, bo_mean)

    def _accumulate_episode_param_mean(self, bottleneck_w, bottleneck_b, fc2_w, fc2_b):
        b = bottleneck_w.size(0)
        a = self.args.n_agents
        wb_mean = bottleneck_w.detach().reshape(b // a, a, self.a_h_dim, self.a_h_dim).mean(dim=0)
        bb_mean = bottleneck_b.detach().reshape(b // a, a, 1, self.a_h_dim).mean(dim=0)
        wo_mean = fc2_w.detach().reshape(b // a, a, self.a_h_dim, self.action_dim).mean(dim=0)
        bo_mean = fc2_b.detach().reshape(b // a, a, 1, self.action_dim).mean(dim=0)
        if self.episode_param_sum_wb is None:
            self.episode_param_sum_wb = th.zeros_like(wb_mean)
            self.episode_param_sum_bb = th.zeros_like(bb_mean)
            self.episode_param_sum_wo = th.zeros_like(wo_mean)
            self.episode_param_sum_bo = th.zeros_like(bo_mean)
            self.episode_param_count = 0
        self.episode_param_sum_wb = self.episode_param_sum_wb + wb_mean
        self.episode_param_sum_bb = self.episode_param_sum_bb + bb_mean
        self.episode_param_sum_wo = self.episode_param_sum_wo + wo_mean
        self.episode_param_sum_bo = self.episode_param_sum_bo + bo_mean
        self.episode_param_count += 1

    def _accumulate_episode_struct_mean(self, struct_feat):
        struct_mean = struct_feat.detach().mean(dim=0)
        if self.episode_struct_sum is None:
            self.episode_struct_sum = th.zeros_like(struct_mean)
            self.episode_struct_count = 0
        self.episode_struct_sum = self.episode_struct_sum + struct_mean
        self.episode_struct_count += 1

    def _flush_episode_ema_update(self):
        if self.full_head_variant == "ema_ep_param_mean" and self.episode_param_count > 0:
            count = float(self.episode_param_count)
            wb_mean = self.episode_param_sum_wb / count
            bb_mean = self.episode_param_sum_bb / count
            wo_mean = self.episode_param_sum_wo / count
            bo_mean = self.episode_param_sum_bo / count
            self._update_head_param_ema_from_agent_params(wb_mean, bb_mean, wo_mean, bo_mean)
        elif self.full_head_variant == "ema_ep_struct_mean" and self.episode_struct_count > 0:
            struct_mean = self.episode_struct_sum / float(self.episode_struct_count)
            group_state = self.group_decoder(struct_mean)
            wb_mean = self.hyper_bottleneck_w(group_state).reshape(self.args.n_agents, self.a_h_dim, self.a_h_dim)
            bb_mean = self.hyper_bottleneck_b(group_state).reshape(self.args.n_agents, 1, self.a_h_dim)
            wo_mean = self.hyper_w(group_state).reshape(self.args.n_agents, self.a_h_dim, self.action_dim)
            bo_mean = self.hyper_b(group_state).reshape(self.args.n_agents, 1, self.action_dim)
            self._update_head_param_ema_from_agent_params(
                wb_mean.detach(), bb_mean.detach(), wo_mean.detach(), bo_mean.detach()
            )

    def _apply_group_dynamic_full_head(self, h, group_state):
        q, _, _, _, _ = self._build_dynamic_full_head_q(h, group_state)
        return q

    def _apply_no_group_dynamic_head(self, h, head_feat, struct_feat=None, node_embed=None, graph_context=None, test_mode=False):
        b, a, _ = h.size()
        self.distill_teacher_q = None
        self.distill_student_q = None
        group_state = self.group_decoder(head_feat.reshape(b * a, -1)).view(b, a, -1)

        if self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_two_layer_head":
            h_pre = self.static_pre_head(h.reshape(b * a, -1)).view(b, a, -1)
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h_pre.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
        elif self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_bottleneck_head":
            bottleneck_w = self.hyper_bottleneck_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.a_h_dim)
            bottleneck_b = self.hyper_bottleneck_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.a_h_dim)
            bottleneck = th.matmul(h.reshape(b * a, 1, self.a_h_dim), bottleneck_w) + bottleneck_b
            bottleneck = F.relu(bottleneck, inplace=True).reshape(b * a, self.a_h_dim)
            q = self.static_out_head(bottleneck).view(b * a, 1, self.action_dim)
        elif self.group_head_mode in {
            "graph_input_fusion_node_embed_struct_feat_full_head",
            "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
            "graph_input_fusion_node_embed_struct_feat_full_head_residual",
            "graph_input_fusion_node_embed_gcn_full_head",
        }:
            dynamic_input = h
            if (
                self.full_head_variant == "grad_decouple"
                and hasattr(self, "struct_only_head_encoder")
                and hasattr(self, "obs_action_exec_encoder")
            ):
                struct_source = struct_feat if struct_feat is not None else head_feat
                struct_head_feat = self.struct_only_head_encoder(struct_source.reshape(b * a, -1)).view(b, a, -1)
                group_state = self.group_decoder(struct_head_feat.reshape(b * a, -1)).view(b, a, -1)
                dynamic_input = self._build_obs_action_exec_input(h, graph_context)
            elif self.full_head_variant == "id_cond" and hasattr(self, "id_cond_head_encoder"):
                agent_ids = self.agent_id_onehot.to(h.device).unsqueeze(0).expand(b, -1, -1)
                id_cond_feat = th.cat([head_feat, agent_ids], dim=-1)
                id_cond_feat = self.id_cond_head_encoder(id_cond_feat.reshape(b * a, -1)).view(b, a, -1)
                group_state = self.group_decoder(id_cond_feat.reshape(b * a, -1)).view(b, a, -1)
            if self.group_head_mode == "graph_input_fusion_node_embed_struct_feat_full_head_residual":
                q_dynamic, _, _, _, _ = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
                q_static = self.static_out_head(h.reshape(b * a, self.a_h_dim)).view(b * a, 1, self.action_dim)
                q = q_static + self.dynamic_residual_scale * q_dynamic
            else:
                if self.full_head_variant in {"dynamic", "rf", "grad_decouple", "id_cond"}:
                    q, _, _, _, _ = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
                elif self.full_head_variant == "ema_step":
                    q_dynamic, wb, bb, wo, bo = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
                    if not test_mode:
                        self._update_head_param_ema(wb, bb, wo, bo)
                    if test_mode and self.full_head_use_ema_in_test and self.head_param_ema_initialized.item() >= 0.5:
                        q = self._apply_full_head_with_fixed_params(
                            h,
                            self.head_param_ema_wb,
                            self.head_param_ema_bb,
                            self.head_param_ema_wo,
                            self.head_param_ema_bo,
                        )
                    else:
                        q = q_dynamic
                elif self.full_head_variant == "ema_ep_param_mean":
                    q_dynamic, wb, bb, wo, bo = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
                    if not test_mode:
                        self._accumulate_episode_param_mean(wb, bb, wo, bo)
                    if test_mode and self.full_head_use_ema_in_test and self.head_param_ema_initialized.item() >= 0.5:
                        q = self._apply_full_head_with_fixed_params(
                            h,
                            self.head_param_ema_wb,
                            self.head_param_ema_bb,
                            self.head_param_ema_wo,
                            self.head_param_ema_bo,
                        )
                    else:
                        q = q_dynamic
                elif self.full_head_variant == "ema_ep_struct_mean":
                    q_dynamic, _, _, _, _ = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
                    if not test_mode and struct_feat is not None:
                        self._accumulate_episode_struct_mean(struct_feat)
                    if test_mode and self.full_head_use_ema_in_test and self.head_param_ema_initialized.item() >= 0.5:
                        q = self._apply_full_head_with_fixed_params(
                            h,
                            self.head_param_ema_wb,
                            self.head_param_ema_bb,
                            self.head_param_ema_wo,
                            self.head_param_ema_bo,
                        )
                    else:
                        q = q_dynamic
                elif self.full_head_variant == "distill":
                    student_source = node_embed if node_embed is not None else h
                    student_head_feat = self.distill_student_encoder(student_source.reshape(b * a, -1)).view(b, a, -1)
                    mean_student_head_feat = self._update_episode_head_feat_mean(student_head_feat)
                    mean_student_group_state = self.group_decoder(mean_student_head_feat.reshape(b * a, -1)).view(b, a, -1)
                    q_student, _, _, _, _ = self._build_dynamic_full_head_q(h, mean_student_group_state, head_input=dynamic_input)
                    self.distill_student_q = q_student
                    if not test_mode:
                        q_teacher, _, _, _, _ = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
                        self.distill_teacher_q = q_teacher.detach()
                    group_state = mean_student_group_state
                    q = q_student
                else:
                    q, _, _, _, _ = self._build_dynamic_full_head_q(h, group_state, head_input=dynamic_input)
        else:
            fc2_w = self.hyper_w(group_state.reshape(b * a, -1)).reshape(b * a, self.a_h_dim, self.action_dim)
            fc2_b = self.hyper_b(group_state.reshape(b * a, -1)).reshape(b * a, 1, self.action_dim)
            q = th.matmul(h.reshape(b * a, 1, self.a_h_dim), fc2_w) + fc2_b
        return q.view(b, a, -1), group_state

    def _apply_no_group_decoupled_head(self, h, node_state, graph_state):
        b, a, _ = h.size()
        node_flat = node_state.reshape(b * a, -1)
        graph_flat = graph_state.reshape(b * a, -1)

        bottleneck_w = self.hyper_bottleneck_w(node_flat).reshape(b * a, self.a_h_dim, self.a_h_dim)
        bottleneck_b = self.hyper_bottleneck_b(node_flat).reshape(b * a, 1, self.a_h_dim)
        bottleneck = th.matmul(h.reshape(b * a, 1, self.a_h_dim), bottleneck_w) + bottleneck_b
        bottleneck = F.relu(bottleneck, inplace=True)

        fc2_w = self.hyper_w(graph_flat).reshape(b * a, self.a_h_dim, self.action_dim)
        fc2_b = self.hyper_b(graph_flat).reshape(b * a, 1, self.action_dim)
        q = th.matmul(bottleneck, fc2_w) + fc2_b

        group_state = 0.5 * (node_state + graph_state)
        return q.view(b, a, -1), group_state

    def _apply_no_group_full_model(self, inputs, h_prev, head_feat):
        b, a, e = inputs.size()
        group_state = self.group_decoder(head_feat.reshape(b * a, -1)).view(b, a, -1)
        state_flat = group_state.reshape(b * a, -1)

        fc1_w = self.hyper_fc1_w(state_flat).reshape(b * a, e, self.a_h_dim)
        fc1_b = self.hyper_fc1_b(state_flat).reshape(b * a, self.a_h_dim)
        x = th.bmm(inputs.reshape(b * a, 1, e), fc1_w).squeeze(1) + fc1_b
        x = F.relu(x, inplace=True)

        w_ih = self.hyper_gru_ih_w(state_flat).reshape(b * a, 3 * self.a_h_dim, self.a_h_dim)
        b_ih = self.hyper_gru_ih_b(state_flat).reshape(b * a, 3 * self.a_h_dim)
        w_hh = self.hyper_gru_hh_w(state_flat).reshape(b * a, 3 * self.a_h_dim, self.a_h_dim)
        b_hh = self.hyper_gru_hh_b(state_flat).reshape(b * a, 3 * self.a_h_dim)

        h_prev_flat = h_prev.reshape(b * a, self.a_h_dim)
        gi = th.bmm(x.unsqueeze(1), w_ih.transpose(1, 2)).squeeze(1) + b_ih
        gh = th.bmm(h_prev_flat.unsqueeze(1), w_hh.transpose(1, 2)).squeeze(1) + b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        resetgate = th.sigmoid(i_r + h_r)
        updategate = th.sigmoid(i_z + h_z)
        newgate = th.tanh(i_n + resetgate * h_n)
        h = newgate + updategate * (h_prev_flat - newgate)

        bottleneck_w = self.hyper_bottleneck_w(state_flat).reshape(b * a, self.a_h_dim, self.a_h_dim)
        bottleneck_b = self.hyper_bottleneck_b(state_flat).reshape(b * a, 1, self.a_h_dim)
        bottleneck = th.matmul(h.reshape(b * a, 1, self.a_h_dim), bottleneck_w) + bottleneck_b
        bottleneck = F.relu(bottleneck, inplace=True)
        fc2_w = self.hyper_w(state_flat).reshape(b * a, self.a_h_dim, self.action_dim)
        fc2_b = self.hyper_b(state_flat).reshape(b * a, 1, self.action_dim)
        q = th.matmul(bottleneck, fc2_w) + fc2_b
        return q.view(b, a, -1), h.view(b, a, -1), group_state

    def forward(self, inputs, hidden_state, graph_context=None, test_mode=False):
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
            "graph_input_fusion_node_embed_no_reg",
            "graph_input_fusion_node_embed_no_groupemb_no_reg",
            "graph_input_fusion_node_embed_struct_only",
            "graph_input_fusion_node_embed_sharp",
            "graph_input_fusion_node_embed_threshold_group",
            "graph_input_fusion_hidden_head",
            "graph_input_fusion_node_embed_head",
            "graph_input_fusion_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_head",
            "graph_input_fusion_node_embed_struct_feat_two_layer_head",
            "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
            "graph_input_fusion_node_embed_struct_feat_full_head",
            "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
            "graph_input_fusion_node_embed_struct_feat_full_head_residual",
            "graph_input_fusion_node_embed_struct_feat_decoupled_head",
            "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
            "graph_input_fusion_node_embed_gcn_full_head",
            "graph_input_fusion_node_embed_subgraph_full_head",
            "graph_input_fusion_node_embed_struct_feat_full_model",
            "graph_input_fusion_node_embed_group_full_head",
            "graph_input_fusion_group_only",
            "graph_input_fusion_head_only",
        ]:
            if self.group_head_mode == "graph_input_fusion_node_embed_struct_only":
                group_probs, struct_feat, group_graphs = self._build_graph_input_fusion_node_embed_struct_only(h, graph_context)
                group_state = self.group_decoder(struct_feat.reshape(b * a, -1)).view(b, a, -1)
            elif self.group_head_mode in self.no_group_head_compare_modes or self.group_head_mode in self.no_group_param_scope_modes:
                group_probs, struct_feat, group_graphs, head_feat, node_embed = self._build_graph_input_fusion_no_group_head(
                    h, graph_context, test_mode=test_mode
                )
                q, group_state = self._apply_no_group_dynamic_head(
                    h, head_feat, struct_feat=struct_feat, node_embed=node_embed, graph_context=graph_context, test_mode=test_mode
                )
            elif self.group_head_mode in self.no_group_decoupled_modes:
                group_probs, struct_feat, group_graphs, node_state, graph_state, node_embed = (
                    self._build_graph_input_fusion_no_group_decoupled(h, graph_context)
                )
                q, group_state = self._apply_no_group_decoupled_head(h, node_state, graph_state)
            elif self.group_head_mode in self.no_group_gcn_modes:
                group_probs, struct_feat, group_graphs, head_feat, node_embed = self._build_graph_input_fusion_no_group_gcn(
                    h, graph_context
                )
                q, group_state = self._apply_no_group_dynamic_head(
                    h, head_feat, struct_feat=struct_feat, node_embed=node_embed, graph_context=graph_context, test_mode=test_mode
                )
            elif self.group_head_mode in self.no_group_subgraph_modes:
                group_probs, struct_feat, group_graphs, head_feat, node_embed = self._build_graph_input_fusion_no_group_subgraph(
                    h, graph_context
                )
                q, group_state = self._apply_no_group_dynamic_head(
                    h, head_feat, struct_feat=struct_feat, node_embed=node_embed, graph_context=graph_context, test_mode=test_mode
                )
            elif self.group_head_mode in self.no_group_full_model_modes:
                h_prev = hidden_state.view(b, a, -1)
                group_probs, struct_feat, group_graphs, head_feat, node_embed = self._build_graph_input_fusion_no_group_head(
                    h_prev, graph_context, test_mode=test_mode
                )
                q, h, group_state = self._apply_no_group_full_model(inputs, h_prev, head_feat)
            else:
                graph_source = None
                node_embed = None
                if self.group_head_mode in [
                    "graph_input_fusion",
                    "graph_input_fusion_node_embed",
                    "graph_input_fusion_node_embed_no_groupemb",
                    "graph_input_fusion_node_embed_no_reg",
                    "graph_input_fusion_node_embed_no_groupemb_no_reg",
                    "graph_input_fusion_node_embed_sharp",
                    "graph_input_fusion_node_embed_threshold_group",
                    "graph_input_fusion_hidden_head",
                    "graph_input_fusion_node_embed_head",
                    "graph_input_fusion_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_head",
                    "graph_input_fusion_node_embed_struct_feat_two_layer_head",
                    "graph_input_fusion_node_embed_struct_feat_bottleneck_head",
                    "graph_input_fusion_node_embed_struct_feat_full_head",
                    "graph_input_fusion_node_embed_struct_feat_full_head_early_node",
                    "graph_input_fusion_node_embed_struct_feat_full_head_residual",
                    "graph_input_fusion_node_embed_struct_feat_decoupled_head",
                    "graph_input_fusion_node_embed_struct_feat_decoupled_residual_head",
                    "graph_input_fusion_node_embed_gcn_full_head",
                    "graph_input_fusion_node_embed_subgraph_full_head",
                    "graph_input_fusion_node_embed_struct_feat_full_model",
                    "graph_input_fusion_node_embed_group_full_head",
                    "graph_input_fusion_group_only",
                    "graph_input_fusion_head_only",
                ]:
                    graph_source = self._build_graph_input_fusion_source(h, graph_context)
                    node_embed = (
                        graph_source
                        if self.group_head_mode in [
                            "graph_input_fusion_node_embed",
                            "graph_input_fusion_node_embed_no_groupemb",
                            "graph_input_fusion_node_embed_no_reg",
                            "graph_input_fusion_node_embed_no_groupemb_no_reg",
                            "graph_input_fusion_node_embed_sharp",
                            "graph_input_fusion_node_embed_threshold_group",
                            "graph_input_fusion_node_embed_group_full_head",
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
                elif self.group_head_mode in {
                    "graph_input_fusion_node_embed_no_groupemb",
                    "graph_input_fusion_node_embed_no_groupemb_no_reg",
                }:
                    group_state = self.group_decoder(struct_feat.reshape(b * a, -1)).view(b, a, -1)
                else:
                    group_state = self.group_decoder((struct_feat + group_emb).reshape(b * a, -1)).view(b, a, -1)
            if self.group_head_mode in self.grouped_full_head_modes:
                q = self._apply_group_dynamic_full_head(h, group_state)
            elif (
                self.group_head_mode not in self.no_group_head_compare_modes
                and self.group_head_mode not in self.no_group_param_scope_modes
                and self.group_head_mode not in self.no_group_decoupled_modes
                and self.group_head_mode not in self.no_group_gcn_modes
                and self.group_head_mode not in self.no_group_subgraph_modes
                and self.group_head_mode not in self.no_group_full_model_modes
            ):
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
