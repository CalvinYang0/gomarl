import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from envs.starcraft.smac_maps import get_map_params


ACTION_EDGE_PUBLIC_PRED_SINGLE_HEAD_VARIANTS = {
    "rpg_action_edge_public_pred_public_hyper_private_input_single_head",
    "rpg_action_edge_public_pred_private_hyper_public_input_single_head",
}
ACTION_EDGE_PUBLIC_PRED_COARSE_HEAD_VARIANTS = {
    "rpg_action_edge_public_pred_coarse_fine_four_layer_head",
    "rpg_action_edge_public_pred_coarse_q_fine_gate_head",
}
ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS = (
    ACTION_EDGE_PUBLIC_PRED_SINGLE_HEAD_VARIANTS | ACTION_EDGE_PUBLIC_PRED_COARSE_HEAD_VARIANTS
)
ACTION_EDGE_STRUCT_TRANSFORMER_REL_PRIVATE_VARIANTS = {
    "rpg_action_edge_graphormer_relation_private_single_head",
    "rpg_action_edge_graphit_relation_private_single_head",
    "rpg_action_edge_edgeset_relation_private_single_head",
    "rpg_action_edge_motif_transformer_relation_private_single_head",
}
ACTION_EDGE_REL_PRIVATE_SINGLE_HEAD_VARIANTS = (
    {"rpg_action_edge_public_pred_relation_private_single_head"}
    | ACTION_EDGE_STRUCT_TRANSFORMER_REL_PRIVATE_VARIANTS
)
ACTION_EDGE_REL_PRIVATE_VARIANTS = (
    ACTION_EDGE_REL_PRIVATE_SINGLE_HEAD_VARIANTS
    | {"rpg_action_edge_public_pred_relation_private_decision_maker"}
)
RPG_POST_TARGET_SELECTION_VARIANTS = {
    "rpg_post_topk_enemy_interaction_hypercond",
    "rpg_post_threshold_enemy_interaction_hypercond",
}
RPG_PRE_RELATION_SELECTION_VARIANTS = {
    "rpg_pre_topk_entity_relation_hypercond",
    "rpg_pre_threshold_entity_relation_hypercond",
}
RPG_ENEMY_TOKEN_VARIANTS = {
    "rpg_no_enemy_token_interaction_hypercond",
    "rpg_private_enemy_token_interaction_hypercond",
    "rpg_delta_enemy_token_interaction_hypercond",
}
RPG_TARGETWISE_ABLATION_VARIANTS = (
    RPG_POST_TARGET_SELECTION_VARIANTS
    | RPG_PRE_RELATION_SELECTION_VARIANTS
    | RPG_ENEMY_TOKEN_VARIANTS
)
RPG_STANDARD_INTERACTION_ABLATION_VARIANTS = (
    RPG_TARGETWISE_ABLATION_VARIANTS - {"rpg_delta_enemy_token_interaction_hypercond"}
)
RPG_POLICY_RELATION_FUSION_HEAD_VARIANTS = {
    "rpg_policy_relation_fusion_head_hypercond",
}
RPG_TOKEN_DECISION_HEAD_VARIANTS = {
    "rpg_entity_token_decision_head_hypercond",
    "rpg_self_enemy_pair_token_decision_head_hypercond",
    "rpg_relation_token_decision_head_hypercond",
} | RPG_POLICY_RELATION_FUSION_HEAD_VARIANTS
PUBLIC_TRANSFORMER_FUTURE_DELTA_VARIANTS = {
    "rpg_public_future_delta_token_transformer_hypercond",
    "rpg_public_future_delta_bias_transformer_hypercond",
}
PUBLIC_TRANSFORMER_FUTURE_DELTA_SINGLE_HEAD_VARIANTS = {
    "rpg_public_future_delta_token_transformer_single_head_hypercond",
    "rpg_public_future_delta_bias_transformer_single_head_hypercond",
}
PUBLIC_TRANSFORMER_PAST_DELTA_VARIANTS = {
    "rpg_public_past_delta_token_transformer_hypercond",
    "rpg_public_past_delta_bias_transformer_hypercond",
}
PUBLIC_TRANSFORMER_PAST_DELTA_SINGLE_HEAD_VARIANTS = {
    "rpg_public_past_delta_token_transformer_single_head_hypercond",
    "rpg_public_past_delta_bias_transformer_single_head_hypercond",
}
PUBLIC_TRANSFORMER_MIXED_VARIANTS = {
    "rpg_public_private_bias_past_delta_token_transformer_hypercond",
    "rpg_public_private_bias_past_delta_token_transformer_enemy_slot_hypercond",
    "rpg_public_private_token_past_delta_bias_transformer_enemy_slot_hypercond",
}
PUBLIC_TRANSFORMER_FULL_TOKEN_VARIANTS = {
    "rpg_public_private_full_token_transformer_hypercond",
}
PUBLIC_TRANSFORMER_FULL_TOKEN_RELATION_HEAD_VARIANTS = {
    "rpg_public_private_full_token_transformer_relation_token_head_hypercond",
}
PUBLIC_TRANSFORMER_FULL_OBS_RELATION_HEAD_VARIANTS = {
    "rpg_full_obs_transformer_relation_token_head_hypercond",
}
PUBLIC_TRANSFORMER_FULL_OBS_VARIANTS = {
    "rpg_full_obs_transformer_hypercond",
}
PUBLIC_TRANSFORMER_GLOBAL_PUBLIC_VARIANTS = {
    "rpg_global_public_transformer_hypercond",
    "rpg_global_public_private_bias_transformer_hypercond",
    "rpg_global_public_private_bias_transformer_eval_global_hypercond",
    "rpg_global_public_private_bias_transformer_memory_eval_hypercond",
    "rpg_global_public_private_bias_past_delta_token_transformer_hypercond",
    "rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond",
    "rpg_global_public_private_bias_past_delta_token_transformer_threshold_hypercond",
    "rpg_global_public_transformer_relation_token_head_hypercond",
}
PUBLIC_TRANSFORMER_EVAL_GLOBAL_VARIANTS = {
    "rpg_global_public_private_bias_transformer_eval_global_hypercond",
}
PUBLIC_TRANSFORMER_MEMORY_EVAL_VARIANTS = {
    "rpg_global_public_private_bias_transformer_memory_eval_hypercond",
}
PUBLIC_TRANSFORMER_RELATION_TOKEN_HEAD_VARIANTS = {
    "rpg_public_transformer_relation_token_head_hypercond",
    "rpg_public_private_bias_transformer_relation_token_head_hypercond",
    "rpg_public_private_bias_past_delta_token_transformer_relation_token_head_hypercond",
    "rpg_global_public_transformer_relation_token_head_hypercond",
} | PUBLIC_TRANSFORMER_FULL_TOKEN_RELATION_HEAD_VARIANTS | PUBLIC_TRANSFORMER_FULL_OBS_RELATION_HEAD_VARIANTS
PUBLIC_TRANSFORMER_RELATION_PAIR_TOKEN_HEAD_VARIANTS = {
    "rpg_public_private_bias_transformer_relation_pair_token_head_hypercond",
}
PUBLIC_TRANSFORMER_RELATION_PRIVATE_TOKEN_HEAD_VARIANTS = {
    "rpg_public_private_bias_transformer_relation_private_token_head_hypercond",
}
PUBLIC_TRANSFORMER_RELATION_DELTA_TOKEN_HEAD_VARIANTS = {
    "rpg_public_private_bias_transformer_relation_delta_token_head_hypercond",
}
PUBLIC_TRANSFORMER_SLOT_TOKEN_HEAD_VARIANTS = {
    "rpg_public_private_bias_transformer_slot_token_head_hypercond",
}
PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS = {
    "rpg_public_private_bias_past_delta_token_transformer_relation_token_topk_hypercond",
}
PUBLIC_TRANSFORMER_TARGET_SELECTION_VARIANTS = {
    "rpg_public_private_bias_transformer_topk_hypercond",
    "rpg_public_private_bias_transformer_threshold_hypercond",
    "rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond",
    "rpg_global_public_private_bias_past_delta_token_transformer_threshold_hypercond",
}
PUBLIC_TRANSFORMER_STABLE_HEAD_VARIANTS = {
    "rpg_public_private_simple_bias_transformer_q_residual_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_smooth_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_smooth_hypercond",
    "rpg_public_private_simple_bias_transformer_smooth_hypercond",
}
PUBLIC_TRANSFORMER_Q_RESIDUAL_HEAD_VARIANTS = {
    "rpg_public_private_simple_bias_transformer_q_residual_hypercond",
}
PUBLIC_TRANSFORMER_PARAM_RESIDUAL_HEAD_VARIANTS = {
    "rpg_public_private_simple_bias_transformer_param_residual_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_smooth_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_smooth_hypercond",
}
PUBLIC_TRANSFORMER_RESIDUAL_L2_HEAD_VARIANTS = {
    "rpg_public_private_simple_bias_transformer_param_residual_l2_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_smooth_hypercond",
}
PUBLIC_TRANSFORMER_RESIDUAL_SMOOTH_HEAD_VARIANTS = {
    "rpg_public_private_simple_bias_transformer_param_residual_smooth_hypercond",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_smooth_hypercond",
}
PUBLIC_TRANSFORMER_SMOOTH_HEAD_VARIANTS = {
    "rpg_public_private_simple_bias_transformer_smooth_hypercond",
} | PUBLIC_TRANSFORMER_RESIDUAL_SMOOTH_HEAD_VARIANTS
SEMANTIC_ROUTER_MODE_BY_MODEL = {
    "rpg_simple_bias_observer_consistency_router_hypercond": "observer_consistency",
    "rpg_simple_bias_temporal_stability_router_hypercond": "temporal_stability",
    "rpg_simple_bias_gradient_importance_router_hypercond": "gradient_importance",
    "rpg_simple_bias_gradient_consistency_router_hypercond": "gradient_consistency",
    "rpg_simple_bias_parameter_sensitivity_router_hypercond": "parameter_sensitivity",
    "rpg_simple_bias_counterfactual_router_hypercond": "counterfactual",
}
PUBLIC_TRANSFORMER_SEMANTIC_ROUTER_VARIANTS = set(SEMANTIC_ROUTER_MODE_BY_MODEL)
PUBLIC_TRANSFORMER_SIMPLE_BIAS_FAMILY = {
    "rpg_public_private_simple_bias_transformer_hypercond",
} | PUBLIC_TRANSFORMER_STABLE_HEAD_VARIANTS | PUBLIC_TRANSFORMER_SEMANTIC_ROUTER_VARIANTS
PUBLIC_TRANSFORMER_FRIEND_MERGED_VARIANTS = (
    PUBLIC_TRANSFORMER_GLOBAL_PUBLIC_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS
    | PUBLIC_TRANSFORMER_FULL_TOKEN_VARIANTS
    | {
        "rpg_public_private_bias_friend_public_transformer_hypercond",
        "rpg_public_private_owner_bias_transformer_hypercond",
        "rpg_public_private_selfattn_bias_transformer_hypercond",
    }
    | PUBLIC_TRANSFORMER_SIMPLE_BIAS_FAMILY
)
PUBLIC_TRANSFORMER_MIXED_SINGLE_HEAD_VARIANTS = {
    "rpg_public_private_bias_past_delta_token_transformer_single_head_hypercond",
}
PUBLIC_TRANSFORMER_PRIVATE_HEAD_INPUT_VARIANTS = {
    "rpg_public_past_delta_bias_transformer_private_head_input_hypercond",
}
PUBLIC_TRANSFORMER_PAIR_INTERACTION_VARIANTS = {
    "rpg_public_private_bias_transformer_pair_interaction_hypercond",
}
PUBLIC_TRANSFORMER_PAIR_CONCAT_INTERACTION_VARIANTS = {
    "rpg_public_private_bias_transformer_pair_concat_interaction_hypercond",
}
PUBLIC_TRANSFORMER_PAIR_FEATURE_INTERACTION_VARIANTS = (
    PUBLIC_TRANSFORMER_PAIR_INTERACTION_VARIANTS | PUBLIC_TRANSFORMER_PAIR_CONCAT_INTERACTION_VARIANTS
)
PUBLIC_TRANSFORMER_PRIVATE_VARIANTS = {
    "rpg_public_private_token_transformer_hypercond",
    "rpg_public_private_bias_transformer_hypercond",
    "rpg_public_private_bias_friend_public_transformer_hypercond",
    "rpg_public_private_owner_bias_transformer_hypercond",
    "rpg_public_private_selfattn_bias_transformer_hypercond",
} | PUBLIC_TRANSFORMER_SIMPLE_BIAS_FAMILY
PUBLIC_TRANSFORMER_PRIVATE_SINGLE_HEAD_VARIANTS = {
    "rpg_public_private_token_transformer_single_head_hypercond",
    "rpg_public_private_bias_transformer_single_head_hypercond",
}
PUBLIC_TRANSFORMER_TOKEN_VARIANTS = {
    "rpg_public_future_delta_token_transformer_hypercond",
    "rpg_public_past_delta_token_transformer_hypercond",
    "rpg_public_private_token_transformer_hypercond",
    "rpg_public_private_bias_past_delta_token_transformer_hypercond",
    "rpg_public_private_token_past_delta_bias_transformer_enemy_slot_hypercond",
}
PUBLIC_TRANSFORMER_BIAS_VARIANTS = {
    "rpg_public_future_delta_bias_transformer_hypercond",
    "rpg_public_past_delta_bias_transformer_hypercond",
    "rpg_public_private_bias_transformer_hypercond",
    "rpg_public_private_bias_friend_public_transformer_hypercond",
    "rpg_public_private_owner_bias_transformer_hypercond",
    "rpg_public_private_selfattn_bias_transformer_hypercond",
    "rpg_public_private_bias_past_delta_token_transformer_hypercond",
    "rpg_public_private_token_past_delta_bias_transformer_enemy_slot_hypercond",
    "rpg_public_past_delta_bias_transformer_private_head_input_hypercond",
} | PUBLIC_TRANSFORMER_SIMPLE_BIAS_FAMILY
PUBLIC_TRANSFORMER_RELATION_VARIANTS = (
    {"rpg_public_transformer_hypercond"}
    | PUBLIC_TRANSFORMER_FUTURE_DELTA_VARIANTS
    | PUBLIC_TRANSFORMER_PAST_DELTA_VARIANTS
    | PUBLIC_TRANSFORMER_MIXED_VARIANTS
    | PUBLIC_TRANSFORMER_FULL_TOKEN_VARIANTS
    | PUBLIC_TRANSFORMER_FULL_OBS_VARIANTS
    | PUBLIC_TRANSFORMER_GLOBAL_PUBLIC_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_PAIR_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_PRIVATE_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_DELTA_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_SLOT_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS
    | PUBLIC_TRANSFORMER_TARGET_SELECTION_VARIANTS
    | PUBLIC_TRANSFORMER_PRIVATE_HEAD_INPUT_VARIANTS
    | PUBLIC_TRANSFORMER_PAIR_FEATURE_INTERACTION_VARIANTS
    | PUBLIC_TRANSFORMER_PRIVATE_VARIANTS
)
PUBLIC_TRANSFORMER_STANDARD_RELATION_VARIANTS = (
    PUBLIC_TRANSFORMER_RELATION_VARIANTS
    - PUBLIC_TRANSFORMER_PRIVATE_HEAD_INPUT_VARIANTS
    - PUBLIC_TRANSFORMER_PAIR_FEATURE_INTERACTION_VARIANTS
)
PUBLIC_TRANSFORMER_SINGLE_HEAD_VARIANTS = (
    {"rpg_public_transformer_single_head_hypercond"}
    | PUBLIC_TRANSFORMER_FUTURE_DELTA_SINGLE_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_PAST_DELTA_SINGLE_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_MIXED_SINGLE_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_PRIVATE_SINGLE_HEAD_VARIANTS
)
PUBLIC_TRANSFORMER_CAPTURER_VARIANTS = (
    PUBLIC_TRANSFORMER_RELATION_VARIANTS | PUBLIC_TRANSFORMER_SINGLE_HEAD_VARIANTS
)
PUBLIC_TRANSFORMER_FUTURE_DELTA_ALL_VARIANTS = (
    PUBLIC_TRANSFORMER_FUTURE_DELTA_VARIANTS | PUBLIC_TRANSFORMER_FUTURE_DELTA_SINGLE_HEAD_VARIANTS
)
GRF_DECISION_MAKER_VARIANTS = {
    "grf_public_private_bias_transformer_decision_maker_hypercond",
    "grf_abs_public_private_bias_transformer_decision_maker_hypercond",
}
GRF_TWO_LAYER_HEAD_VARIANTS = {
    "grf_public_private_bias_transformer_two_layer_head_hypercond",
    "grf_abs_public_private_bias_transformer_two_layer_head_hypercond",
}
GRF_LINEAR_HEAD_VARIANTS = {
    "grf_public_private_bias_transformer_linear_head_hypercond",
    "grf_abs_public_private_bias_transformer_linear_head_hypercond",
}
GRF_PUBLIC_TRANSFORMER_VARIANTS = {
    "grf_public_private_bias_transformer_hypercond",
    "grf_abs_public_private_bias_transformer_hypercond",
    *GRF_TWO_LAYER_HEAD_VARIANTS,
    *GRF_LINEAR_HEAD_VARIANTS,
    *GRF_DECISION_MAKER_VARIANTS,
}
PUBLIC_TRANSFORMER_MODE_BY_MODEL = {
    "rpg_public_transformer_hypercond": "baseline",
    "rpg_public_transformer_single_head_hypercond": "baseline",
    "rpg_public_future_delta_token_transformer_hypercond": "future_delta_token",
    "rpg_public_future_delta_token_transformer_single_head_hypercond": "future_delta_token",
    "rpg_public_future_delta_bias_transformer_hypercond": "future_delta_bias",
    "rpg_public_future_delta_bias_transformer_single_head_hypercond": "future_delta_bias",
    "rpg_public_private_token_transformer_hypercond": "private_token",
    "rpg_public_private_token_transformer_single_head_hypercond": "private_token",
    "rpg_public_private_bias_transformer_hypercond": "private_bias",
    "rpg_public_private_bias_friend_public_transformer_hypercond": "private_bias",
    "rpg_public_private_owner_bias_transformer_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_q_residual_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_param_residual_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_param_residual_smooth_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_param_residual_l2_smooth_hypercond": "private_bias",
    "rpg_public_private_simple_bias_transformer_smooth_hypercond": "private_bias",
    **{
        name: "private_bias"
        for name in PUBLIC_TRANSFORMER_SEMANTIC_ROUTER_VARIANTS
    },
    "rpg_public_private_selfattn_bias_transformer_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_pair_interaction_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_pair_concat_interaction_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_single_head_hypercond": "private_bias",
    "rpg_public_past_delta_token_transformer_hypercond": "past_delta_token",
    "rpg_public_past_delta_token_transformer_single_head_hypercond": "past_delta_token",
    "rpg_public_past_delta_bias_transformer_hypercond": "past_delta_bias",
    "rpg_public_past_delta_bias_transformer_single_head_hypercond": "past_delta_bias",
    "rpg_public_private_bias_past_delta_token_transformer_hypercond": "private_bias_past_delta_token",
    "rpg_public_private_bias_past_delta_token_transformer_single_head_hypercond": "private_bias_past_delta_token",
    "rpg_public_private_bias_past_delta_token_transformer_enemy_slot_hypercond": "private_bias_past_delta_token",
    "rpg_public_private_token_past_delta_bias_transformer_enemy_slot_hypercond": "private_token_past_delta_bias",
    "rpg_public_past_delta_bias_transformer_private_head_input_hypercond": "past_delta_bias",
    "rpg_global_public_transformer_hypercond": "baseline",
    "rpg_global_public_private_bias_transformer_hypercond": "private_bias",
    "rpg_global_public_private_bias_transformer_eval_global_hypercond": "private_bias",
    "rpg_global_public_private_bias_transformer_memory_eval_hypercond": "private_bias",
    "rpg_global_public_private_bias_past_delta_token_transformer_hypercond": "private_bias_past_delta_token",
    "rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond": "private_bias_past_delta_token",
    "rpg_global_public_private_bias_past_delta_token_transformer_threshold_hypercond": "private_bias_past_delta_token",
    "rpg_public_transformer_relation_token_head_hypercond": "baseline",
    "rpg_public_private_bias_transformer_relation_token_head_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_relation_pair_token_head_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_relation_private_token_head_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_relation_delta_token_head_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_slot_token_head_hypercond": "private_bias",
    "rpg_public_private_bias_past_delta_token_transformer_relation_token_head_hypercond": "private_bias_past_delta_token",
    "rpg_public_private_bias_past_delta_token_transformer_relation_token_topk_hypercond": "private_bias_past_delta_token",
    "rpg_public_private_bias_transformer_topk_hypercond": "private_bias",
    "rpg_public_private_bias_transformer_threshold_hypercond": "private_bias",
    "rpg_global_public_transformer_relation_token_head_hypercond": "baseline",
    "rpg_public_private_full_token_transformer_hypercond": "public_private_full_token",
    "rpg_public_private_full_token_transformer_relation_token_head_hypercond": "public_private_full_token",
    "rpg_full_obs_transformer_hypercond": "full_obs",
    "rpg_full_obs_transformer_relation_token_head_hypercond": "full_obs",
}
PUBLIC_TRANSFORMER_TOKEN_DECISION_HEAD_VARIANTS = (
    PUBLIC_TRANSFORMER_RELATION_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_PAIR_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_PRIVATE_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_DELTA_TOKEN_HEAD_VARIANTS
    | PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS
)
RELATION_TOKEN_DECISION_HEAD_VARIANTS = (
    {"rpg_relation_token_decision_head_hypercond"} | PUBLIC_TRANSFORMER_TOKEN_DECISION_HEAD_VARIANTS
)
TOKEN_DECISION_HEAD_VARIANTS = RPG_TOKEN_DECISION_HEAD_VARIANTS | PUBLIC_TRANSFORMER_TOKEN_DECISION_HEAD_VARIANTS


def _neg_inf_like(tensor):
    if tensor.is_floating_point():
        return th.finfo(tensor.dtype).min
    return -9999999


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


class BiasMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, key_mask, attn_bias=None):
        batch_size, n_tokens, _ = x.shape
        qkv = self.qkv(x).view(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e4)
        attn = F.softmax(scores, dim=-1)
        out = th.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, n_tokens, self.dim)
        return self.out_proj(out)


class BiasTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.self_attn = BiasMultiHeadSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, key_mask, attn_bias=None):
        x = self.norm1(x + self.self_attn(x, key_mask, attn_bias=attn_bias))
        x = self.norm2(x + self.ffn(x))
        if key_mask is not None:
            x = x * key_mask.unsqueeze(-1).float()
        return x


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
        masked_logits = logits.masked_fill(~valid_mask, _neg_inf_like(logits))
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


class PreSelectRPGRelationCapturer(RPGInspiredRelationCapturer):
    # Pre-relation bottleneck: score ally/enemy context entities from the
    # self-centered query, then let the original RPG-style cross-attention see
    # only selected context entities. Target Q-values are still produced for
    # every observed enemy, so this isolates relation-generation selection.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        selection_mode="topk",
        topk=5,
        threshold=0.5,
    ):
        super().__init__(
            move_dim=move_dim,
            own_dim=own_dim,
            ally_feat_dim=ally_feat_dim,
            enemy_feat_dim=enemy_feat_dim,
            relation_dim=relation_dim,
            output_dim=output_dim,
        )
        self.selection_mode = selection_mode
        self.topk = int(topk)
        self.threshold = float(threshold)

    def _select_entity_mask(self, scores, entity_mask):
        valid_mask = entity_mask.bool()
        if scores.size(-1) == 0:
            return valid_mask
        masked_scores = scores.masked_fill(~valid_mask, _neg_inf_like(scores))
        valid_any = valid_mask.any(dim=-1, keepdim=True)
        if self.selection_mode == "topk":
            k = max(1, min(self.topk, scores.size(-1)))
            _, indices = th.topk(masked_scores, k=k, dim=-1)
            gathered_valid = th.gather(valid_mask, dim=-1, index=indices)
            selected = scores.new_zeros(scores.shape)
            selected.scatter_(dim=-1, index=indices, src=gathered_valid.float())
            selected = selected.bool()
        else:
            selected = (th.sigmoid(scores) >= self.threshold) & valid_mask
            selected_any = selected.any(dim=-1, keepdim=True)
            fallback_idx = masked_scores.argmax(dim=-1, keepdim=True)
            fallback = th.zeros_like(selected)
            fallback.scatter_(dim=-1, index=fallback_idx, src=valid_any)
            selected = th.where(selected_any, selected, fallback)
        return selected & valid_mask

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        self_token = self.self_encoder(self_feat)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        ally_tokens = self.ally_encoder(ally_feat) * ally_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        query = self.self_query(self_token)
        scale = math.sqrt(float(self.relation_dim))
        ally_scores = th.matmul(query.unsqueeze(2), self.ally_key(ally_tokens).transpose(-1, -2)).squeeze(2) / scale
        enemy_scores = th.matmul(query.unsqueeze(2), self.enemy_key(enemy_tokens).transpose(-1, -2)).squeeze(2) / scale
        entity_scores = th.cat([ally_scores, enemy_scores], dim=-1)
        entity_mask = th.cat([ally_mask, enemy_mask], dim=-1)
        selected_entity_mask = self._select_entity_mask(entity_scores, entity_mask)
        selected_ally_mask = selected_entity_mask[:, :, : ally_feat.size(2)]
        selected_enemy_mask = selected_entity_mask[:, :, ally_feat.size(2) :]

        ally_context, ally_attn = self._masked_cross_attention(
            self_token, ally_tokens, selected_ally_mask, self.ally_key, self.ally_value
        )
        enemy_context, enemy_attn = self._masked_cross_attention(
            self_token, enemy_tokens, selected_enemy_mask, self.enemy_key, self.enemy_value
        )

        instant = self.instant_pattern(th.cat([self_token, ally_context, enemy_context], dim=-1))
        temporal_input = th.cat([self_token, instant], dim=-1)

        batch_size, n_agents, _ = temporal_input.shape
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class PublicRPGRelationCapturer(nn.Module):
    # Public-information relation generator. It keeps observer-invariant entity
    # state such as health/shield/type, and removes private self-view fields
    # such as movement context, relative position, and attack availability. The
    # downstream maker remains unchanged.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        unit_type_bits=0,
        shield_bits_ally=0,
        shield_bits_enemy=0,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        n_actions=0,
    ):
        super().__init__()
        del move_dim, own_dim
        self.ally_feat_dim = ally_feat_dim
        self.enemy_feat_dim = enemy_feat_dim
        self.relation_dim = relation_dim
        self.unit_type_bits = unit_type_bits
        self.shield_bits_ally = shield_bits_ally
        self.shield_bits_enemy = shield_bits_enemy
        self.obs_all_health = obs_all_health
        self.obs_own_health = obs_own_health
        self.obs_last_action = obs_last_action
        self.n_actions = n_actions

        self.public_self_dim = 1
        self.public_ally_dim = 1
        self.public_enemy_dim = 1
        if obs_own_health:
            self.public_self_dim += 1 + shield_bits_ally
        if obs_all_health:
            self.public_ally_dim += 1 + shield_bits_ally
            self.public_enemy_dim += 1 + shield_bits_enemy
        self.public_self_dim += unit_type_bits
        self.public_ally_dim += unit_type_bits
        self.public_enemy_dim += unit_type_bits

        self.self_encoder = nn.Sequential(
            nn.Linear(self.public_self_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.ally_encoder = nn.Sequential(
            nn.Linear(self.public_ally_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.public_enemy_encoder = nn.Sequential(
            nn.Linear(self.public_enemy_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.public_query_encoder = nn.Sequential(
            nn.Linear(relation_dim * 3, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )

        self.query = nn.Linear(relation_dim, relation_dim)
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

    def _masked_mean(self, tokens, mask):
        denom = mask.sum(dim=2, keepdim=True).clamp(min=1).float()
        return (tokens * mask.unsqueeze(-1).float()).sum(dim=2) / denom

    def _masked_attention(self, query, tokens, mask, key_proj, value_proj):
        scale = math.sqrt(float(self.relation_dim))
        key = key_proj(tokens)
        value = value_proj(tokens)
        logits = th.matmul(self.query(query).unsqueeze(2), key.transpose(-1, -2)).squeeze(2) / scale
        valid_mask = mask.bool()
        valid_any = valid_mask.any(dim=-1, keepdim=True)
        masked_logits = logits.masked_fill(~valid_mask, _neg_inf_like(logits))
        attn = F.softmax(masked_logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))
        context = th.matmul(attn.unsqueeze(2), value).squeeze(2)
        return context, attn

    def _self_public_features(self, self_feat):
        batch_size, n_agents, _ = self_feat.shape
        features = [self_feat.new_ones(batch_size, n_agents, 1)]
        idx = 0
        if self.obs_own_health:
            features.append(self_feat[:, :, idx : idx + 1])
            idx += 1
            if self.shield_bits_ally > 0:
                features.append(self_feat[:, :, idx : idx + 1])
                idx += 1
        if self.unit_type_bits > 0:
            features.append(self_feat[:, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _enemy_public_features(self, enemy_feat, enemy_mask):
        features = [enemy_mask.unsqueeze(-1).float()]
        idx = 4
        if self.obs_all_health:
            features.append(enemy_feat[:, :, :, idx : idx + 1])
            idx += 1
            if self.shield_bits_enemy > 0:
                features.append(enemy_feat[:, :, :, idx : idx + 1])
                idx += 1
        if self.unit_type_bits > 0:
            features.append(enemy_feat[:, :, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _ally_public_features(self, ally_feat, ally_mask):
        features = [ally_mask.unsqueeze(-1).float()]
        idx = 4
        if self.obs_all_health:
            features.append(ally_feat[:, :, :, idx : idx + 1])
            idx += 1
            if self.shield_bits_ally > 0:
                features.append(ally_feat[:, :, :, idx : idx + 1])
                idx += 1
        if self.unit_type_bits > 0:
            features.append(ally_feat[:, :, :, idx : idx + self.unit_type_bits])
            idx += self.unit_type_bits
        # Deliberately skip ally last action: it is agent-specific execution
        # context, not the public situation signal for head generation.
        return th.cat(features, dim=-1)

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        self_public = self._self_public_features(self_feat)
        ally_public = self._ally_public_features(ally_feat, ally_mask)
        enemy_public = self._enemy_public_features(enemy_feat, enemy_mask)
        self_token = self.self_encoder(self_public)
        ally_tokens = self.ally_encoder(ally_public) * ally_mask.unsqueeze(-1).float()
        public_enemy_tokens = self.public_enemy_encoder(enemy_public) * enemy_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        ally_mean = self._masked_mean(ally_tokens, ally_mask)
        enemy_mean = self._masked_mean(public_enemy_tokens, enemy_mask)
        public_query = self.public_query_encoder(th.cat([self_token, ally_mean, enemy_mean], dim=-1))

        ally_context, ally_attn = self._masked_attention(
            public_query, ally_tokens, ally_mask, self.ally_key, self.ally_value
        )
        enemy_context, enemy_attn = self._masked_attention(
            public_query, public_enemy_tokens, enemy_mask, self.enemy_key, self.enemy_value
        )
        instant = self.instant_pattern(th.cat([public_query, ally_context, enemy_context], dim=-1))
        temporal_input = th.cat([public_query, instant], dim=-1)

        batch_size, n_agents, _ = temporal_input.shape
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class PublicTransformerRelationCapturer(nn.Module):
    # Public-entity Transformer relation generator. It keeps the downstream RPG
    # decision maker unchanged and only changes how the relation condition is
    # formed from public entity tokens.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        unit_type_bits=0,
        shield_bits_ally=0,
        shield_bits_enemy=0,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        n_actions=0,
        n_allies=0,
        n_enemies=0,
        mode="baseline",
        num_heads=4,
        num_layers=1,
        use_encoded_enemy_tokens=False,
        merge_friendly_public_side=False,
        private_owner_side=False,
        private_bias_style="pair_mlp",
        semantic_router_mode=None,
        semantic_router_ema=0.99,
        semantic_router_threshold=0.5,
        semantic_router_temperature=0.1,
        semantic_router_warmup_steps=250000,
        semantic_router_freeze_steps=5000000,
    ):
        super().__init__()
        self.move_dim = move_dim
        self.own_dim = own_dim
        self.ally_feat_dim = ally_feat_dim
        self.enemy_feat_dim = enemy_feat_dim
        self.obs_last_action = bool(obs_last_action)
        self.n_actions = int(n_actions)
        self.n_allies = int(n_allies)
        self.n_enemies = int(n_enemies)
        self.relation_dim = relation_dim
        self.output_dim = output_dim
        self.unit_type_bits = unit_type_bits
        self.shield_bits_ally = shield_bits_ally
        self.shield_bits_enemy = shield_bits_enemy
        self.obs_all_health = obs_all_health
        self.obs_own_health = obs_own_health
        self.mode = mode
        self.num_heads = num_heads
        self.use_encoded_enemy_tokens = bool(use_encoded_enemy_tokens)
        self.merge_friendly_public_side = bool(merge_friendly_public_side)
        self.self_public_side_id = 1 if merge_friendly_public_side else 0
        self.private_owner_side = bool(private_owner_side)
        self.private_bias_style = private_bias_style
        self.latest_encoded_self_token = None
        self.latest_encoded_enemy_tokens = None
        self.semantic_router_mode = semantic_router_mode
        self.semantic_router_ema = float(semantic_router_ema)
        self.semantic_router_threshold = float(semantic_router_threshold)
        if not 0.0 < self.semantic_router_threshold < 1.0:
            raise ValueError("semantic_router_threshold must be strictly between 0 and 1")
        self.semantic_router_temperature = float(semantic_router_temperature)
        self.semantic_router_warmup_steps = int(semantic_router_warmup_steps)
        self.semantic_router_freeze_steps = int(semantic_router_freeze_steps)
        self.public_self_dim = 1 + unit_type_bits
        self.public_ally_dim = 1 + unit_type_bits
        self.public_enemy_dim = 1 + unit_type_bits
        self.self_value_dim = 0
        self.ally_value_dim = 0
        self.enemy_value_dim = 0
        if obs_own_health:
            self.self_value_dim = 1 + shield_bits_ally
            self.public_self_dim += self.self_value_dim
        if obs_all_health:
            self.ally_value_dim = 1 + shield_bits_ally
            self.enemy_value_dim = 1 + shield_bits_enemy
            self.public_ally_dim += self.ally_value_dim
            self.public_enemy_dim += self.enemy_value_dim

        expected_own_dim = self.self_value_dim + self.unit_type_bits
        expected_ally_dim = 4 + self.ally_value_dim + self.unit_type_bits
        expected_enemy_dim = 4 + self.enemy_value_dim + self.unit_type_bits
        if semantic_router_mode is not None and (
            self.own_dim != expected_own_dim
            or self.ally_feat_dim != expected_ally_dim
            or self.enemy_feat_dim != expected_enemy_dim
        ):
            raise ValueError(
                "Slot-level semantic routing currently requires the standard "
                "SMAC entity layout without ally last-action or timestep "
                "extras. Got own/ally/enemy dims "
                "{}/{}/{}; expected {}/{}/{}.".format(
                    self.own_dim,
                    self.ally_feat_dim,
                    self.enemy_feat_dim,
                    expected_own_dim,
                    expected_ally_dim,
                    expected_enemy_dim,
                )
            )

        (
            self.semantic_names,
            self.semantic_fields,
            manual_route,
        ) = self._build_semantic_slot_layout()
        self.register_buffer("semantic_manual_token_route", manual_route)
        self.register_buffer("semantic_token_route", manual_route.clone())
        self.register_buffer("semantic_token_probability", manual_route.clone())
        self.register_buffer("semantic_route_score", th.zeros(len(self.semantic_names)))
        self.register_buffer("semantic_route_score_initialized", th.tensor(False))
        self.register_buffer("semantic_route_frozen", th.tensor(False))
        self.register_buffer("semantic_gradient_mean", th.zeros(len(self.semantic_names)))
        self.register_buffer("semantic_gradient_abs_mean", th.zeros(len(self.semantic_names)))
        self.register_buffer("semantic_route_last_switch_rate", th.tensor(0.0))
        self.register_buffer("semantic_route_version", th.tensor(0, dtype=th.long))
        self.semantic_probe_scale = (
            nn.Parameter(th.ones(len(self.semantic_names)), requires_grad=True)
            if semantic_router_mode in {
                "gradient_importance",
                "gradient_consistency",
                "parameter_sensitivity",
            }
            else None
        )
        self.semantic_route_probe = (
            nn.Parameter(th.zeros(len(self.semantic_names)), requires_grad=True)
            if semantic_router_mode == "counterfactual"
            else None
        )
        self._semantic_online_score_sum = None
        self._semantic_online_score_count = 0
        self._semantic_forced_group = None
        self._semantic_forced_token_branch = None
        self.capture_semantic_observation_score = False

        self.cls_token = nn.Parameter(th.zeros(1, 1, 1, relation_dim))
        self.side_embedding = nn.Embedding(3, relation_dim)  # self, ally, enemy
        self.side_pair_bias = nn.Embedding(9, num_heads)
        self.private_side_embedding = nn.Embedding(3, relation_dim)

        self.self_public_encoder = self._make_encoder(self.public_self_dim)
        self.ally_public_encoder = self._make_encoder(self.public_ally_dim)
        self.enemy_public_encoder = self._make_encoder(self.public_enemy_dim)
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )

        self.self_delta_encoder = self._make_encoder(1 + self.self_value_dim)
        self.ally_delta_encoder = self._make_encoder(1 + self.ally_value_dim)
        self.enemy_delta_encoder = self._make_encoder(1 + self.enemy_value_dim)

        self.self_private_encoder = self._make_encoder(move_dim)
        self.ally_private_encoder = self._make_encoder(4)
        self.enemy_private_encoder = self._make_encoder(4)
        self.self_full_token_fuser = self._make_fuser()
        self.ally_full_token_fuser = self._make_fuser()
        self.enemy_full_token_fuser = self._make_fuser()
        self.self_full_obs_encoder = self._make_encoder(move_dim + own_dim)
        self.ally_full_obs_encoder = self._make_encoder(ally_feat_dim)
        self.enemy_full_obs_encoder = self._make_encoder(enemy_feat_dim)

        self.future_self_encoder = self._make_encoder(1 + self.self_value_dim)
        self.future_ally_encoder = self._make_encoder(1 + self.ally_value_dim)
        self.future_enemy_encoder = self._make_encoder(1 + self.enemy_value_dim)
        self.future_self_gru = nn.GRUCell(relation_dim, relation_dim)
        self.future_ally_gru = nn.GRUCell(relation_dim, relation_dim)
        self.future_enemy_gru = nn.GRUCell(relation_dim, relation_dim)
        self.future_self_decoder = nn.Linear(relation_dim, max(1, self.self_value_dim))
        self.future_ally_decoder = nn.Linear(relation_dim, max(1, self.ally_value_dim))
        self.future_enemy_decoder = nn.Linear(relation_dim, max(1, self.enemy_value_dim))

        self.bias_mlp = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, num_heads),
        )
        self.simple_bias = nn.Linear(relation_dim, num_heads)
        self.private_bias_attention = MaskedSelfAttentionBlock(relation_dim)
        self.transformer_layers = nn.ModuleList(
            BiasTransformerEncoderLayer(relation_dim, num_heads)
            for _ in range(max(1, num_layers))
        )
        self.temporal_gru = nn.GRUCell(relation_dim * 2, relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
        self.latest_aux_loss = None
        self.latest_aux_stats = {}

    def _build_semantic_slot_layout(self):
        """Describe every raw SMAC observation scalar as an independent route slot."""
        names = []
        fields = []
        manual_route = []

        def add(name, field, token_branch):
            names.append(name)
            fields.append(field)
            manual_route.append(float(bool(token_branch)))

        move_names = ("north", "south", "east", "west")
        for index in range(self.move_dim):
            suffix = move_names[index] if index < len(move_names) else str(index)
            add(f"self_move_{suffix}", "move_availability", False)

        own_index = 0
        if self.obs_own_health and own_index < self.own_dim:
            add("self_health", "health", True)
            own_index += 1
            if self.shield_bits_ally > 0 and own_index < self.own_dim:
                add("self_shield", "shield", True)
                own_index += 1
        for type_index in range(self.unit_type_bits):
            if own_index >= self.own_dim:
                break
            add(f"self_unit_type_{type_index}", "unit_type", True)
            own_index += 1
        while own_index < self.own_dim:
            add(f"self_extra_{own_index}", "self_extra", False)
            own_index += 1

        def add_entity_slots(side, count, feat_dim, value_dim, shield_bits):
            base_names = (
                ("visible", "visibility")
                if side == "ally"
                else ("attack_available", "attack_availability")
            )
            for entity_index in range(count):
                feature_index = 0
                fixed = (
                    base_names,
                    ("distance", "distance"),
                    ("relative_x", "relative_x"),
                    ("relative_y", "relative_y"),
                )
                for feature_name, field_name in fixed:
                    if feature_index >= feat_dim:
                        break
                    add(
                        f"{side}_{entity_index}_{feature_name}",
                        field_name,
                        False,
                    )
                    feature_index += 1
                if value_dim > 0 and feature_index < feat_dim:
                    add(f"{side}_{entity_index}_health", "health", True)
                    feature_index += 1
                    if shield_bits > 0 and feature_index < feat_dim:
                        add(f"{side}_{entity_index}_shield", "shield", True)
                        feature_index += 1
                for type_index in range(self.unit_type_bits):
                    if feature_index >= feat_dim:
                        break
                    add(
                        f"{side}_{entity_index}_unit_type_{type_index}",
                        "unit_type",
                        True,
                    )
                    feature_index += 1
                action_index = 0
                while feature_index < feat_dim:
                    field_name = "last_action" if self.obs_last_action else f"{side}_extra"
                    add(
                        f"{side}_{entity_index}_{field_name}_{action_index}",
                        field_name,
                        False,
                    )
                    feature_index += 1
                    action_index += 1

        add_entity_slots(
            "ally",
            self.n_allies,
            self.ally_feat_dim,
            self.ally_value_dim,
            self.shield_bits_ally,
        )
        add_entity_slots(
            "enemy",
            self.n_enemies,
            self.enemy_feat_dim,
            self.enemy_value_dim,
            self.shield_bits_enemy,
        )

        expected = (
            self.move_dim
            + self.own_dim
            + self.n_allies * self.ally_feat_dim
            + self.n_enemies * self.enemy_feat_dim
        )
        if len(names) != expected:
            raise ValueError(
                "Semantic slot layout mismatch: built {} slots for {} raw features".format(
                    len(names), expected
                )
            )
        return tuple(names), tuple(fields), th.tensor(manual_route, dtype=th.float32)

    def _flatten_semantic_slots(self, self_feat, ally_feat, enemy_feat):
        batch_size, n_agents, _ = self_feat.shape
        return th.cat(
            [
                self_feat,
                ally_feat.reshape(batch_size, n_agents, -1),
                enemy_feat.reshape(batch_size, n_agents, -1),
            ],
            dim=-1,
        )

    def _semantic_slot_views(self, values):
        offset = 0
        self_count = self.move_dim + self.own_dim
        self_values = values[offset : offset + self_count].view(1, 1, self_count)
        offset += self_count
        ally_count = self.n_allies * self.ally_feat_dim
        ally_values = values[offset : offset + ally_count].view(
            1, 1, self.n_allies, self.ally_feat_dim
        )
        offset += ally_count
        enemy_count = self.n_enemies * self.enemy_feat_dim
        enemy_values = values[offset : offset + enemy_count].view(
            1, 1, self.n_enemies, self.enemy_feat_dim
        )
        return self_values, ally_values, enemy_values

    @staticmethod
    def _centered_encode(encoder, values):
        return encoder(values) - encoder(th.zeros_like(values))

    @property
    def semantic_router_active(self):
        return self.semantic_router_mode is not None

    def semantic_router_uses_probe(self):
        return self.semantic_probe_scale is not None and not bool(
            self.semantic_route_frozen.item()
        )

    def semantic_router_needs_parameter_graph(self):
        return self.semantic_router_mode == "parameter_sensitivity" and not bool(
            self.semantic_route_frozen.item()
        )

    def semantic_router_needs_observation_score(self):
        return self.semantic_router_mode in {
            "observer_consistency",
            "temporal_stability",
        } and not bool(self.semantic_route_frozen.item())

    def semantic_router_needs_counterfactual(self):
        # Slot-level exact leave-one-out audits would require one complete
        # sequence rollout per raw observation slot. The counterfactual model
        # instead uses the signed straight-through route gradient below.
        return False

    def semantic_router_uses_route_probe(self):
        return (
            self.semantic_route_probe is not None
            and not bool(self.semantic_route_frozen.item())
        )

    def set_semantic_route_override(self, group_index, token_branch):
        self._semantic_forced_group = int(group_index)
        self._semantic_forced_token_branch = float(bool(token_branch))

    def clear_semantic_route_override(self):
        self._semantic_forced_group = None
        self._semantic_forced_token_branch = None

    def _current_semantic_token_route(self, reference):
        route = self.semantic_token_route.to(device=reference.device, dtype=reference.dtype).clone()
        if self._semantic_forced_group is not None:
            route[self._semantic_forced_group] = self._semantic_forced_token_branch
        if self.semantic_router_uses_route_probe() and th.is_grad_enabled():
            probe = self.semantic_route_probe.to(device=reference.device, dtype=reference.dtype)
            route = route + probe - probe.detach()
        return route

    def _semantic_scales(self, reference):
        if not self.semantic_router_uses_probe():
            return reference.new_ones(len(self.semantic_names))
        return self.semantic_probe_scale.to(device=reference.device, dtype=reference.dtype)

    def _accumulate_semantic_online_score(self, score):
        if not self.semantic_router_active or not th.is_grad_enabled():
            return
        score = score.detach().float()
        if self._semantic_online_score_sum is None:
            self._semantic_online_score_sum = th.zeros_like(score)
        self._semantic_online_score_sum = self._semantic_online_score_sum + score
        self._semantic_online_score_count += 1

    def _consume_semantic_online_score(self):
        if self._semantic_online_score_sum is None or self._semantic_online_score_count == 0:
            return None
        score = self._semantic_online_score_sum / float(self._semantic_online_score_count)
        self._semantic_online_score_sum = None
        self._semantic_online_score_count = 0
        return score

    def _semantic_route_probabilities(self, scores):
        temperature = max(self.semantic_router_temperature, 1e-6)
        if self.semantic_router_mode in {
            "observer_consistency",
            "temporal_stability",
            "gradient_consistency",
        }:
            # These criteria are already normalized to [0, 1].
            return scores.clamp(min=0.0, max=1.0)
        if self.semantic_router_mode in {
            "gradient_importance",
            "parameter_sensitivity",
        }:
            # Their absolute scales drift during learning. Compare every slot
            # with the current mean sensitivity without fixing how many pass.
            reference = scores.mean().clamp(min=1e-8)
            relative_score = scores / reference
            return th.sigmoid((relative_score - 1.0) / temperature)
        # Counterfactual scores are signed: positive means TOKEN is expected to
        # lower TD loss. Normalize magnitude but preserve the zero boundary.
        reference = scores.abs().mean().clamp(min=1e-8)
        return th.sigmoid(scores / (reference * temperature))

    def _apply_semantic_route_scores(self, scores, t_env):
        scores = scores.detach().to(self.semantic_route_score)
        if scores.numel() != self.semantic_route_score.numel():
            raise ValueError(
                "Semantic router produced {} slot scores for a {}-slot layout".format(
                    scores.numel(), self.semantic_route_score.numel()
                )
            )
        scores = scores.reshape_as(self.semantic_route_score)
        previous_route = self.semantic_token_route.clone()
        if not bool(self.semantic_route_score_initialized.item()):
            self.semantic_route_score.copy_(scores)
            self.semantic_route_score_initialized.fill_(True)
        else:
            self.semantic_route_score.mul_(self.semantic_router_ema).add_(
                scores, alpha=1.0 - self.semantic_router_ema
            )

        if t_env < self.semantic_router_warmup_steps:
            self.semantic_token_route.copy_(self.semantic_manual_token_route)
            self.semantic_token_probability.copy_(self.semantic_manual_token_route)
            self.semantic_route_last_switch_rate.copy_(
                (self.semantic_token_route != previous_route).float().mean()
            )
            return
        if bool(self.semantic_route_frozen.item()):
            return

        probability = self._semantic_route_probabilities(self.semantic_route_score)
        route = (probability > self.semantic_router_threshold).to(
            dtype=self.semantic_token_route.dtype
        )
        self.semantic_token_route.copy_(route)
        switch_rate = (route != previous_route).float().mean()
        self.semantic_route_last_switch_rate.copy_(switch_rate)
        if bool(switch_rate.item() > 0):
            self.semantic_route_version.add_(1)
        self.semantic_token_probability.copy_(probability)
        if t_env >= self.semantic_router_freeze_steps:
            if not bool(self.semantic_route_frozen.item()):
                self.semantic_route_version.add_(1)
            self.semantic_route_frozen.fill_(True)

    def update_semantic_router(self, t_env, external_score=None):
        if not self.semantic_router_active:
            return
        if self.semantic_router_mode in {"observer_consistency", "temporal_stability"}:
            score = self._consume_semantic_online_score()
            if score is None:
                return
        elif external_score is None:
            return
        else:
            raw_score = external_score.detach().to(self.semantic_route_score)
            if self.semantic_router_mode == "gradient_importance":
                score = raw_score.abs()
            elif self.semantic_router_mode == "gradient_consistency":
                self.semantic_gradient_mean.mul_(self.semantic_router_ema).add_(
                    raw_score, alpha=1.0 - self.semantic_router_ema
                )
                self.semantic_gradient_abs_mean.mul_(self.semantic_router_ema).add_(
                    raw_score.abs(), alpha=1.0 - self.semantic_router_ema
                )
                score = self.semantic_gradient_mean.abs() / self.semantic_gradient_abs_mean.clamp(min=1e-8)
            elif self.semantic_router_mode == "parameter_sensitivity":
                # Groups that most affect generated decision parameters receive
                # the richer token path; the remaining groups use simple bias.
                score = raw_score.abs()
            else:
                # Counterfactual score is L_bias - L_token, so positive values
                # directly favor the token branch.
                score = raw_score
        self._apply_semantic_route_scores(score, int(t_env))

    def semantic_router_stats(self):
        if not self.semantic_router_active:
            return {}
        probability = self.semantic_token_probability.clamp(min=1e-6, max=1.0 - 1e-6)
        entropy = -(
            probability * probability.log()
            + (1.0 - probability) * (1.0 - probability).log()
        ).mean()
        threshold_distance = (probability - self.semantic_router_threshold).abs()
        threshold_margin = threshold_distance.min()
        stats = {
            "semantic_route_token_count": self.semantic_token_route.sum().detach(),
            "semantic_route_bias_count": (1.0 - self.semantic_token_route).sum().detach(),
            "semantic_route_token_fraction": self.semantic_token_route.mean().detach(),
            "semantic_route_frozen": self.semantic_route_frozen.float().detach(),
            "semantic_route_switch_rate": self.semantic_route_last_switch_rate.detach(),
            "semantic_route_stability": (1.0 - self.semantic_route_last_switch_rate).detach(),
            "semantic_route_score_mean": self.semantic_route_score.mean().detach(),
            "semantic_route_score_std": self.semantic_route_score.std(unbiased=False).detach(),
            "semantic_route_score_margin": threshold_margin.detach(),
            "semantic_route_threshold": probability.new_tensor(
                self.semantic_router_threshold
            ),
            "semantic_route_threshold_margin": threshold_margin.detach(),
            "semantic_route_probability_entropy": entropy.detach(),
            "semantic_route_manual_agreement": (
                self.semantic_token_route == self.semantic_manual_token_route
            ).float().mean().detach(),
            "semantic_route_version": self.semantic_route_version.float().detach(),
        }
        for field_name in sorted(set(self.semantic_fields)):
            indices = [
                index
                for index, candidate in enumerate(self.semantic_fields)
                if candidate == field_name
            ]
            index_tensor = th.tensor(
                indices, device=self.semantic_token_route.device, dtype=th.long
            )
            stats[f"semantic_route_field_{field_name}_token_fraction"] = (
                self.semantic_token_route.index_select(0, index_tensor).mean().detach()
            )
            stats[f"semantic_route_field_{field_name}_score_mean"] = (
                self.semantic_route_score.index_select(0, index_tensor).mean().detach()
            )
        return stats

    def semantic_route_summary(self):
        token_names = [
            name
            for name, branch in zip(self.semantic_names, self.semantic_token_route.tolist())
            if branch >= 0.5
        ]
        bias_names = [
            name
            for name, branch in zip(self.semantic_names, self.semantic_token_route.tolist())
            if branch < 0.5
        ]
        return "TOKEN=[{}] | BIAS=[{}] | frozen={} | version={}".format(
            ",".join(token_names),
            ",".join(bias_names),
            int(self.semantic_route_frozen.item()),
            int(self.semantic_route_version.item()),
        )

    def copy_semantic_router_from(self, source):
        if not self.semantic_router_active or not source.semantic_router_active:
            return
        for name in (
            "semantic_token_route",
            "semantic_token_probability",
            "semantic_route_score",
            "semantic_route_score_initialized",
            "semantic_route_frozen",
            "semantic_gradient_mean",
            "semantic_gradient_abs_mean",
            "semantic_route_last_switch_rate",
            "semantic_route_version",
        ):
            getattr(self, name).copy_(getattr(source, name))

    def _make_encoder(self, input_dim):
        return nn.Sequential(
            nn.Linear(max(1, input_dim), self.relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.relation_dim, self.relation_dim),
        )

    def _make_fuser(self):
        return nn.Sequential(
            nn.Linear(2 * self.relation_dim, self.relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.relation_dim, self.relation_dim),
        )

    def _side(self, side_id, device):
        if self.private_owner_side:
            return self.side_embedding.weight.new_zeros(self.relation_dim).to(device)
        return self.side_embedding.weight[int(side_id)]

    def _own_feat(self, self_feat):
        return self_feat[:, :, self.move_dim :]

    def _self_public_features(self, self_feat):
        own_feat = self._own_feat(self_feat)
        batch_size, n_agents, _ = self_feat.shape
        features = [self_feat.new_ones(batch_size, n_agents, 1)]
        idx = 0
        if self.obs_own_health:
            features.append(own_feat[:, :, idx : idx + self.self_value_dim])
            idx += self.self_value_dim
        if self.unit_type_bits > 0:
            features.append(own_feat[:, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _ally_public_features(self, ally_feat, ally_mask):
        features = [ally_mask.unsqueeze(-1).float()]
        idx = 4
        if self.obs_all_health:
            features.append(ally_feat[:, :, :, idx : idx + self.ally_value_dim])
            idx += self.ally_value_dim
        if self.unit_type_bits > 0:
            features.append(ally_feat[:, :, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1) * ally_mask.unsqueeze(-1).float()

    def _enemy_public_features(self, enemy_feat, enemy_mask):
        features = [enemy_mask.unsqueeze(-1).float()]
        idx = 4
        if self.obs_all_health:
            features.append(enemy_feat[:, :, :, idx : idx + self.enemy_value_dim])
            idx += self.enemy_value_dim
        if self.unit_type_bits > 0:
            features.append(enemy_feat[:, :, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1) * enemy_mask.unsqueeze(-1).float()

    def _self_values(self, self_feat):
        own_feat = self._own_feat(self_feat)
        if self.self_value_dim == 0:
            return own_feat.new_zeros(own_feat.size(0), own_feat.size(1), 0)
        return own_feat[:, :, : self.self_value_dim]

    def _ally_values(self, ally_feat):
        if self.ally_value_dim == 0:
            return ally_feat.new_zeros(*ally_feat.shape[:-1], 0)
        return ally_feat[:, :, :, 4 : 4 + self.ally_value_dim]

    def _enemy_values(self, enemy_feat):
        if self.enemy_value_dim == 0:
            return enemy_feat.new_zeros(*enemy_feat.shape[:-1], 0)
        return enemy_feat[:, :, :, 4 : 4 + self.enemy_value_dim]

    def _encode_delta(self, values_delta, valid_mask, encoder):
        delta_input = th.cat(
            [valid_mask.unsqueeze(-1).float(), values_delta * valid_mask.unsqueeze(-1).float()],
            dim=-1,
        )
        return encoder(delta_input) * valid_mask.unsqueeze(-1).float()

    def _predict_next_values(self, prev_values, curr_values, prev_mask, curr_mask, encoder, gru, decoder, value_dim):
        if value_dim == 0:
            return curr_values
        prev_input = th.cat(
            [prev_mask.unsqueeze(-1).float(), prev_values * prev_mask.unsqueeze(-1).float()],
            dim=-1,
        )
        curr_input = th.cat(
            [curr_mask.unsqueeze(-1).float(), curr_values * curr_mask.unsqueeze(-1).float()],
            dim=-1,
        )
        prev_embed = encoder(prev_input)
        curr_embed = encoder(curr_input)
        flat_curr = curr_embed.reshape(-1, self.relation_dim)
        flat_prev = prev_embed.reshape(-1, self.relation_dim)
        pred_embed = gru(flat_curr, flat_prev)
        pred = decoder(pred_embed).view(*curr_values.shape[:-1], -1)[..., :value_dim]
        return pred * curr_mask.unsqueeze(-1).float()

    def _future_delta_embeddings(
        self,
        self_feat,
        ally_feat,
        enemy_feat,
        ally_mask,
        enemy_mask,
        prev_self_feat,
        prev_ally_feat,
        prev_enemy_feat,
        prev_ally_mask,
        prev_enemy_mask,
        next_self_feat,
        next_ally_feat,
        next_enemy_feat,
        next_ally_mask,
        next_enemy_mask,
        next_obs_mask,
    ):
        batch_size, n_agents, _ = self_feat.shape
        self_mask = self_feat.abs().sum(dim=-1) > 0
        prev_self_mask = prev_self_feat.abs().sum(dim=-1) > 0
        current_self_values = self._self_values(self_feat)
        prev_self_values = self._self_values(prev_self_feat)
        next_self_values = self._self_values(next_self_feat)
        current_ally_values = self._ally_values(ally_feat)
        prev_ally_values = self._ally_values(prev_ally_feat)
        next_ally_values = self._ally_values(next_ally_feat)
        current_enemy_values = self._enemy_values(enemy_feat)
        prev_enemy_values = self._enemy_values(prev_enemy_feat)
        next_enemy_values = self._enemy_values(next_enemy_feat)

        pred_self = self._predict_next_values(
            prev_self_values,
            current_self_values,
            prev_self_mask,
            self_mask,
            self.future_self_encoder,
            self.future_self_gru,
            self.future_self_decoder,
            self.self_value_dim,
        )
        pred_ally = self._predict_next_values(
            prev_ally_values,
            current_ally_values,
            prev_ally_mask,
            ally_mask,
            self.future_ally_encoder,
            self.future_ally_gru,
            self.future_ally_decoder,
            self.ally_value_dim,
        )
        pred_enemy = self._predict_next_values(
            prev_enemy_values,
            current_enemy_values,
            prev_enemy_mask,
            enemy_mask,
            self.future_enemy_encoder,
            self.future_enemy_gru,
            self.future_enemy_decoder,
            self.enemy_value_dim,
        )

        self_delta = pred_self - current_self_values
        ally_delta = pred_ally - current_ally_values
        enemy_delta = pred_enemy - current_enemy_values
        self_embed = self._encode_delta(self_delta, self_mask, self.self_delta_encoder)
        ally_embed = self._encode_delta(ally_delta, ally_mask, self.ally_delta_encoder)
        enemy_embed = self._encode_delta(enemy_delta, enemy_mask, self.enemy_delta_encoder)

        loss_parts = []
        denom_parts = []
        valid_masks = []
        valid_next = next_obs_mask.bool() if next_obs_mask is not None else self_mask.new_zeros(batch_size, n_agents)
        if self.self_value_dim > 0:
            mask = (self_mask & valid_next).unsqueeze(-1)
            loss_parts.append(F.smooth_l1_loss(pred_self, next_self_values, reduction="none") * mask.float())
            denom_parts.append(mask.float().sum())
            valid_masks.append(mask)
        if self.ally_value_dim > 0:
            mask = (ally_mask & next_ally_mask & valid_next.unsqueeze(-1)).unsqueeze(-1)
            loss_parts.append(F.smooth_l1_loss(pred_ally, next_ally_values, reduction="none") * mask.float())
            denom_parts.append(mask.float().sum())
            valid_masks.append(mask)
        if self.enemy_value_dim > 0:
            mask = (enemy_mask & next_enemy_mask & valid_next.unsqueeze(-1)).unsqueeze(-1)
            loss_parts.append(F.smooth_l1_loss(pred_enemy, next_enemy_values, reduction="none") * mask.float())
            denom_parts.append(mask.float().sum())
            valid_masks.append(mask)

        if loss_parts:
            denom = th.stack(denom_parts).sum().clamp(min=1.0)
            aux_loss = th.stack([part.sum() for part in loss_parts]).sum() / denom
            mask_frac = th.stack([mask.float().mean() for mask in valid_masks]).mean()
        else:
            aux_loss = self_feat.new_zeros(())
            mask_frac = self_feat.new_zeros(())

        def _masked_abs_mean(delta, mask):
            if delta.size(-1) == 0:
                return delta.new_zeros(())
            weight = mask.unsqueeze(-1).float()
            return (delta.abs() * weight).sum() / weight.sum().clamp(min=1.0)

        self.latest_aux_loss = aux_loss
        self.latest_aux_stats = {
            "public_future_loss_raw": aux_loss.detach(),
            "public_future_mask_frac": mask_frac.detach(),
            "public_future_delta_abs": th.stack(
                [
                    _masked_abs_mean(self_delta, self_mask),
                    _masked_abs_mean(ally_delta, ally_mask),
                    _masked_abs_mean(enemy_delta, enemy_mask),
                ]
            ).mean().detach(),
        }
        return self_embed, ally_embed, enemy_embed

    def _past_delta_embeddings(
        self,
        self_feat,
        ally_feat,
        enemy_feat,
        ally_mask,
        enemy_mask,
        prev_self_feat,
        prev_ally_feat,
        prev_enemy_feat,
        prev_ally_mask,
        prev_enemy_mask,
    ):
        self_mask = self_feat.abs().sum(dim=-1) > 0
        prev_self_mask = prev_self_feat.abs().sum(dim=-1) > 0
        self_valid = self_mask & prev_self_mask
        ally_valid = ally_mask & prev_ally_mask
        enemy_valid = enemy_mask & prev_enemy_mask
        self_delta = self._self_values(self_feat) - self._self_values(prev_self_feat)
        ally_delta = self._ally_values(ally_feat) - self._ally_values(prev_ally_feat)
        enemy_delta = self._enemy_values(enemy_feat) - self._enemy_values(prev_enemy_feat)
        return (
            self._encode_delta(self_delta, self_valid, self.self_delta_encoder),
            self._encode_delta(ally_delta, ally_valid, self.ally_delta_encoder),
            self._encode_delta(enemy_delta, enemy_valid, self.enemy_delta_encoder),
        )

    def _private_embeddings(self, self_feat, ally_feat, enemy_feat, ally_mask, enemy_mask):
        move_feat = self_feat[:, :, : self.move_dim]
        self_embed = self.self_private_encoder(move_feat)
        ally_embed = self.ally_private_encoder(ally_feat[:, :, :, :4]) * ally_mask.unsqueeze(-1).float()
        enemy_embed = self.enemy_private_encoder(enemy_feat[:, :, :, :4]) * enemy_mask.unsqueeze(-1).float()
        if self.private_owner_side:
            self_embed = self_embed + self._private_side(0, self_feat.device).view(1, 1, -1)
            ally_embed = (
                ally_embed
                + self._private_side(1, self_feat.device).view(1, 1, 1, -1)
                * ally_mask.unsqueeze(-1).float()
            )
            enemy_embed = (
                enemy_embed
                + self._private_side(2, self_feat.device).view(1, 1, 1, -1)
                * enemy_mask.unsqueeze(-1).float()
            )
        return self_embed, ally_embed, enemy_embed

    def _private_side(self, side_id, device):
        return self.private_side_embedding.weight[int(side_id)]

    def _semantic_routed_embeddings(
        self,
        self_feat,
        ally_feat,
        enemy_feat,
        ally_mask,
        enemy_mask,
    ):
        route = self._current_semantic_token_route(self_feat)
        scales = self._semantic_scales(self_feat)
        apply_probe_scales = self.semantic_router_uses_probe() and th.is_grad_enabled()
        self_route, ally_route, enemy_route = self._semantic_slot_views(route)
        self_scales, ally_scales, enemy_scales = self._semantic_slot_views(scales)
        if apply_probe_scales:
            scaled_self_feat = self_feat * self_scales
            scaled_ally_feat = ally_feat * ally_scales
            scaled_enemy_feat = enemy_feat * enemy_scales
        else:
            scaled_self_feat = self_feat
            scaled_ally_feat = ally_feat
            scaled_enemy_feat = enemy_feat

        self_public = self._self_public_features(scaled_self_feat)
        ally_public = self._ally_public_features(scaled_ally_feat, ally_mask)
        enemy_public = self._enemy_public_features(scaled_enemy_feat, enemy_mask)
        self_public_end = self.move_dim + self.self_value_dim + self.unit_type_bits
        self_public_route = th.cat(
            [
                route.new_ones(1, 1, 1),
                self_route[:, :, self.move_dim : self_public_end],
            ],
            dim=-1,
        )
        ally_public_end = 4 + self.ally_value_dim + self.unit_type_bits
        enemy_public_end = 4 + self.enemy_value_dim + self.unit_type_bits
        ally_public_route = th.cat(
            [
                route.new_ones(1, 1, self.n_allies, 1),
                ally_route[:, :, :, 4:ally_public_end],
            ],
            dim=-1,
        )
        enemy_public_route = th.cat(
            [
                route.new_ones(1, 1, self.n_enemies, 1),
                enemy_route[:, :, :, 4:enemy_public_end],
            ],
            dim=-1,
        )

        ally_encoder = self.self_public_encoder if self.merge_friendly_public_side else self.ally_public_encoder
        self_full_embed = self.self_public_encoder(self_public)
        ally_full_embed = ally_encoder(ally_public) * ally_mask.unsqueeze(-1).float()
        enemy_full_embed = self.enemy_public_encoder(enemy_public) * enemy_mask.unsqueeze(-1).float()
        self_token_base = self.self_public_encoder(self_public * self_public_route)
        ally_token_base = ally_encoder(ally_public * ally_public_route) * ally_mask.unsqueeze(-1).float()
        enemy_token_base = self.enemy_public_encoder(
            enemy_public * enemy_public_route
        ) * enemy_mask.unsqueeze(-1).float()

        self_geometry_input = scaled_self_feat[:, :, : self.move_dim]
        ally_geometry_input = scaled_ally_feat[:, :, :, :4]
        enemy_geometry_input = scaled_enemy_feat[:, :, :, :4]
        self_geometry_route = self_route[:, :, : self.move_dim]
        ally_geometry_route = ally_route[:, :, :, :4]
        enemy_geometry_route = enemy_route[:, :, :, :4]
        self_geometry_token = self._centered_encode(
            self.self_private_encoder,
            self_geometry_input * self_geometry_route,
        )
        ally_geometry_token = self._centered_encode(
            self.ally_private_encoder,
            ally_geometry_input * ally_geometry_route,
        ) * ally_mask.unsqueeze(-1).float()
        enemy_geometry_token = self._centered_encode(
            self.enemy_private_encoder,
            enemy_geometry_input * enemy_geometry_route,
        ) * enemy_mask.unsqueeze(-1).float()

        self_token = self_token_base + self_geometry_token + self._side(
            self.self_public_side_id, self_feat.device
        ).view(1, 1, -1)
        ally_tokens = ally_token_base + ally_geometry_token + self._side(
            1, self_feat.device
        ).view(1, 1, 1, -1)
        enemy_tokens = enemy_token_base + enemy_geometry_token + self._side(
            2, self_feat.device
        ).view(1, 1, 1, -1)
        ally_tokens = ally_tokens * ally_mask.unsqueeze(-1).float()
        enemy_tokens = enemy_tokens * enemy_mask.unsqueeze(-1).float()
        self_owner = self._private_side(0, self_feat.device).view(1, 1, -1)
        ally_owner = (
            self._private_side(1, self_feat.device).view(1, 1, 1, -1)
            * ally_mask.unsqueeze(-1).float()
        )
        enemy_owner = (
            self._private_side(2, self_feat.device).view(1, 1, 1, -1)
            * enemy_mask.unsqueeze(-1).float()
        )
        self_geometry_bias = self.self_private_encoder(
            self_geometry_input * (1.0 - self_geometry_route)
        )
        ally_geometry_bias = self.ally_private_encoder(
            ally_geometry_input * (1.0 - ally_geometry_route)
        ) * ally_mask.unsqueeze(-1).float()
        enemy_geometry_bias = self.enemy_private_encoder(
            enemy_geometry_input * (1.0 - enemy_geometry_route)
        ) * enemy_mask.unsqueeze(-1).float()

        # Public fields omitted from the entity token are converted into the
        # same simple attention-bias representation as the private fields.
        bias_self = self_full_embed - self_token_base
        bias_ally = ally_full_embed - ally_token_base
        bias_enemy = enemy_full_embed - enemy_token_base
        bias_self = bias_self + self_geometry_bias + self_owner
        bias_ally = bias_ally + ally_geometry_bias + ally_owner
        bias_enemy = bias_enemy + enemy_geometry_bias + enemy_owner
        return self_token, ally_tokens, enemy_tokens, bias_self, bias_ally, bias_enemy

    def _collect_semantic_observation_score(
        self,
        self_feat,
        ally_feat,
        enemy_feat,
        prev_self_feat,
        prev_ally_feat,
        prev_enemy_feat,
    ):
        if self.semantic_router_mode not in {"observer_consistency", "temporal_stability"}:
            return
        # Rollout and target-network forwards run under no_grad and cannot
        # contribute to the learner-side routing estimate. Once the route is
        # frozen, collecting further statistics is also redundant.
        if (
            not th.is_grad_enabled()
            or not self.capture_semantic_observation_score
            or bool(self.semantic_route_frozen.item())
        ):
            return
        current = self._flatten_semantic_slots(self_feat, ally_feat, enemy_feat)
        if self.semantic_router_mode == "observer_consistency":
            centered = current - current.mean(dim=1, keepdim=True)
            dispersion = centered.pow(2).mean(dim=(0, 1))
            scale = current.pow(2).mean(dim=(0, 1)).clamp(min=1e-8)
            score = (1.0 - dispersion / scale).clamp(min=0.0, max=1.0)
        else:
            previous = self._flatten_semantic_slots(
                prev_self_feat, prev_ally_feat, prev_enemy_feat
            )
            change = (current - previous).abs().mean(dim=(0, 1))
            scale = (
                current.abs().mean(dim=(0, 1))
                + previous.abs().mean(dim=(0, 1))
            ).clamp(min=1e-8)
            score = (1.0 - change / scale).clamp(min=0.0, max=1.0)
        self._accumulate_semantic_online_score(score)

    def _simple_private_bias(self, mod_tokens, batch_size, n_agents):
        n_entity_tokens = mod_tokens.size(2)
        token_bias = self.simple_bias(mod_tokens)
        pair_bias = token_bias.unsqueeze(2).expand(-1, -1, n_entity_tokens, -1, -1)
        full_bias = mod_tokens.new_zeros(batch_size, n_agents, n_entity_tokens + 1, n_entity_tokens + 1, self.num_heads)
        full_bias[:, :, 1:, 1:] = pair_bias
        return full_bias.permute(0, 1, 4, 2, 3).reshape(
            batch_size * n_agents, self.num_heads, n_entity_tokens + 1, n_entity_tokens + 1
        )

    def _build_attention_bias(self, mod_self, mod_ally, mod_enemy, batch_size, n_agents, ally_mask, enemy_mask):
        mod_tokens = th.cat([mod_self.unsqueeze(2), mod_ally, mod_enemy], dim=2)
        n_entity_tokens = mod_tokens.size(2)
        if self.private_bias_style == "selfattn_simple":
            self_mask = th.ones(batch_size, n_agents, 1, device=mod_tokens.device, dtype=th.bool)
            private_mask = th.cat([self_mask, ally_mask, enemy_mask], dim=2)
            mod_tokens, _ = self.private_bias_attention(mod_tokens, private_mask)
            return self._simple_private_bias(mod_tokens, batch_size, n_agents)
        if self.private_bias_style == "simple":
            return self._simple_private_bias(mod_tokens, batch_size, n_agents)

        left = mod_tokens.unsqueeze(3).expand(-1, -1, -1, n_entity_tokens, -1)
        right = mod_tokens.unsqueeze(2).expand(-1, -1, n_entity_tokens, -1, -1)
        pair = th.cat([left, right], dim=-1)
        pair_bias = self.bias_mlp(pair)

        if self.private_bias_style != "pair_mlp_no_side":
            device = mod_tokens.device
            side_ids = th.cat(
                [
                    th.full((1,), self.self_public_side_id, device=device, dtype=th.long),
                    th.ones(mod_ally.size(2), device=device, dtype=th.long),
                    th.full((mod_enemy.size(2),), 2, device=device, dtype=th.long),
                ],
                dim=0,
            )
            side_pair = side_ids.unsqueeze(1) * 3 + side_ids.unsqueeze(0)
            pair_bias = pair_bias + self.side_pair_bias(side_pair).view(1, 1, n_entity_tokens, n_entity_tokens, -1)

        full_bias = mod_tokens.new_zeros(batch_size, n_agents, n_entity_tokens + 1, n_entity_tokens + 1, self.num_heads)
        full_bias[:, :, 1:, 1:] = pair_bias
        return full_bias.permute(0, 1, 4, 2, 3).reshape(
            batch_size * n_agents, self.num_heads, n_entity_tokens + 1, n_entity_tokens + 1
        )

    def _masked_uniform_attention(self, mask):
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        return mask.float() / denom

    def forward(
        self,
        self_feat,
        ally_feat,
        enemy_feat,
        prev_relation_hidden,
        prev_self_feat=None,
        prev_ally_feat=None,
        prev_enemy_feat=None,
        next_self_feat=None,
        next_ally_feat=None,
        next_enemy_feat=None,
        next_obs_mask=None,
    ):
        self.latest_aux_loss = None
        self.latest_aux_stats = {}
        batch_size, n_agents, _ = self_feat.shape
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        if prev_self_feat is None:
            prev_self_feat = th.zeros_like(self_feat)
            prev_ally_feat = th.zeros_like(ally_feat)
            prev_enemy_feat = th.zeros_like(enemy_feat)
        if next_self_feat is None:
            next_self_feat = th.zeros_like(self_feat)
            next_ally_feat = th.zeros_like(ally_feat)
            next_enemy_feat = th.zeros_like(enemy_feat)
        prev_ally_mask = prev_ally_feat.abs().sum(dim=-1) > 0
        prev_enemy_mask = prev_enemy_feat.abs().sum(dim=-1) > 0
        next_ally_mask = next_ally_feat.abs().sum(dim=-1) > 0
        next_enemy_mask = next_enemy_feat.abs().sum(dim=-1) > 0

        self._collect_semantic_observation_score(
            self_feat,
            ally_feat,
            enemy_feat,
            prev_self_feat,
            prev_ally_feat,
            prev_enemy_feat,
        )

        if self.mode == "full_obs":
            self_token = self.self_full_obs_encoder(self_feat) + self._side(0, self_feat.device).view(1, 1, -1)
            ally_tokens = self.ally_full_obs_encoder(ally_feat) + self._side(1, self_feat.device).view(1, 1, 1, -1)
            enemy_public_tokens = self.enemy_full_obs_encoder(enemy_feat) + self._side(2, self_feat.device).view(
                1, 1, 1, -1
            )
            ally_tokens = ally_tokens * ally_mask.unsqueeze(-1).float()
            enemy_public_tokens = enemy_public_tokens * enemy_mask.unsqueeze(-1).float()
        elif self.semantic_router_active:
            (
                self_token,
                ally_tokens,
                enemy_public_tokens,
                bias_self,
                bias_ally,
                bias_enemy,
            ) = self._semantic_routed_embeddings(
                self_feat,
                ally_feat,
                enemy_feat,
                ally_mask,
                enemy_mask,
            )
        else:
            self_public = self._self_public_features(self_feat)
            ally_public = self._ally_public_features(ally_feat, ally_mask)
            enemy_public = self._enemy_public_features(enemy_feat, enemy_mask)
            self_token = self.self_public_encoder(self_public) + self._side(
                self.self_public_side_id, self_feat.device
            ).view(1, 1, -1)
            ally_encoder = self.self_public_encoder if self.merge_friendly_public_side else self.ally_public_encoder
            ally_tokens = ally_encoder(ally_public) + self._side(1, self_feat.device).view(1, 1, 1, -1)
            enemy_public_tokens = self.enemy_public_encoder(enemy_public) + self._side(2, self_feat.device).view(
                1, 1, 1, -1
            )
            ally_tokens = ally_tokens * ally_mask.unsqueeze(-1).float()
            enemy_public_tokens = enemy_public_tokens * enemy_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        token_self = token_ally = token_enemy = None
        if not self.semantic_router_active:
            bias_self = bias_ally = bias_enemy = None
        if self.mode in {"future_delta_token", "future_delta_bias"}:
            mod_self, mod_ally, mod_enemy = self._future_delta_embeddings(
                self_feat,
                ally_feat,
                enemy_feat,
                ally_mask,
                enemy_mask,
                prev_self_feat,
                prev_ally_feat,
                prev_enemy_feat,
                prev_ally_mask,
                prev_enemy_mask,
                next_self_feat,
                next_ally_feat,
                next_enemy_feat,
                next_ally_mask,
                next_enemy_mask,
                next_obs_mask,
            )
            if self.mode == "future_delta_token":
                token_self, token_ally, token_enemy = mod_self, mod_ally, mod_enemy
            else:
                bias_self, bias_ally, bias_enemy = mod_self, mod_ally, mod_enemy
        elif self.mode in {"past_delta_token", "past_delta_bias"}:
            mod_self, mod_ally, mod_enemy = self._past_delta_embeddings(
                self_feat,
                ally_feat,
                enemy_feat,
                ally_mask,
                enemy_mask,
                prev_self_feat,
                prev_ally_feat,
                prev_enemy_feat,
                prev_ally_mask,
                prev_enemy_mask,
            )
            if self.mode == "past_delta_token":
                token_self, token_ally, token_enemy = mod_self, mod_ally, mod_enemy
            else:
                bias_self, bias_ally, bias_enemy = mod_self, mod_ally, mod_enemy
        elif self.mode in {"private_token", "private_bias"} and not self.semantic_router_active:
            mod_self, mod_ally, mod_enemy = self._private_embeddings(
                self_feat, ally_feat, enemy_feat, ally_mask, enemy_mask
            )
            if self.mode == "private_token":
                token_self, token_ally, token_enemy = mod_self, mod_ally, mod_enemy
            else:
                bias_self, bias_ally, bias_enemy = mod_self, mod_ally, mod_enemy
        elif self.mode == "private_bias_past_delta_token":
            token_self, token_ally, token_enemy = self._past_delta_embeddings(
                self_feat,
                ally_feat,
                enemy_feat,
                ally_mask,
                enemy_mask,
                prev_self_feat,
                prev_ally_feat,
                prev_enemy_feat,
                prev_ally_mask,
                prev_enemy_mask,
            )
            bias_self, bias_ally, bias_enemy = self._private_embeddings(
                self_feat, ally_feat, enemy_feat, ally_mask, enemy_mask
            )
        elif self.mode == "private_token_past_delta_bias":
            token_self, token_ally, token_enemy = self._private_embeddings(
                self_feat, ally_feat, enemy_feat, ally_mask, enemy_mask
            )
            bias_self, bias_ally, bias_enemy = self._past_delta_embeddings(
                self_feat,
                ally_feat,
                enemy_feat,
                ally_mask,
                enemy_mask,
                prev_self_feat,
                prev_ally_feat,
                prev_enemy_feat,
                prev_ally_mask,
                prev_enemy_mask,
            )

        if self.mode == "public_private_full_token":
            full_self, full_ally, full_enemy = self._private_embeddings(
                self_feat, ally_feat, enemy_feat, ally_mask, enemy_mask
            )
            self_token = self.self_full_token_fuser(th.cat([self_token, full_self], dim=-1))
            ally_tokens = self.ally_full_token_fuser(th.cat([ally_tokens, full_ally], dim=-1))
            enemy_public_tokens = self.enemy_full_token_fuser(th.cat([enemy_public_tokens, full_enemy], dim=-1))
            ally_tokens = ally_tokens * ally_mask.unsqueeze(-1).float()
            enemy_public_tokens = enemy_public_tokens * enemy_mask.unsqueeze(-1).float()

        if token_self is not None:
            self_token = self_token + token_self
            ally_tokens = ally_tokens + token_ally
            enemy_public_tokens = enemy_public_tokens + token_enemy

        cls_token = self.cls_token.expand(batch_size, n_agents, -1, -1)
        tokens = th.cat([cls_token, self_token.unsqueeze(2), ally_tokens, enemy_public_tokens], dim=2)
        cls_mask = th.ones(batch_size, n_agents, 1, device=self_feat.device, dtype=th.bool)
        self_mask = th.ones(batch_size, n_agents, 1, device=self_feat.device, dtype=th.bool)
        token_mask = th.cat([cls_mask, self_mask, ally_mask, enemy_mask], dim=2)

        attn_bias = None
        if bias_self is not None:
            attn_bias = self._build_attention_bias(
                bias_self, bias_ally, bias_enemy, batch_size, n_agents, ally_mask, enemy_mask
            )

        flat_tokens = tokens.reshape(batch_size * n_agents, tokens.size(2), self.relation_dim)
        flat_mask = token_mask.reshape(batch_size * n_agents, token_mask.size(2))
        for layer in self.transformer_layers:
            flat_tokens = layer(flat_tokens, flat_mask, attn_bias=attn_bias)
        encoded = flat_tokens.view(batch_size, n_agents, tokens.size(2), self.relation_dim)

        cls_out = encoded[:, :, 0]
        self_out = encoded[:, :, 1]
        self.latest_encoded_self_token = self_out
        temporal_input = th.cat([self_out, cls_out], dim=-1)
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)
        ally_attn = self._masked_uniform_attention(ally_mask)
        enemy_attn = self._masked_uniform_attention(enemy_mask)
        if self.use_encoded_enemy_tokens:
            enemy_start = 2 + ally_tokens.size(2)
            enemy_tokens = encoded[:, :, enemy_start:] * enemy_mask.unsqueeze(-1).float()
        self.latest_encoded_enemy_tokens = enemy_tokens
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class GRFPublicPrivateBiasTransformerCapturer(nn.Module):
    # GRF compact observations expose fixed semantic blocks rather than SMAC
    # visibility slots. This capturer builds public entity tokens and lets local
    # observer-dependent geometry modulate attention through a private bias.
    def __init__(
        self,
        n_agents,
        relation_dim,
        output_dim,
        num_heads=4,
        num_layers=1,
        use_absolute_public=False,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.relation_dim = relation_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_absolute_public = use_absolute_public
        self.n_opponents = 2
        self.expected_obs_dim = 4 * n_agents + 14

        self.cls_token = nn.Parameter(th.zeros(1, 1, relation_dim))
        self.side_embedding = nn.Embedding(4, relation_dim)  # self, ally, opponent, ball
        self.self_encoder = self._make_encoder(4)
        self.ally_encoder = self._make_encoder(4)
        self.opponent_encoder = self._make_encoder(4)
        self.ball_encoder = self._make_encoder(6)
        self.private_encoder = self._make_encoder(4)
        self.private_bias = nn.Sequential(
            nn.Linear(2 * relation_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, num_heads),
        )
        self.transformer_layers = nn.ModuleList(
            BiasTransformerEncoderLayer(relation_dim, num_heads)
            for _ in range(max(1, num_layers))
        )
        self.temporal_gru = nn.GRUCell(relation_dim * 2, relation_dim)
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def _make_encoder(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.relation_dim, self.relation_dim),
        )

    def _split_obs(self, obs):
        if obs.size(-1) < self.expected_obs_dim:
            raise ValueError(
                "GRF private-bias public transformer expected obs_dim >= {}, got {}.".format(
                    self.expected_obs_dim, obs.size(-1)
                )
            )
        obs = obs[..., : self.expected_obs_dim]
        batch_size, n_agents, _ = obs.shape
        if n_agents != self.n_agents:
            raise ValueError("Expected {} agents, got {}.".format(self.n_agents, n_agents))

        idx = 0
        self_pos = obs[:, :, idx : idx + 2]
        idx += 2
        ally_pos = obs[:, :, idx : idx + 2 * (self.n_agents - 1)].reshape(
            batch_size, n_agents, self.n_agents - 1, 2
        )
        idx += 2 * (self.n_agents - 1)
        self_dir = obs[:, :, idx : idx + 2]
        idx += 2
        ally_dir = obs[:, :, idx : idx + 2 * (self.n_agents - 1)].reshape(
            batch_size, n_agents, self.n_agents - 1, 2
        )
        idx += 2 * (self.n_agents - 1)
        opponent_pos = obs[:, :, idx : idx + 2 * self.n_opponents].reshape(
            batch_size, n_agents, self.n_opponents, 2
        )
        idx += 2 * self.n_opponents
        opponent_dir = obs[:, :, idx : idx + 2 * self.n_opponents].reshape(
            batch_size, n_agents, self.n_opponents, 2
        )
        idx += 2 * self.n_opponents
        ball = obs[:, :, idx : idx + 6]

        return self_pos, ally_pos, self_dir, ally_dir, opponent_pos, opponent_dir, ball

    def _build_public_features(self, self_pos, ally_pos, self_dir, ally_dir, opponent_pos, opponent_dir, ball):
        if not self.use_absolute_public:
            return self_pos, ally_pos, self_dir, ally_dir, opponent_pos, opponent_dir, ball

        ego_pos = self_pos.unsqueeze(2)
        ally_abs_pos = ally_pos + ego_pos
        opponent_abs_pos = opponent_pos + ego_pos
        ball_abs = th.cat([ball[:, :, :2] + self_pos, ball[:, :, 2:]], dim=-1)
        return self_pos, ally_abs_pos, self_dir, ally_dir, opponent_abs_pos, opponent_dir, ball_abs

    def _build_private_features(self, self_pos, ally_pos, self_dir, ally_dir, opponent_pos, opponent_dir, ball):
        if self.use_absolute_public:
            self_private_pos = th.zeros_like(self_pos)
        else:
            self_private_pos = self_pos

        ball_private = ball[:, :, :4].unsqueeze(2)
        return th.cat(
            [
                th.cat([self_private_pos, self_dir], dim=-1).unsqueeze(2),
                th.cat([ally_pos, ally_dir], dim=-1),
                th.cat([opponent_pos, opponent_dir], dim=-1),
                ball_private,
            ],
            dim=2,
        )

    def _build_private_bias(self, entity_private, token_count):
        batch_size, n_agents, n_entities, _ = entity_private.shape
        private_tokens = self.private_encoder(entity_private)
        left = private_tokens.unsqueeze(3).expand(-1, -1, -1, n_entities, -1)
        right = private_tokens.unsqueeze(2).expand(-1, -1, n_entities, -1, -1)
        pair = th.cat([left, right], dim=-1)
        pair_bias = self.private_bias(pair).permute(0, 1, 4, 2, 3)

        bias = entity_private.new_zeros(batch_size, n_agents, self.num_heads, token_count, token_count)
        bias[:, :, :, 1:, 1:] = pair_bias
        return bias.reshape(batch_size * n_agents, self.num_heads, token_count, token_count)

    def forward(self, obs, prev_relation_hidden):
        batch_size, n_agents, _ = obs.shape
        (
            self_pos,
            ally_pos,
            self_dir,
            ally_dir,
            opponent_pos,
            opponent_dir,
            ball,
        ) = self._split_obs(obs)
        (
            public_self_pos,
            public_ally_pos,
            public_self_dir,
            public_ally_dir,
            public_opponent_pos,
            public_opponent_dir,
            public_ball,
        ) = self._build_public_features(
            self_pos, ally_pos, self_dir, ally_dir, opponent_pos, opponent_dir, ball
        )

        self_token = self.self_encoder(th.cat([public_self_pos, public_self_dir], dim=-1))
        ally_token = self.ally_encoder(th.cat([public_ally_pos, public_ally_dir], dim=-1))
        opponent_token = self.opponent_encoder(th.cat([public_opponent_pos, public_opponent_dir], dim=-1))
        ball_token = self.ball_encoder(public_ball).unsqueeze(2)

        self_side = self.side_embedding.weight[0].view(1, 1, 1, -1)
        ally_side = self.side_embedding.weight[1].view(1, 1, 1, -1)
        opponent_side = self.side_embedding.weight[2].view(1, 1, 1, -1)
        ball_side = self.side_embedding.weight[3].view(1, 1, 1, -1)

        entity_tokens = th.cat(
            [
                self_token.unsqueeze(2) + self_side,
                ally_token + ally_side,
                opponent_token + opponent_side,
                ball_token + ball_side,
            ],
            dim=2,
        )
        token_count = entity_tokens.size(2) + 1
        cls = self.cls_token.expand(batch_size, n_agents, -1).unsqueeze(2)
        tokens = th.cat([cls, entity_tokens], dim=2)

        entity_private = self._build_private_features(
            self_pos, ally_pos, self_dir, ally_dir, opponent_pos, opponent_dir, ball
        )
        attn_bias = self._build_private_bias(entity_private, token_count)

        flat_tokens = tokens.reshape(batch_size * n_agents, token_count, self.relation_dim)
        key_mask = th.ones(batch_size * n_agents, token_count, dtype=th.bool, device=obs.device)
        for layer in self.transformer_layers:
            flat_tokens = layer(flat_tokens, key_mask, attn_bias=attn_bias)

        context_token = flat_tokens[:, 0].reshape(batch_size, n_agents, self.relation_dim)
        encoded_entities = flat_tokens[:, 1:].reshape(
            batch_size, n_agents, entity_tokens.size(2), self.relation_dim
        )
        ally_start = 1
        opponent_start = ally_start + (self.n_agents - 1)
        ball_index = opponent_start + self.n_opponents
        self.latest_self_token = encoded_entities[:, :, 0]
        self.latest_ally_tokens = encoded_entities[:, :, ally_start:opponent_start]
        self.latest_opponent_tokens = encoded_entities[:, :, opponent_start:ball_index]
        self.latest_ball_token = encoded_entities[:, :, ball_index]
        self.latest_context_token = context_token
        if prev_relation_hidden is None:
            prev_relation_hidden = context_token.new_zeros(batch_size, n_agents, self.relation_dim)
        prev_flat = prev_relation_hidden.reshape(batch_size * n_agents, self.relation_dim)
        gru_input = th.cat([context_token.reshape(batch_size * n_agents, -1), prev_flat], dim=-1)
        next_relation_hidden = self.temporal_gru(gru_input, prev_flat).reshape(
            batch_size, n_agents, self.relation_dim
        )
        condition = self.output_encoder(next_relation_hidden)
        return condition, next_relation_hidden


class MaskedSelfAttentionBlock(nn.Module):
    def __init__(self, relation_dim):
        super().__init__()
        self.relation_dim = relation_dim
        self.query = nn.Linear(relation_dim, relation_dim)
        self.key = nn.Linear(relation_dim, relation_dim)
        self.value = nn.Linear(relation_dim, relation_dim)
        self.out = nn.Linear(relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, tokens, token_mask):
        batch_size, n_agents, n_tokens, _ = tokens.shape
        flat_tokens = tokens.reshape(batch_size * n_agents, n_tokens, self.relation_dim)
        flat_mask = token_mask.reshape(batch_size * n_agents, n_tokens).bool()
        valid_any = flat_mask.any(dim=-1, keepdim=True).unsqueeze(-1)

        query = self.query(flat_tokens)
        key = self.key(flat_tokens)
        value = self.value(flat_tokens)
        logits = th.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.relation_dim))
        logits = logits.masked_fill(~flat_mask.unsqueeze(1), _neg_inf_like(logits))
        attn = F.softmax(logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))

        context = th.matmul(attn, value)
        updated = self.norm(flat_tokens + F.elu(self.out(context), inplace=True))
        updated = updated * flat_mask.unsqueeze(-1).float()
        return (
            updated.view(batch_size, n_agents, n_tokens, self.relation_dim),
            attn.view(batch_size, n_agents, n_tokens, n_tokens),
        )


class SemanticSelfAttentionRelationCapturer(nn.Module):
    # Feature-level semantic relation generator. Each observation field is
    # encoded through its own semantic path, then owner embeddings preserve
    # whether the value came from self, an ally, or an enemy.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        unit_type_bits=0,
        shield_bits_ally=0,
        shield_bits_enemy=0,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_timestep_number=False,
        n_actions=0,
    ):
        super().__init__()
        self.move_dim = move_dim
        self.own_dim = own_dim
        self.ally_feat_dim = ally_feat_dim
        self.enemy_feat_dim = enemy_feat_dim
        self.relation_dim = relation_dim
        self.unit_type_bits = unit_type_bits
        self.shield_bits_ally = shield_bits_ally
        self.shield_bits_enemy = shield_bits_enemy
        self.obs_all_health = obs_all_health
        self.obs_own_health = obs_own_health
        self.obs_last_action = obs_last_action
        self.obs_timestep_number = obs_timestep_number
        self.n_actions = n_actions

        self.owner_embedding = nn.Embedding(3, relation_dim)  # self, ally, enemy
        self.move_encoder = self._make_encoder(move_dim)
        self.health_encoder = self._make_encoder(1)
        self.shield_encoder = self._make_encoder(1)
        self.spatial_encoder = self._make_encoder(3)
        self.interaction_encoder = self._make_encoder(1)
        self.type_encoder = self._make_encoder(max(1, unit_type_bits))
        self.last_action_encoder = self._make_encoder(max(1, n_actions))
        self.self_attention = MaskedSelfAttentionBlock(relation_dim)

        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
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

    def _make_encoder(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.relation_dim, self.relation_dim),
        )

    def _owner(self, owner_id, device):
        return self.owner_embedding(th.tensor(owner_id, device=device, dtype=th.long)).view(1, 1, 1, -1)

    def _entity_indicator(self, batch_size, n_agents, n_entities, device):
        eye = th.eye(n_entities, device=device).view(1, 1, n_entities, n_entities)
        return eye.expand(batch_size, n_agents, -1, -1)

    def _parse_own(self, own_feat):
        idx = 0
        health = shield = unit_type = timestep = None
        if self.obs_own_health:
            health = own_feat[:, :, idx : idx + 1]
            idx += 1
            if self.shield_bits_ally > 0:
                shield = own_feat[:, :, idx : idx + 1]
                idx += 1
        if self.unit_type_bits > 0:
            unit_type = own_feat[:, :, idx : idx + self.unit_type_bits]
            idx += self.unit_type_bits
        if self.obs_timestep_number and idx < own_feat.size(-1):
            timestep = own_feat[:, :, idx : idx + 1]
        return health, shield, unit_type, timestep

    def _parse_enemy(self, enemy_feat):
        idx = 0
        available = enemy_feat[:, :, :, idx : idx + 1]
        idx += 1
        spatial = enemy_feat[:, :, :, idx : idx + 3]
        idx += 3
        health = shield = unit_type = None
        if self.obs_all_health:
            health = enemy_feat[:, :, :, idx : idx + 1]
            idx += 1
            if self.shield_bits_enemy > 0:
                shield = enemy_feat[:, :, :, idx : idx + 1]
                idx += 1
        if self.unit_type_bits > 0:
            unit_type = enemy_feat[:, :, :, idx : idx + self.unit_type_bits]
        return available, spatial, health, shield, unit_type

    def _parse_ally(self, ally_feat):
        idx = 0
        visible = ally_feat[:, :, :, idx : idx + 1]
        idx += 1
        spatial = ally_feat[:, :, :, idx : idx + 3]
        idx += 3
        health = shield = unit_type = last_action = None
        if self.obs_all_health:
            health = ally_feat[:, :, :, idx : idx + 1]
            idx += 1
            if self.shield_bits_ally > 0:
                shield = ally_feat[:, :, :, idx : idx + 1]
                idx += 1
        if self.unit_type_bits > 0:
            unit_type = ally_feat[:, :, :, idx : idx + self.unit_type_bits]
            idx += self.unit_type_bits
        if self.obs_last_action and self.n_actions > 0 and idx < ally_feat.size(-1):
            last_action = ally_feat[:, :, :, idx : idx + self.n_actions]
        return visible, spatial, health, shield, unit_type, last_action

    def _pool_by_mark(self, updated_tokens, token_mask, mark):
        if mark.size(-1) == 0:
            return updated_tokens.new_zeros(updated_tokens.size(0), updated_tokens.size(1), self.relation_dim)
        token_weight = token_mask.unsqueeze(-1).unsqueeze(-1).float()
        weighted = (updated_tokens.unsqueeze(-2) * mark.unsqueeze(-1) * token_weight).sum(dim=2)
        denom = (mark * token_mask.unsqueeze(-1).float()).sum(dim=2).clamp(min=1.0)
        pooled = weighted / denom.unsqueeze(-1)
        entity_valid = denom > 0
        count = entity_valid.sum(dim=-1, keepdim=True).clamp(min=1).float()
        return (pooled * entity_valid.unsqueeze(-1).float()).sum(dim=2) / count

    def _aggregate_entity_attention(self, pair_attn, token_mask, query_mark, entity_mark):
        if entity_mark.size(-1) == 0:
            return pair_attn.new_zeros(pair_attn.size(0), pair_attn.size(1), 0)
        query_weight = query_mark.squeeze(-1) * token_mask.float()
        query_weight = query_weight / query_weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
        token_attn = (pair_attn * query_weight.unsqueeze(-1)).sum(dim=2)
        return (token_attn.unsqueeze(-1) * entity_mark).sum(dim=2)

    def _build_semantic_tokens(self, move_feat, own_feat, ally_feat, enemy_feat):
        batch_size, n_agents, _ = move_feat.shape
        n_allies = ally_feat.size(2)
        n_enemies = enemy_feat.size(2)
        device = move_feat.device

        tokens, masks, self_marks, ally_marks, enemy_marks = [], [], [], [], []
        self_mask = (move_feat.abs().sum(dim=-1, keepdim=True) + own_feat.abs().sum(dim=-1, keepdim=True)) > 0
        self_mark = th.ones(batch_size, n_agents, 1, 1, device=device)

        self_token = self.move_encoder(move_feat).unsqueeze(2)
        self_token_mask = self_mask
        self_ally_mark = th.zeros(batch_size, n_agents, 1, n_allies, device=device)
        self_enemy_mark = th.zeros(batch_size, n_agents, 1, n_enemies, device=device)
        self_marks.append(self_mark)
        ally_marks.append(self_ally_mark)
        enemy_marks.append(self_enemy_mark)
        tokens.append(self_token + self._owner(0, device))
        masks.append(self_token_mask.bool())

        own_health, own_shield, own_type, own_timestep = self._parse_own(own_feat)
        for value, encoder in (
            (own_health, self.health_encoder),
            (own_shield, self.shield_encoder),
            (own_type, self.type_encoder),
            (own_timestep, self.interaction_encoder),
        ):
            if value is not None:
                encoded = encoder(value if value.size(-1) > 0 else value.new_zeros(*value.shape[:-1], 1)).unsqueeze(2)
                tokens.append(encoded + self._owner(0, device))
                masks.append(self_mask.bool())
                self_marks.append(self_mark)
                ally_marks.append(self_ally_mark)
                enemy_marks.append(self_enemy_mark)

        ally_visible, ally_spatial, ally_health, ally_shield, ally_type, ally_last_action = self._parse_ally(ally_feat)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        ally_indicator = self._entity_indicator(batch_size, n_agents, n_allies, device)
        for value, encoder in (
            (ally_visible, self.interaction_encoder),
            (ally_spatial, self.spatial_encoder),
            (ally_health, self.health_encoder),
            (ally_shield, self.shield_encoder),
            (ally_type, self.type_encoder),
            (ally_last_action, self.last_action_encoder),
        ):
            if value is not None:
                encoded = encoder(value if value.size(-1) > 0 else value.new_zeros(*value.shape[:-1], 1))
                tokens.append(encoded + self._owner(1, device))
                masks.append(ally_mask.bool())
                self_marks.append(th.zeros(batch_size, n_agents, n_allies, 1, device=device))
                ally_marks.append(ally_indicator)
                enemy_marks.append(th.zeros(batch_size, n_agents, n_allies, n_enemies, device=device))

        enemy_available, enemy_spatial, enemy_health, enemy_shield, enemy_type = self._parse_enemy(enemy_feat)
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        enemy_indicator = self._entity_indicator(batch_size, n_agents, n_enemies, device)
        for value, encoder in (
            (enemy_available, self.interaction_encoder),
            (enemy_spatial, self.spatial_encoder),
            (enemy_health, self.health_encoder),
            (enemy_shield, self.shield_encoder),
            (enemy_type, self.type_encoder),
        ):
            if value is not None:
                encoded = encoder(value if value.size(-1) > 0 else value.new_zeros(*value.shape[:-1], 1))
                tokens.append(encoded + self._owner(2, device))
                masks.append(enemy_mask.bool())
                self_marks.append(th.zeros(batch_size, n_agents, n_enemies, 1, device=device))
                ally_marks.append(th.zeros(batch_size, n_agents, n_enemies, n_allies, device=device))
                enemy_marks.append(enemy_indicator)

        return (
            th.cat(tokens, dim=2),
            th.cat(masks, dim=2),
            th.cat(self_marks, dim=2),
            th.cat(ally_marks, dim=2),
            th.cat(enemy_marks, dim=2),
        )

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        move_feat = self_feat[:, :, : self.move_dim]
        own_feat = self_feat[:, :, self.move_dim :]
        tokens, token_mask, self_mark, ally_mark, enemy_mark = self._build_semantic_tokens(
            move_feat, own_feat, ally_feat, enemy_feat
        )
        updated_tokens, pair_attn = self.self_attention(tokens, token_mask)

        self_context = self._pool_by_mark(updated_tokens, token_mask, self_mark)
        ally_context = self._pool_by_mark(updated_tokens, token_mask, ally_mark)
        enemy_context = self._pool_by_mark(updated_tokens, token_mask, enemy_mark)
        instant = self.instant_pattern(th.cat([self_context, ally_context, enemy_context], dim=-1))
        temporal_input = th.cat([self_context, instant], dim=-1)

        batch_size, n_agents, _ = temporal_input.shape
        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)

        ally_attn = self._aggregate_entity_attention(pair_attn, token_mask, self_mark, ally_mark)
        enemy_attn = self._aggregate_entity_attention(pair_attn, token_mask, self_mark, enemy_mark)
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class EntitySelfAttentionRelationCapturer(nn.Module):
    # Entity-role variant: keep self/ally/enemy entity tokens, but replace
    # first-person cross-attention with one masked self-attention block.
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
        self.owner_embedding = nn.Embedding(3, relation_dim)
        self.self_attention = MaskedSelfAttentionBlock(relation_dim)
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

    def _masked_mean(self, tokens, mask):
        denom = mask.sum(dim=2, keepdim=True).clamp(min=1).float()
        return (tokens * mask.unsqueeze(-1).float()).sum(dim=2) / denom

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        batch_size, n_agents, _ = self_feat.shape
        self_mask = th.ones(batch_size, n_agents, 1, device=self_feat.device, dtype=th.bool)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0

        self_token = self.self_encoder(self_feat).unsqueeze(2)
        ally_tokens = self.ally_encoder(ally_feat) * ally_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        self_owner = self.owner_embedding.weight[0].view(1, 1, 1, -1)
        ally_owner = self.owner_embedding.weight[1].view(1, 1, 1, -1)
        enemy_owner = self.owner_embedding.weight[2].view(1, 1, 1, -1)
        tokens = th.cat([self_token + self_owner, ally_tokens + ally_owner, enemy_tokens + enemy_owner], dim=2)
        token_mask = th.cat([self_mask, ally_mask, enemy_mask], dim=2)
        updated, pair_attn = self.self_attention(tokens, token_mask)

        self_context = updated[:, :, 0]
        ally_start = 1
        enemy_start = ally_start + ally_feat.size(2)
        ally_context = self._masked_mean(updated[:, :, ally_start:enemy_start], ally_mask)
        enemy_context = self._masked_mean(updated[:, :, enemy_start:], enemy_mask)
        instant = self.instant_pattern(th.cat([self_context, ally_context, enemy_context], dim=-1))
        temporal_input = th.cat([self_context, instant], dim=-1)

        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)

        self_attn = pair_attn[:, :, 0]
        ally_attn = self_attn[:, :, ally_start:enemy_start]
        enemy_attn = self_attn[:, :, enemy_start:]
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class TopKEntitySelfAttentionRelationCapturer(EntitySelfAttentionRelationCapturer):
    # Entity self-attention with an explicit top-k readout bottleneck. All
    # observed entities exchange information through self-attention, but only
    # the k entities most attended by the self token are pooled into the
    # relation pattern. Enemy action scores are still produced for every enemy
    # slot, so the SMAC action dimension remains unchanged.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        topk=5,
    ):
        super().__init__(
            move_dim=move_dim,
            own_dim=own_dim,
            ally_feat_dim=ally_feat_dim,
            enemy_feat_dim=enemy_feat_dim,
            relation_dim=relation_dim,
            output_dim=output_dim,
        )
        self.topk = int(topk)

    def _topk_entity_mask(self, scores, entity_mask):
        if scores.size(-1) == 0:
            return entity_mask
        topk = max(1, min(self.topk, scores.size(-1)))
        masked_scores = scores.masked_fill(~entity_mask.bool(), _neg_inf_like(scores))
        _, indices = th.topk(masked_scores, k=topk, dim=-1)
        gathered_valid = th.gather(entity_mask.bool(), dim=-1, index=indices)
        selected = scores.new_zeros(scores.shape)
        selected.scatter_(dim=-1, index=indices, src=gathered_valid.float())
        return selected.bool() & entity_mask.bool()

    def forward(self, self_feat, ally_feat, enemy_feat, prev_relation_hidden):
        batch_size, n_agents, _ = self_feat.shape
        self_mask = th.ones(batch_size, n_agents, 1, device=self_feat.device, dtype=th.bool)
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0

        self_token = self.self_encoder(self_feat).unsqueeze(2)
        ally_tokens = self.ally_encoder(ally_feat) * ally_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        self_owner = self.owner_embedding.weight[0].view(1, 1, 1, -1)
        ally_owner = self.owner_embedding.weight[1].view(1, 1, 1, -1)
        enemy_owner = self.owner_embedding.weight[2].view(1, 1, 1, -1)
        tokens = th.cat([self_token + self_owner, ally_tokens + ally_owner, enemy_tokens + enemy_owner], dim=2)
        token_mask = th.cat([self_mask, ally_mask, enemy_mask], dim=2)
        updated, pair_attn = self.self_attention(tokens, token_mask)

        ally_start = 1
        enemy_start = ally_start + ally_feat.size(2)
        entity_scores = pair_attn[:, :, 0, 1:]
        entity_mask = th.cat([ally_mask, enemy_mask], dim=-1)
        selected_entity_mask = self._topk_entity_mask(entity_scores, entity_mask)
        selected_ally_mask = selected_entity_mask[:, :, : ally_feat.size(2)]
        selected_enemy_mask = selected_entity_mask[:, :, ally_feat.size(2) :]

        self_context = updated[:, :, 0]
        ally_context = self._masked_mean(updated[:, :, ally_start:enemy_start], selected_ally_mask)
        enemy_context = self._masked_mean(updated[:, :, enemy_start:], selected_enemy_mask)
        instant = self.instant_pattern(th.cat([self_context, ally_context, enemy_context], dim=-1))
        temporal_input = th.cat([self_context, instant], dim=-1)

        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)

        self_attn = pair_attn[:, :, 0]
        ally_attn = self_attn[:, :, ally_start:enemy_start] * selected_ally_mask.float()
        enemy_attn = self_attn[:, :, enemy_start:] * selected_enemy_mask.float()
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask


class ActionEdgeGraphRelationCapturer(nn.Module):
    # Action-edge graph relation generator. Nodes encode observer-invariant
    # public entity state; directed edges encode predicted action intent. The
    # action predictor reads obs_{t-1}, obs_t, and their difference, while
    # ground-truth actions are used only for an auxiliary prediction loss.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        obs_dim,
        n_agents,
        n_actions,
        n_enemies,
        unit_type_bits=0,
        shield_bits_ally=0,
        shield_bits_enemy=0,
        obs_all_health=True,
        obs_own_health=True,
        graph_encoder_type="pool",
        use_oracle_edges=False,
        oracle_edge_mode="current",
        predictor_input_mode="full",
        use_public_memory=False,
        return_target_context=False,
        no_self_identity=False,
    ):
        super().__init__()
        del own_dim
        self.move_dim = move_dim
        self.relation_dim = relation_dim
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_enemies = n_enemies
        self.n_ego_actions = n_actions - n_enemies
        self.unit_type_bits = unit_type_bits
        self.shield_bits_ally = shield_bits_ally
        self.shield_bits_enemy = shield_bits_enemy
        self.obs_all_health = obs_all_health
        self.obs_own_health = obs_own_health
        self.graph_encoder_type = graph_encoder_type
        self.use_oracle_edges = use_oracle_edges
        self.oracle_edge_mode = oracle_edge_mode
        self.predictor_input_mode = predictor_input_mode
        self.use_public_memory = use_public_memory
        self.return_target_context = return_target_context
        self.no_self_identity = no_self_identity
        self.latest_aux_stats = {}
        self.latest_enemy_graph_tokens = None

        self.public_self_dim = 1 + unit_type_bits
        self.public_ally_dim = 1 + unit_type_bits
        self.public_enemy_dim = 1 + unit_type_bits
        if obs_own_health:
            self.public_self_dim += 1 + shield_bits_ally
        if obs_all_health:
            self.public_ally_dim += 1 + shield_bits_ally
            self.public_enemy_dim += 1 + shield_bits_enemy
        self.public_obs_dim = (
            self.public_self_dim
            + (n_agents - 1) * self.public_ally_dim
            + n_enemies * self.public_enemy_dim
        )
        predictor_source_dim = self.public_obs_dim if predictor_input_mode == "public" else obs_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(self.public_self_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.ally_encoder = nn.Sequential(
            nn.Linear(self.public_ally_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.public_enemy_encoder = nn.Sequential(
            nn.Linear(self.public_enemy_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(enemy_feat_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(predictor_source_dim * 3, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, n_agents * n_actions),
        )
        self.actor_message = nn.Sequential(
            nn.Linear(relation_dim + n_actions, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.edge_message = nn.Sequential(
            nn.Linear(relation_dim * 2 + n_enemies + 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.rgcn_action_transforms = nn.ModuleList(
            [nn.Linear(relation_dim, relation_dim, bias=False) for _ in range(n_actions)]
        )
        self.rgcn_node_update = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.egcn_actor_value = nn.Linear(relation_dim, relation_dim, bias=False)
        self.egcn_actor_gate = nn.Sequential(
            nn.Linear(n_actions + 1, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
            nn.Sigmoid(),
        )
        self.egcn_attack_value = nn.Linear(relation_dim, relation_dim, bias=False)
        self.egcn_attack_gate = nn.Sequential(
            nn.Linear(n_enemies + 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
            nn.Sigmoid(),
        )
        self.egcn_node_update = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.egcn_plus_actor_value = nn.Linear(relation_dim, relation_dim, bias=False)
        self.egcn_plus_actor_gate = nn.Sequential(
            nn.Linear(n_actions + 1, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
            nn.Sigmoid(),
        )
        self.egcn_plus_attack_value = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.egcn_plus_attack_gate = nn.Sequential(
            nn.Linear(n_enemies + 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
            nn.Sigmoid(),
        )
        self.egcn_plus_node_update = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.struct_num_heads = 4 if relation_dim % 4 == 0 else 1
        self.struct_graph_token = nn.Parameter(th.zeros(1, 1, relation_dim))
        self.struct_role_embedding = nn.Embedding(3, relation_dim)
        self.struct_centrality_encoder = nn.Sequential(
            nn.Linear(4, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.struct_pair_bias = nn.Sequential(
            nn.Linear(6, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, self.struct_num_heads),
        )
        self.struct_kernel_scale = nn.Parameter(th.ones(self.struct_num_heads))
        self.struct_transformer = nn.ModuleList(
            [BiasTransformerEncoderLayer(relation_dim, self.struct_num_heads)]
        )
        self.edge_set_encoder = nn.Sequential(
            nn.Linear(relation_dim * 2 + 1, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.edge_set_transformer = nn.ModuleList(
            [BiasTransformerEncoderLayer(relation_dim, self.struct_num_heads)]
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(move_dim + 4, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.instant_pattern = nn.Sequential(
            nn.Linear(relation_dim * 3, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.condition_fuser = nn.Sequential(
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

    def _masked_mean(self, tokens, mask):
        denom = mask.sum(dim=2, keepdim=True).clamp(min=1).float()
        return (tokens * mask.unsqueeze(-1).float()).sum(dim=2) / denom

    def _other_agent_ids(self, device):
        agent_ids = th.arange(self.n_agents, device=device)
        return th.stack([agent_ids[agent_ids != agent_id] for agent_id in range(self.n_agents)], dim=0)

    def _self_public_features(self, self_feat):
        own_feat = self_feat[:, :, self.move_dim :]
        batch_size, n_agents, _ = own_feat.shape
        features = [own_feat.new_ones(batch_size, n_agents, 1)]
        idx = 0
        if self.obs_own_health:
            features.append(own_feat[:, :, idx : idx + 1])
            idx += 1
            if self.shield_bits_ally > 0:
                features.append(own_feat[:, :, idx : idx + 1])
                idx += 1
        if self.unit_type_bits > 0:
            features.append(own_feat[:, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _self_as_ally_public_features(self, self_feat):
        own_feat = self_feat[:, :, self.move_dim :]
        batch_size, n_agents, _ = own_feat.shape
        features = [own_feat.new_ones(batch_size, n_agents, 1)]
        idx = 0
        if self.obs_own_health:
            own_health = own_feat[:, :, idx : idx + 1]
            idx += 1
            own_shield = None
            if self.shield_bits_ally > 0:
                own_shield = own_feat[:, :, idx : idx + 1]
                idx += 1
        else:
            own_health = own_feat.new_zeros(batch_size, n_agents, 1)
            own_shield = own_feat.new_zeros(batch_size, n_agents, 1) if self.shield_bits_ally > 0 else None
        if self.obs_all_health:
            features.append(own_health)
            if self.shield_bits_ally > 0:
                features.append(own_shield)
        if self.unit_type_bits > 0:
            features.append(own_feat[:, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _ally_public_features(self, ally_feat, ally_mask):
        features = [ally_mask.unsqueeze(-1).float()]
        idx = 4
        if self.obs_all_health:
            features.append(ally_feat[:, :, :, idx : idx + 1])
            idx += 1
            if self.shield_bits_ally > 0:
                features.append(ally_feat[:, :, :, idx : idx + 1])
                idx += 1
        if self.unit_type_bits > 0:
            features.append(ally_feat[:, :, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _enemy_public_features(self, enemy_feat, enemy_mask):
        features = [enemy_mask.unsqueeze(-1).float()]
        idx = 4
        if self.obs_all_health:
            features.append(enemy_feat[:, :, :, idx : idx + 1])
            idx += 1
            if self.shield_bits_enemy > 0:
                features.append(enemy_feat[:, :, :, idx : idx + 1])
                idx += 1
        if self.unit_type_bits > 0:
            features.append(enemy_feat[:, :, :, idx : idx + self.unit_type_bits])
        return th.cat(features, dim=-1)

    def _public_parts(self, self_feat, ally_feat, enemy_feat):
        ally_mask = ally_feat.abs().sum(dim=-1) > 0
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        self_public = self._self_public_features(self_feat)
        ally_public = self._ally_public_features(ally_feat, ally_mask)
        enemy_public = self._enemy_public_features(enemy_feat, enemy_mask)
        return self_public, ally_public, enemy_public, ally_mask, enemy_mask

    def _public_flat(self, self_public, ally_public, enemy_public):
        return th.cat(
            [
                self_public.reshape(self_public.size(0), self_public.size(1), -1),
                ally_public.reshape(ally_public.size(0), ally_public.size(1), -1),
                enemy_public.reshape(enemy_public.size(0), enemy_public.size(1), -1),
            ],
            dim=-1,
        )

    def _cover_missing_public(self, current, previous, current_mask, previous_mask):
        covered_mask = current_mask | previous_mask
        covered = th.where(current_mask.unsqueeze(-1), current, previous)
        return covered, covered_mask

    def _private_view_context(self, obs, self_feat, enemy_feat, enemy_mask):
        del obs
        move_feat = self_feat[:, :, : self.move_dim]
        enemy_private = enemy_feat[:, :, :, :4]
        enemy_private_mean = self._masked_mean(enemy_private, enemy_mask)
        return self.private_encoder(th.cat([move_feat, enemy_private_mean], dim=-1))

    def _action_targets_for_actor_slots(self, target_actions, action_target_mask, ally_mask):
        if target_actions is None:
            return None, None
        target_actions = target_actions.long()
        if action_target_mask is None:
            action_target_mask = th.ones_like(target_actions, dtype=th.bool)
        else:
            action_target_mask = action_target_mask.bool()
        other_ids = self._other_agent_ids(target_actions.device)
        self_targets = target_actions.unsqueeze(2)
        ally_targets = target_actions[:, other_ids]
        actor_targets = th.cat([self_targets, ally_targets], dim=2)
        self_target_mask = action_target_mask.unsqueeze(2)
        ally_target_mask = action_target_mask[:, other_ids]
        actor_target_mask = th.cat([self_target_mask, ally_target_mask], dim=2)
        actor_obs_mask = th.cat(
            [
                th.ones_like(self_targets, dtype=th.bool),
                ally_mask.bool(),
            ],
            dim=2,
        )
        return actor_targets, actor_target_mask & actor_obs_mask

    def _action_prediction_loss(self, actor_logits, actor_targets, actor_target_mask):
        if actor_targets is None or actor_target_mask is None or not actor_target_mask.any():
            return actor_logits.new_zeros(())
        flat_logits = actor_logits.reshape(-1, self.n_actions)
        flat_targets = actor_targets.reshape(-1)
        flat_mask = actor_target_mask.reshape(-1).float()
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        return (loss * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)

    def _action_prediction_stats(self, actor_logits, actor_probs, edge_probs, actor_targets, actor_target_mask):
        stats = {
            "action_edge_encoder_pool": actor_logits.new_tensor(1.0 if self.graph_encoder_type == "pool" else 0.0),
            "action_edge_encoder_rgcn": actor_logits.new_tensor(1.0 if self.graph_encoder_type == "rgcn" else 0.0),
            "action_edge_encoder_egcn": actor_logits.new_tensor(1.0 if self.graph_encoder_type == "egcn" else 0.0),
            "action_edge_encoder_egcn_plus": actor_logits.new_tensor(
                1.0 if self.graph_encoder_type == "egcn_plus" else 0.0
            ),
            "action_edge_encoder_graphormer": actor_logits.new_tensor(
                1.0 if self.graph_encoder_type == "graphormer" else 0.0
            ),
            "action_edge_encoder_graphit": actor_logits.new_tensor(
                1.0 if self.graph_encoder_type == "graphit" else 0.0
            ),
            "action_edge_encoder_edgeset": actor_logits.new_tensor(
                1.0 if self.graph_encoder_type == "edgeset" else 0.0
            ),
            "action_edge_encoder_motif_transformer": actor_logits.new_tensor(
                1.0 if self.graph_encoder_type == "motif_transformer" else 0.0
            ),
            "action_edge_oracle": actor_logits.new_tensor(1.0 if self.use_oracle_edges else 0.0),
            "action_edge_prev_oracle": actor_logits.new_tensor(
                1.0 if self.use_oracle_edges and self.oracle_edge_mode == "previous" else 0.0
            ),
            "action_edge_no_self_identity": actor_logits.new_tensor(1.0 if self.no_self_identity else 0.0),
            "action_pred_public_input": actor_logits.new_tensor(1.0 if self.predictor_input_mode == "public" else 0.0),
            "action_edge_public_memory": actor_logits.new_tensor(1.0 if self.use_public_memory else 0.0),
            "action_edge_target_context": actor_logits.new_tensor(1.0 if self.return_target_context else 0.0),
        }
        entropy = -(actor_probs * actor_probs.clamp(min=1e-8).log()).sum(dim=-1)
        stats["action_pred_entropy"] = entropy.mean().detach()
        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        edge_attack_probs = edge_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        stats["pred_attack_rate"] = attack_probs.sum(dim=-1).mean().detach()
        stats["edge_attack_mass"] = edge_attack_probs.sum(dim=-1).mean().detach()
        if actor_targets is None or actor_target_mask is None or not actor_target_mask.any():
            stats["action_pred_acc"] = actor_logits.new_zeros(())
            stats["attack_vs_nonattack_acc"] = actor_logits.new_zeros(())
            stats["attack_target_acc"] = actor_logits.new_zeros(())
            stats["true_attack_rate"] = actor_logits.new_zeros(())
            return stats
        pred = actor_logits.argmax(dim=-1)
        mask = actor_target_mask.float()
        acc = ((pred == actor_targets).float() * mask).sum() / mask.sum().clamp(min=1.0)
        stats["action_pred_acc"] = acc.detach()
        pred_attack = pred >= self.n_ego_actions
        true_attack = actor_targets >= self.n_ego_actions
        attack_binary_acc = ((pred_attack == true_attack).float() * mask).sum() / mask.sum().clamp(min=1.0)
        attack_mask = (true_attack & actor_target_mask).float()
        attack_target_acc = ((pred == actor_targets).float() * attack_mask).sum() / attack_mask.sum().clamp(min=1.0)
        stats["attack_vs_nonattack_acc"] = attack_binary_acc.detach()
        stats["attack_target_acc"] = attack_target_acc.detach()
        stats["true_attack_rate"] = (true_attack.float() * mask).sum().detach() / mask.sum().clamp(min=1.0)
        return stats

    def _oracle_actor_probs(self, actor_targets, actor_target_mask, actor_probs):
        if actor_targets is None or actor_target_mask is None:
            return actor_probs
        oracle = F.one_hot(actor_targets.long().clamp(min=0), num_classes=self.n_actions).to(actor_probs.dtype)
        return th.where(actor_target_mask.unsqueeze(-1).bool(), oracle, actor_probs)

    def _pool_graph_context(self, self_token, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, actor_probs):
        ego_mass = actor_probs[:, :, :, : self.n_ego_actions].sum(dim=-1, keepdim=True)
        actor_messages = self.actor_message(th.cat([actor_tokens, actor_probs], dim=-1))
        actor_context = self._masked_mean(actor_messages, actor_mask)

        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        actor_exp = actor_tokens.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        enemy_exp = public_enemy_tokens.unsqueeze(2).expand(-1, -1, self.n_agents, -1, -1)
        target_attack_prob = attack_probs.unsqueeze(-1)
        attack_dist = attack_probs.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        ego_mass_exp = ego_mass.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        edge_input = th.cat([actor_exp, enemy_exp, attack_dist, target_attack_prob, ego_mass_exp], dim=-1)
        edge_messages = self.edge_message(edge_input)
        edge_weight = (
            target_attack_prob
            * actor_mask.unsqueeze(-1).unsqueeze(-1).float()
            * enemy_mask.unsqueeze(2).unsqueeze(-1).float()
        )
        enemy_received = (edge_messages * edge_weight).sum(dim=2)
        enemy_denom = edge_weight.sum(dim=2).clamp(min=1.0)
        enemy_received = enemy_received / enemy_denom
        self.latest_enemy_graph_tokens = enemy_received * enemy_mask.unsqueeze(-1).float()
        enemy_context = self._masked_mean(enemy_received, enemy_mask)
        return actor_context, enemy_context, attack_probs

    def _rgcn_graph_context(self, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, actor_probs):
        actor_received = actor_tokens.new_zeros(actor_tokens.shape)
        for action_id in range(self.n_ego_actions):
            transformed = self.rgcn_action_transforms[action_id](actor_tokens)
            weight = actor_probs[:, :, :, action_id].unsqueeze(-1) * actor_mask.unsqueeze(-1).float()
            actor_received = actor_received + transformed * weight
        actor_updated = self.rgcn_node_update(th.cat([actor_tokens, actor_received], dim=-1))
        actor_context = self._masked_mean(actor_updated, actor_mask)

        enemy_received = public_enemy_tokens.new_zeros(public_enemy_tokens.shape)
        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        for enemy_id in range(self.n_enemies):
            action_id = self.n_ego_actions + enemy_id
            transformed = self.rgcn_action_transforms[action_id](actor_tokens)
            weight = attack_probs[:, :, :, enemy_id].unsqueeze(-1) * actor_mask.unsqueeze(-1).float()
            enemy_received[:, :, enemy_id] = (transformed * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1.0)
        enemy_updated = self.rgcn_node_update(th.cat([public_enemy_tokens, enemy_received], dim=-1))
        self.latest_enemy_graph_tokens = enemy_updated * enemy_mask.unsqueeze(-1).float()
        enemy_context = self._masked_mean(enemy_updated, enemy_mask)
        return actor_context, enemy_context, attack_probs

    def _egcn_graph_context(self, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, actor_probs):
        ego_mass = actor_probs[:, :, :, : self.n_ego_actions].sum(dim=-1, keepdim=True)
        actor_edge_feat = th.cat([actor_probs, ego_mass], dim=-1)
        actor_msg = self.egcn_actor_value(actor_tokens) * self.egcn_actor_gate(actor_edge_feat)
        actor_msg = actor_msg * actor_mask.unsqueeze(-1).float()
        actor_updated = self.egcn_node_update(th.cat([actor_tokens, actor_msg], dim=-1))
        actor_context = self._masked_mean(actor_updated, actor_mask)

        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        target_attack_prob = attack_probs.unsqueeze(-1)
        attack_dist = attack_probs.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        ego_mass_exp = ego_mass.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        attack_edge_feat = th.cat([attack_dist, target_attack_prob, ego_mass_exp], dim=-1)
        attack_value = self.egcn_attack_value(actor_tokens).unsqueeze(3)
        attack_msg = attack_value * self.egcn_attack_gate(attack_edge_feat)
        edge_weight = (
            target_attack_prob
            * actor_mask.unsqueeze(-1).unsqueeze(-1).float()
            * enemy_mask.unsqueeze(2).unsqueeze(-1).float()
        )
        enemy_received = (attack_msg * edge_weight).sum(dim=2) / edge_weight.sum(dim=2).clamp(min=1.0)
        enemy_updated = self.egcn_node_update(th.cat([public_enemy_tokens, enemy_received], dim=-1))
        self.latest_enemy_graph_tokens = enemy_updated * enemy_mask.unsqueeze(-1).float()
        enemy_context = self._masked_mean(enemy_updated, enemy_mask)
        return actor_context, enemy_context, attack_probs

    def _egcn_plus_graph_context(self, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, actor_probs):
        ego_mass = actor_probs[:, :, :, : self.n_ego_actions].sum(dim=-1, keepdim=True)
        actor_edge_feat = th.cat([actor_probs, ego_mass], dim=-1)
        actor_msg = self.egcn_plus_actor_value(actor_tokens) * self.egcn_plus_actor_gate(actor_edge_feat)
        actor_msg = actor_msg * actor_mask.unsqueeze(-1).float()
        actor_updated = self.egcn_plus_node_update(th.cat([actor_tokens, actor_msg], dim=-1))
        actor_context = self._masked_mean(actor_updated, actor_mask)

        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        actor_exp = actor_tokens.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        enemy_exp = public_enemy_tokens.unsqueeze(2).expand(-1, -1, self.n_agents, -1, -1)
        target_attack_prob = attack_probs.unsqueeze(-1)
        attack_dist = attack_probs.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        ego_mass_exp = ego_mass.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        attack_edge_feat = th.cat([attack_dist, target_attack_prob, ego_mass_exp], dim=-1)
        attack_value = self.egcn_plus_attack_value(th.cat([actor_exp, enemy_exp], dim=-1))
        attack_msg = attack_value * self.egcn_plus_attack_gate(attack_edge_feat)
        edge_weight = (
            target_attack_prob
            * actor_mask.unsqueeze(-1).unsqueeze(-1).float()
            * enemy_mask.unsqueeze(2).unsqueeze(-1).float()
        )
        enemy_received = (attack_msg * edge_weight).sum(dim=2) / edge_weight.sum(dim=2).clamp(min=1.0)
        enemy_updated = self.egcn_plus_node_update(th.cat([public_enemy_tokens, enemy_received], dim=-1))
        self.latest_enemy_graph_tokens = enemy_updated * enemy_mask.unsqueeze(-1).float()
        enemy_context = self._masked_mean(enemy_updated, enemy_mask)
        return actor_context, enemy_context, attack_probs

    def _structure_role_ids(self, device):
        return th.tensor(
            [0] + [1] * (self.n_agents - 1) + [2] * self.n_enemies,
            device=device,
            dtype=th.long,
        )

    def _masked_mean_flat(self, tokens, mask):
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        return (tokens * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def _structure_node_inputs(self, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, attack_probs, motif=False):
        batch_size, n_agents, _, _ = actor_tokens.shape
        attack_weight = (
            attack_probs
            * actor_mask.unsqueeze(-1).float()
            * enemy_mask.unsqueeze(2).float()
        )
        actor_sum = attack_weight.sum(dim=-1)
        actor_max = attack_weight.max(dim=-1).values
        enemy_sum = attack_weight.sum(dim=2)
        enemy_max = attack_weight.max(dim=2).values

        actor_extra = th.zeros_like(actor_sum)
        actor_extra_max = th.zeros_like(actor_sum)
        enemy_extra = th.zeros_like(enemy_sum)
        enemy_extra_max = th.zeros_like(enemy_sum)
        if motif:
            flat_p = attack_weight.reshape(batch_size * n_agents, self.n_agents, self.n_enemies)
            cofocus = th.bmm(flat_p, flat_p.transpose(1, 2)) / max(1, self.n_enemies)
            pressure = th.bmm(flat_p.transpose(1, 2), flat_p) / max(1, self.n_agents)
            actor_extra = cofocus.sum(dim=-1).view(batch_size, n_agents, self.n_agents)
            actor_extra_max = cofocus.max(dim=-1).values.view(batch_size, n_agents, self.n_agents)
            enemy_extra = pressure.sum(dim=-1).view(batch_size, n_agents, self.n_enemies)
            enemy_extra_max = pressure.max(dim=-1).values.view(batch_size, n_agents, self.n_enemies)

        actor_centrality = th.stack([actor_sum, actor_max, actor_extra, actor_extra_max], dim=-1)
        enemy_centrality = th.stack([enemy_sum, enemy_max, enemy_extra, enemy_extra_max], dim=-1)
        centrality = th.cat([actor_centrality, enemy_centrality], dim=2)
        node_tokens = th.cat([actor_tokens, public_enemy_tokens], dim=2)
        node_mask = th.cat([actor_mask, enemy_mask], dim=2)
        role_emb = self.struct_role_embedding(self._structure_role_ids(node_tokens.device)).view(
            1, 1, self.n_agents + self.n_enemies, -1
        )
        node_tokens = node_tokens + role_emb + self.struct_centrality_encoder(centrality)
        node_tokens = node_tokens * node_mask.unsqueeze(-1).float()
        return node_tokens, node_mask, attack_weight

    def _directed_attack_adjacency(self, attack_weight):
        batch_size, n_agents, _, _ = attack_weight.shape
        n_nodes = self.n_agents + self.n_enemies
        adj = attack_weight.new_zeros(batch_size * n_agents, n_nodes, n_nodes)
        flat_p = attack_weight.reshape(batch_size * n_agents, self.n_agents, self.n_enemies)
        adj[:, : self.n_agents, self.n_agents :] = flat_p
        return adj, flat_p

    def _graphormer_bias(self, attack_weight, motif=False):
        directed, flat_p = self._directed_attack_adjacency(attack_weight)
        n_flat, n_nodes, _ = directed.shape
        sym_adj = directed + directed.transpose(1, 2)
        degree = sym_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = sym_adj / degree
        if motif:
            two_hop = directed.new_zeros(n_flat, n_nodes, n_nodes)
            two_hop[:, : self.n_agents, : self.n_agents] = th.bmm(
                flat_p, flat_p.transpose(1, 2)
            ) / max(1, self.n_enemies)
            two_hop[:, self.n_agents :, self.n_agents :] = th.bmm(
                flat_p.transpose(1, 2), flat_p
            ) / max(1, self.n_agents)
        else:
            two_hop = th.bmm(norm_adj, norm_adj)

        idx = th.arange(n_nodes, device=directed.device)
        actor_node = idx < self.n_agents
        enemy_node = ~actor_node
        actor_actor = (actor_node[:, None] & actor_node[None, :]).to(dtype=directed.dtype)
        enemy_enemy = (enemy_node[:, None] & enemy_node[None, :]).to(dtype=directed.dtype)
        cross = (actor_node[:, None] ^ actor_node[None, :]).to(dtype=directed.dtype)
        pair_feat = th.stack(
            [
                directed,
                directed.transpose(1, 2),
                two_hop,
                actor_actor.expand_as(directed),
                enemy_enemy.expand_as(directed),
                cross.expand_as(directed),
            ],
            dim=-1,
        )
        node_bias = self.struct_pair_bias(pair_feat).permute(0, 3, 1, 2).contiguous()
        full_bias = node_bias.new_zeros(n_flat, self.struct_num_heads, n_nodes + 1, n_nodes + 1)
        full_bias[:, :, 1:, 1:] = node_bias
        return full_bias

    def _graphit_bias(self, attack_weight):
        directed, _ = self._directed_attack_adjacency(attack_weight)
        n_flat, n_nodes, _ = directed.shape
        sym_adj = directed + directed.transpose(1, 2)
        degree = sym_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = sym_adj / degree
        identity = th.eye(n_nodes, device=directed.device, dtype=directed.dtype).unsqueeze(0)
        two_hop = th.bmm(norm_adj, norm_adj)
        kernel = (identity + 0.7 * norm_adj + 0.3 * two_hop).clamp(min=1e-4)
        node_bias = kernel.log().unsqueeze(1) * self.struct_kernel_scale.view(1, -1, 1, 1)
        full_bias = node_bias.new_zeros(n_flat, self.struct_num_heads, n_nodes + 1, n_nodes + 1)
        full_bias[:, :, 1:, 1:] = node_bias
        return full_bias

    def _run_struct_transformer(self, node_tokens, node_mask, attn_bias):
        batch_size, n_agents, n_nodes, _ = node_tokens.shape
        flat_nodes = node_tokens.reshape(batch_size * n_agents, n_nodes, self.relation_dim)
        flat_mask = node_mask.reshape(batch_size * n_agents, n_nodes)
        graph_token = self.struct_graph_token.expand(batch_size * n_agents, -1, -1)
        tokens = th.cat([graph_token, flat_nodes], dim=1)
        token_mask = th.cat(
            [th.ones(batch_size * n_agents, 1, device=node_mask.device, dtype=th.bool), flat_mask],
            dim=1,
        )
        for layer in self.struct_transformer:
            tokens = layer(tokens, token_mask, attn_bias=attn_bias)
        graph_context = tokens[:, 0]
        updated_nodes = tokens[:, 1:]
        actor_updated = updated_nodes[:, : self.n_agents]
        enemy_updated = updated_nodes[:, self.n_agents :]
        actor_mean = self._masked_mean_flat(actor_updated, flat_mask[:, : self.n_agents])
        enemy_mean = self._masked_mean_flat(enemy_updated, flat_mask[:, self.n_agents :])
        self.latest_enemy_graph_tokens = (
            enemy_updated.view(batch_size, n_agents, self.n_enemies, self.relation_dim)
            * node_mask[:, :, self.n_agents :].unsqueeze(-1).float()
        )
        return (
            (graph_context + actor_mean).view(batch_size, n_agents, self.relation_dim),
            enemy_mean.view(batch_size, n_agents, self.relation_dim),
        )

    def _structure_transformer_graph_context(
        self, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, actor_probs, mode
    ):
        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        motif = mode == "motif_transformer"
        node_tokens, node_mask, attack_weight = self._structure_node_inputs(
            actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, attack_probs, motif=motif
        )
        if mode == "graphit":
            attn_bias = self._graphit_bias(attack_weight)
        elif mode in {"graphormer", "motif_transformer"}:
            attn_bias = self._graphormer_bias(attack_weight, motif=motif)
        else:
            attn_bias = None
        actor_context, enemy_context = self._run_struct_transformer(node_tokens, node_mask, attn_bias)
        return actor_context, enemy_context, attack_probs

    def _edgeset_transformer_graph_context(self, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, actor_probs):
        batch_size, n_agents, _, _ = actor_tokens.shape
        attack_probs = actor_probs[:, :, :, self.n_ego_actions : self.n_ego_actions + self.n_enemies]
        node_tokens, node_mask, attack_weight = self._structure_node_inputs(
            actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, attack_probs, motif=False
        )
        actor_exp = actor_tokens.unsqueeze(3).expand(-1, -1, -1, self.n_enemies, -1)
        enemy_exp = public_enemy_tokens.unsqueeze(2).expand(-1, -1, self.n_agents, -1, -1)
        edge_tokens = self.edge_set_encoder(
            th.cat([actor_exp, enemy_exp, attack_weight.unsqueeze(-1)], dim=-1)
        )
        edge_mask = actor_mask.unsqueeze(-1) & enemy_mask.unsqueeze(2)
        n_nodes = self.n_agents + self.n_enemies
        flat_nodes = node_tokens.reshape(batch_size * n_agents, n_nodes, self.relation_dim)
        flat_node_mask = node_mask.reshape(batch_size * n_agents, n_nodes)
        flat_edges = edge_tokens.reshape(batch_size * n_agents, self.n_agents * self.n_enemies, self.relation_dim)
        flat_edge_mask = edge_mask.reshape(batch_size * n_agents, self.n_agents * self.n_enemies)
        graph_token = self.struct_graph_token.expand(batch_size * n_agents, -1, -1)
        tokens = th.cat([graph_token, flat_nodes, flat_edges], dim=1)
        token_mask = th.cat(
            [
                th.ones(batch_size * n_agents, 1, device=node_mask.device, dtype=th.bool),
                flat_node_mask,
                flat_edge_mask,
            ],
            dim=1,
        )
        for layer in self.edge_set_transformer:
            tokens = layer(tokens, token_mask, attn_bias=None)
        graph_context = tokens[:, 0]
        updated_nodes = tokens[:, 1 : 1 + n_nodes]
        actor_updated = updated_nodes[:, : self.n_agents]
        enemy_updated = updated_nodes[:, self.n_agents :]
        actor_mean = self._masked_mean_flat(actor_updated, flat_node_mask[:, : self.n_agents])
        enemy_mean = self._masked_mean_flat(enemy_updated, flat_node_mask[:, self.n_agents :])
        self.latest_enemy_graph_tokens = (
            enemy_updated.view(batch_size, n_agents, self.n_enemies, self.relation_dim)
            * enemy_mask.unsqueeze(-1).float()
        )
        return (
            (graph_context + actor_mean).view(batch_size, n_agents, self.relation_dim),
            enemy_mean.view(batch_size, n_agents, self.relation_dim),
            attack_probs,
        )

    def forward(
        self,
        self_feat,
        ally_feat,
        enemy_feat,
        prev_relation_hidden,
        obs,
        prev_obs,
        target_actions=None,
        action_target_mask=None,
        prev_action_targets=None,
    ):
        batch_size, n_agents, _ = self_feat.shape
        self.latest_enemy_graph_tokens = None
        move_dim = self.move_dim
        prev_move = prev_obs[:, :, :move_dim]
        prev_idx = move_dim
        prev_enemy_total = self.n_enemies * enemy_feat.size(-1)
        prev_enemy_feat = prev_obs[:, :, prev_idx : prev_idx + prev_enemy_total].view_as(enemy_feat)
        prev_idx += prev_enemy_total
        prev_ally_total = (self.n_agents - 1) * ally_feat.size(-1)
        prev_ally_feat = prev_obs[:, :, prev_idx : prev_idx + prev_ally_total].view_as(ally_feat)
        prev_idx += prev_ally_total
        prev_own_feat = prev_obs[:, :, prev_idx:]
        prev_self_feat = th.cat([prev_move, prev_own_feat], dim=-1)

        self_public, ally_public, enemy_public, ally_mask, enemy_mask = self._public_parts(
            self_feat, ally_feat, enemy_feat
        )
        prev_self_public, prev_ally_public, prev_enemy_public, prev_ally_mask, prev_enemy_mask = self._public_parts(
            prev_self_feat, prev_ally_feat, prev_enemy_feat
        )
        if self.use_public_memory:
            ally_public, ally_mask = self._cover_missing_public(ally_public, prev_ally_public, ally_mask, prev_ally_mask)
            enemy_public, enemy_mask = self._cover_missing_public(enemy_public, prev_enemy_public, enemy_mask, prev_enemy_mask)

        if self.no_self_identity:
            self_token = self.ally_encoder(self._self_as_ally_public_features(self_feat))
        else:
            self_token = self.self_encoder(self_public)
        ally_tokens = self.ally_encoder(ally_public) * ally_mask.unsqueeze(-1).float()
        public_enemy_tokens = self.public_enemy_encoder(enemy_public) * enemy_mask.unsqueeze(-1).float()
        enemy_tokens = self.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()

        if self.predictor_input_mode == "public":
            current_source = self._public_flat(self_public, ally_public, enemy_public)
            previous_source = self._public_flat(prev_self_public, prev_ally_public, prev_enemy_public)
        else:
            current_source = obs
            previous_source = prev_obs
        obs_delta = current_source - previous_source
        action_input = th.cat([current_source, previous_source, obs_delta], dim=-1)
        actor_logits = self.action_predictor(action_input).view(
            batch_size, n_agents, self.n_agents, self.n_actions
        )
        actor_probs = F.softmax(actor_logits, dim=-1)

        actor_tokens = th.cat([self_token.unsqueeze(2), ally_tokens], dim=2)
        actor_mask = th.cat(
            [th.ones(batch_size, n_agents, 1, device=self_feat.device, dtype=th.bool), ally_mask],
            dim=2,
        )
        actor_targets, actor_target_mask = self._action_targets_for_actor_slots(
            target_actions, action_target_mask, ally_mask
        )
        prev_actor_targets, prev_actor_target_mask = self._action_targets_for_actor_slots(
            prev_action_targets, action_target_mask, ally_mask
        )
        oracle_targets = prev_actor_targets if self.oracle_edge_mode == "previous" else actor_targets
        oracle_target_mask = prev_actor_target_mask if self.oracle_edge_mode == "previous" else actor_target_mask
        edge_probs = (
            self._oracle_actor_probs(oracle_targets, oracle_target_mask, actor_probs)
            if self.use_oracle_edges
            else actor_probs
        )
        if self.graph_encoder_type == "rgcn":
            actor_context, enemy_context, attack_probs = self._rgcn_graph_context(
                actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, edge_probs
            )
        elif self.graph_encoder_type == "egcn_plus":
            actor_context, enemy_context, attack_probs = self._egcn_plus_graph_context(
                actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, edge_probs
            )
        elif self.graph_encoder_type == "egcn":
            actor_context, enemy_context, attack_probs = self._egcn_graph_context(
                actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, edge_probs
            )
        elif self.graph_encoder_type in {"graphormer", "graphit", "motif_transformer"}:
            actor_context, enemy_context, attack_probs = self._structure_transformer_graph_context(
                actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, edge_probs, self.graph_encoder_type
            )
        elif self.graph_encoder_type == "edgeset":
            actor_context, enemy_context, attack_probs = self._edgeset_transformer_graph_context(
                actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, edge_probs
            )
        else:
            actor_context, enemy_context, attack_probs = self._pool_graph_context(
                self_token, actor_tokens, actor_mask, public_enemy_tokens, enemy_mask, edge_probs
            )

        private_context = self._private_view_context(obs, self_feat, enemy_feat, enemy_mask)
        graph_signal = self.instant_pattern(th.cat([self_token, actor_context, enemy_context], dim=-1))
        instant = self.condition_fuser(th.cat([graph_signal, private_context], dim=-1))
        temporal_input = th.cat([private_context, instant], dim=-1)

        flat_input = temporal_input.reshape(batch_size * n_agents, -1)
        flat_prev = prev_relation_hidden.reshape(batch_size * n_agents, -1)
        relation_hidden = self.temporal_gru(flat_input, flat_prev).view(batch_size, n_agents, self.relation_dim)
        condition = self.output_encoder(relation_hidden)

        action_loss = self._action_prediction_loss(actor_logits, actor_targets, actor_target_mask)
        self.latest_aux_stats = self._action_prediction_stats(
            actor_logits, actor_probs, edge_probs, actor_targets, actor_target_mask
        )
        self.latest_aux_stats["action_pred_loss_raw"] = action_loss.detach()

        ally_attn = edge_probs[:, :, 1:, self.n_ego_actions :].sum(dim=-1) * ally_mask.float()
        enemy_attn = (attack_probs * actor_mask.unsqueeze(-1).float()).sum(dim=2) * enemy_mask.float()
        return condition, relation_hidden, ally_attn, enemy_attn, enemy_tokens, enemy_mask, action_loss


class DeltaObservationRelationCapturer(nn.Module):
    # Temporal relation variant that removes the relation GRU. It combines the
    # current instant relation pattern with an explicit obs_t - obs_{t-1}
    # embedding, while leaving the policy GRU untouched.
    def __init__(
        self,
        move_dim,
        own_dim,
        ally_feat_dim,
        enemy_feat_dim,
        relation_dim,
        output_dim,
        obs_dim,
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
        self.delta_encoder = nn.Sequential(
            nn.Linear(obs_dim, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )
        self.output_encoder = nn.Sequential(
            nn.Linear(relation_dim * 2, output_dim),
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
        masked_logits = logits.masked_fill(~valid_mask, _neg_inf_like(logits))
        attn = F.softmax(masked_logits, dim=-1)
        attn = th.where(valid_any, attn, th.zeros_like(attn))
        context = th.matmul(attn.unsqueeze(2), value).squeeze(2)
        return context, attn

    def forward(self, self_feat, ally_feat, enemy_feat, delta_obs):
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
        delta_context = self.delta_encoder(delta_obs)
        condition = self.output_encoder(th.cat([instant, delta_context], dim=-1))
        return condition, ally_attn, enemy_attn, enemy_tokens, enemy_mask


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
        logits = logits.masked_fill(~node_mask, _neg_inf_like(logits))
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
        logits = logits.masked_fill(~valid_mask, _neg_inf_like(logits))
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
        type_logits = self.type_score(typed_messages).squeeze(-1)
        type_logits = type_logits.masked_fill(~type_mask, _neg_inf_like(type_logits))
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
        logits = logits.masked_fill(~source_mask, _neg_inf_like(logits))
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
        logits = logits.masked_fill(~valid_mask, _neg_inf_like(logits))
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
        logits = logits.masked_fill(~node_mask.unsqueeze(1), _neg_inf_like(logits))
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
        "local_linear_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_route": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_full_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_readout_structured_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_linear_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_flat_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_public_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_private_interaction_input_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_global_filled_obs_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_distill_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_public_delta_aux_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        **{
            name: {"uses_hypernet": True, "execution_scope": "ctde"}
            for name in PUBLIC_TRANSFORMER_CAPTURER_VARIANTS
        },
        "rpg_residual_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_film_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_moe_interaction_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_smooth_linear_interaction_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_semantic_selfattn_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_entity_selfattn_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_topk_entity_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        **{
            name: {"uses_hypernet": True, "execution_scope": "ctde"}
            for name in RPG_TARGETWISE_ABLATION_VARIANTS
        },
        **{
            name: {"uses_hypernet": True, "execution_scope": "ctde"}
            for name in TOKEN_DECISION_HEAD_VARIANTS
        },
        "rpg_action_edge_graph_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_rgcn_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_egcn_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_egcn_plus_public_pred_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_oracle_graph_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_oracle_no_self_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_prev_oracle_graph_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_public_pred_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_public_pred_public_hyper_private_input_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_public_pred_private_hyper_public_input_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_public_pred_coarse_fine_four_layer_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_public_pred_coarse_q_fine_gate_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_public_pred_relation_private_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_public_pred_relation_private_decision_maker": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_graphormer_relation_private_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_graphit_relation_private_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_edgeset_relation_private_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_motif_transformer_relation_private_single_head": {
            "uses_hypernet": True,
            "execution_scope": "ctde",
        },
        "rpg_action_edge_public_memory_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_global_public_pred_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_target_context_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_action_edge_coarse_private_fine_gate_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_public_hyper_private_input_single_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_private_hyper_public_input_single_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_delta_relation_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_coarse_self_fine_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_coarse_fine_four_layer_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_coarse_q_fine_gate_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_relation_prototype_single_head": {"uses_hypernet": True, "execution_scope": "ctde"},
        "rpg_fixed_structured_maker": {"uses_hypernet": False, "execution_scope": "ctde"},
        "rpg_fixed_linear_structured_maker": {"uses_hypernet": False, "execution_scope": "ctde"},
        "two_graph_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "hetero_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctde"},
        "global_two_graph_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctce"},
        "global_hetero_gat_hypercond": {"uses_hypernet": True, "execution_scope": "ctce"},
        "graph_hypercond": {"uses_hypernet": True, "execution_scope": "ctce"},
        "graph_route": {"uses_hypernet": True, "execution_scope": "ctce"},
        "qmix_minimal": {"uses_hypernet": False, "execution_scope": "ctde"},
        **{
            name: {"uses_hypernet": True, "execution_scope": "ctde"}
            for name in GRF_PUBLIC_TRANSFORMER_VARIANTS
        },
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
        self.stable_residual_gate_bias = float(getattr(args, "clean_stable_residual_gate_bias", -3.0))
        self.stable_residual_l2_coef = float(getattr(args, "clean_stable_residual_l2_coef", 0.001))
        self.self_fine_delta_scale = float(getattr(args, "clean_self_fine_delta_scale", 0.1))
        self.self_fine_gate_scale = float(getattr(args, "clean_self_fine_gate_scale", 1.0))
        self.relation_prototypes = int(getattr(args, "clean_relation_prototypes", self.route_num))
        self.relation_distill_coef = float(getattr(args, "clean_relation_distill_coef", 0.05))
        self.relation_teacher_td_coef = float(getattr(args, "clean_relation_teacher_td_coef", 0.2))
        self.smooth_head_loss_coef = float(getattr(args, "clean_smooth_head_loss_coef", 0.0))
        self.smooth_head_knn = int(getattr(args, "clean_smooth_head_knn", 4))
        self.smooth_head_sample_size = int(getattr(args, "clean_smooth_head_sample_size", 256))
        self.relation_topk = int(getattr(args, "clean_relation_topk", 5))
        self.target_topk = int(getattr(args, "clean_target_topk", self.relation_topk))
        self.target_threshold = float(getattr(args, "clean_target_threshold", 0.5))
        self.pre_relation_threshold = float(getattr(args, "clean_pre_relation_threshold", self.target_threshold))
        self.action_pred_loss_coef = float(getattr(args, "clean_action_pred_loss_coef", 0.05))
        self.public_delta_loss_coef = float(getattr(args, "clean_public_delta_loss_coef", 0.05))
        self.public_transformer_layers = int(getattr(args, "clean_public_transformer_layers", 1))
        self.public_transformer_heads = int(getattr(args, "clean_public_transformer_heads", 4))
        self.semantic_router_ema = float(getattr(args, "clean_semantic_router_ema", 0.99))
        self.semantic_router_threshold = float(
            getattr(args, "clean_semantic_router_threshold", 0.5)
        )
        self.semantic_router_temperature = float(
            getattr(args, "clean_semantic_router_temperature", 0.1)
        )
        self.semantic_router_warmup_steps = int(
            getattr(args, "clean_semantic_router_warmup_steps", 250000)
        )
        self.semantic_router_freeze_steps = int(
            getattr(args, "clean_semantic_router_freeze_steps", 5000000)
        )
        self.public_transformer_delta_loss_coef = float(
            getattr(args, "clean_public_transformer_delta_loss_coef", self.public_delta_loss_coef)
        )

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
            "rpg_public_hyper_private_input_single_head",
            "rpg_private_hyper_public_input_single_head",
        }:
            self.rpg_obs_layout = self._build_rpg_obs_layout()
            self.rpg_relation_capturer = None
        elif self.model_type in GRF_PUBLIC_TRANSFORMER_VARIANTS:
            self._init_grf_relation_capturer()
        elif self.model_type in {
            "local_structured_hypercond",
            "local_linear_interaction_hypercond",
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_flat_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_CAPTURER_VARIANTS,
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            *ACTION_EDGE_REL_PRIVATE_VARIANTS,
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_delta_relation_hypercond",
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_relation_prototype_single_head",
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

        if self.model_type in {"dynamic_route", "graph_route", "rpg_relation_route", "rpg_relation_prototype_single_head"}:
            route_num = self.relation_prototypes if self.model_type == "rpg_relation_prototype_single_head" else self.route_num
            self.route_logits_head = nn.Sequential(
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, route_num),
            )
            self.route_codebook = nn.Parameter(th.empty(route_num, self.cond_dim))
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
            "local_linear_interaction_hypercond",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_flat_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_RELATION_VARIANTS,
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            *ACTION_EDGE_REL_PRIVATE_VARIANTS,
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_public_hyper_private_input_single_head",
            "rpg_private_hyper_public_input_single_head",
            "rpg_delta_relation_hypercond",
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_relation_prototype_single_head",
            *GRF_DECISION_MAKER_VARIANTS,
            *GRF_LINEAR_HEAD_VARIANTS,
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

        if self.model_type in GRF_DECISION_MAKER_VARIANTS:
            self._init_grf_decision_maker_head()
        else:
            self.grf_decision_hyper = None

        if self.model_type in GRF_LINEAR_HEAD_VARIANTS:
            self.grf_linear_head_w = nn.Linear(self.cond_dim, self.hidden_dim * self.n_actions)
            self.grf_linear_head_b = nn.Linear(self.cond_dim, self.n_actions)
        else:
            self.grf_linear_head_w = None
            self.grf_linear_head_b = None

        if self.model_type in {
            "local_structured_hypercond",
            "local_linear_interaction_hypercond",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_flat_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_RELATION_VARIANTS,
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            "rpg_action_edge_public_pred_relation_private_decision_maker",
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_delta_relation_hypercond",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
        }:
            self.rpg_n_ego_actions = self.n_actions - self.rpg_obs_layout["n_enemies"]
            self.rpg_ego_input_dim = (
                self.hidden_dim + self.rpg_relation_dim
                if self.model_type in PUBLIC_TRANSFORMER_SLOT_TOKEN_HEAD_VARIANTS
                else self.hidden_dim
            )
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
                self.rpg_ego_bottleneck_w = nn.Linear(self.cond_dim, self.rpg_ego_input_dim * self.hidden_dim)
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
            elif self.model_type in {
                "local_linear_interaction_hypercond",
                "rpg_linear_interaction_hypercond",
                "rpg_public_relation_hypercond",
                "rpg_global_filled_obs_hypercond",
                "rpg_relation_distill_hypercond",
                "rpg_public_delta_aux_hypercond",
                *PUBLIC_TRANSFORMER_STANDARD_RELATION_VARIANTS,
                "rpg_semantic_selfattn_relation_hypercond",
                "rpg_entity_selfattn_relation_hypercond",
                "rpg_topk_entity_relation_hypercond",
                *RPG_STANDARD_INTERACTION_ABLATION_VARIANTS,
                "rpg_action_edge_graph_hypercond",
                "rpg_action_edge_rgcn_hypercond",
                "rpg_action_edge_egcn_hypercond",
                "rpg_action_edge_egcn_plus_public_pred_hypercond",
                "rpg_action_edge_oracle_graph_hypercond",
                "rpg_action_edge_oracle_no_self_hypercond",
                "rpg_action_edge_prev_oracle_graph_hypercond",
                "rpg_action_edge_public_pred_hypercond",
                "rpg_action_edge_public_pred_relation_private_decision_maker",
                "rpg_action_edge_public_memory_hypercond",
                "rpg_action_edge_global_public_pred_hypercond",
                "rpg_action_edge_target_context_hypercond",
                "rpg_delta_relation_hypercond",
            }:
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type in PUBLIC_TRANSFORMER_PAIR_INTERACTION_VARIANTS:
                self.rpg_interaction_input_dim = self.hidden_dim + 2 * self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type in PUBLIC_TRANSFORMER_PAIR_CONCAT_INTERACTION_VARIANTS:
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type in PUBLIC_TRANSFORMER_PRIVATE_HEAD_INPUT_VARIANTS:
                self.rpg_interaction_input_dim = self.hidden_dim + 2 * self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_delta_enemy_token_interaction_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + 2 * self.rpg_relation_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_flat_interaction_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(
                    self.cond_dim, self.hidden_dim * self.rpg_obs_layout["n_enemies"]
                )
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, self.rpg_obs_layout["n_enemies"])
                self.rpg_interaction_encoder = None
                self.rpg_interaction_scorer = None
            elif self.model_type == "rpg_private_interaction_input_hypercond":
                self.rpg_interaction_input_dim = self.hidden_dim + self.rpg_relation_dim
                # Private target input keeps observer-dependent information:
                # movement/action context plus target availability and relative
                # geometry. Observer-invariant entity state such as health,
                # shield, and unit type is deliberately excluded.
                private_input_dim = self.rpg_obs_layout["move_dim"] + 4
                self.rpg_interaction_bottleneck_w = None
                self.rpg_interaction_bottleneck_b = None
                self.rpg_interaction_out_w = nn.Linear(self.cond_dim, self.rpg_interaction_input_dim)
                self.rpg_interaction_out_b = nn.Linear(self.cond_dim, 1)
                self.rpg_interaction_encoder = nn.Sequential(
                    nn.Linear(private_input_dim, self.rpg_interaction_input_dim),
                    nn.ELU(inplace=True),
                    nn.Linear(self.rpg_interaction_input_dim, self.rpg_interaction_input_dim),
                    nn.ELU(inplace=True),
                )
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
                elif self.model_type in {
                    "rpg_linear_interaction_hypercond",
                    "rpg_flat_interaction_hypercond",
                    "rpg_public_relation_hypercond",
                    "rpg_private_interaction_input_hypercond",
                    "rpg_global_filled_obs_hypercond",
                    "rpg_relation_distill_hypercond",
                    "rpg_public_delta_aux_hypercond",
                    *PUBLIC_TRANSFORMER_RELATION_VARIANTS,
                    "rpg_semantic_selfattn_relation_hypercond",
                    "rpg_entity_selfattn_relation_hypercond",
                    "rpg_topk_entity_relation_hypercond",
                    *RPG_TARGETWISE_ABLATION_VARIANTS,
                    *TOKEN_DECISION_HEAD_VARIANTS,
                    "rpg_action_edge_graph_hypercond",
                    "rpg_action_edge_rgcn_hypercond",
                    "rpg_action_edge_egcn_hypercond",
                    "rpg_action_edge_egcn_plus_public_pred_hypercond",
                    "rpg_action_edge_oracle_graph_hypercond",
                    "rpg_action_edge_oracle_no_self_hypercond",
                    "rpg_action_edge_prev_oracle_graph_hypercond",
                    "rpg_action_edge_public_pred_hypercond",
                    "rpg_action_edge_public_pred_relation_private_decision_maker",
                    "rpg_action_edge_public_memory_hypercond",
                    "rpg_action_edge_global_public_pred_hypercond",
                    "rpg_action_edge_target_context_hypercond",
                    "rpg_delta_relation_hypercond",
                }:
                    nn.init.orthogonal_(self.rpg_interaction_out_w.weight, gain=1.0)
                    nn.init.zeros_(self.rpg_interaction_out_w.bias)
                    nn.init.zeros_(self.rpg_interaction_out_b.weight)
                    nn.init.zeros_(self.rpg_interaction_out_b.bias)
                elif self.model_type == "local_linear_interaction_hypercond":
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
            self.rpg_ego_input_dim = None
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

        if self.model_type in (
            PUBLIC_TRANSFORMER_Q_RESIDUAL_HEAD_VARIANTS | PUBLIC_TRANSFORMER_PARAM_RESIDUAL_HEAD_VARIANTS
        ):
            self.rpg_ego_base_maker = nn.Sequential(
                nn.Linear(self.rpg_ego_input_dim, self.hidden_dim),
                nn.ELU(inplace=True),
                nn.Linear(self.hidden_dim, self.rpg_n_ego_actions),
            )
            self.rpg_interaction_base_scorer = nn.Linear(self.rpg_interaction_input_dim, 1)
            self.rpg_residual_ego_gate = nn.Sequential(
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, 1),
            )
            self.rpg_residual_interaction_gate = nn.Sequential(
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, 1),
            )
            nn.init.orthogonal_(self.rpg_ego_base_maker[0].weight, gain=math.sqrt(2.0))
            nn.init.zeros_(self.rpg_ego_base_maker[0].bias)
            nn.init.orthogonal_(self.rpg_ego_base_maker[2].weight, gain=1.0)
            nn.init.zeros_(self.rpg_ego_base_maker[2].bias)
            nn.init.orthogonal_(self.rpg_interaction_base_scorer.weight, gain=1.0)
            nn.init.zeros_(self.rpg_interaction_base_scorer.bias)
            nn.init.zeros_(self.rpg_residual_ego_gate[-1].weight)
            nn.init.constant_(self.rpg_residual_ego_gate[-1].bias, self.stable_residual_gate_bias)
            nn.init.zeros_(self.rpg_residual_interaction_gate[-1].weight)
            nn.init.constant_(
                self.rpg_residual_interaction_gate[-1].bias,
                self.stable_residual_gate_bias,
            )
        else:
            self.rpg_ego_base_maker = None
            self.rpg_interaction_base_scorer = None
            self.rpg_residual_ego_gate = None
            self.rpg_residual_interaction_gate = None

        if self.model_type in PUBLIC_TRANSFORMER_PAIR_FEATURE_INTERACTION_VARIANTS:
            self.rpg_hidden_pair_encoder = nn.Linear(self.hidden_dim, self.rpg_relation_dim)
        else:
            self.rpg_hidden_pair_encoder = None
        if self.model_type in PUBLIC_TRANSFORMER_PAIR_CONCAT_INTERACTION_VARIANTS:
            self.rpg_pair_concat_encoder = nn.Sequential(
                nn.Linear(2 * self.rpg_relation_dim, self.rpg_relation_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.rpg_relation_dim, self.rpg_relation_dim),
            )
        else:
            self.rpg_pair_concat_encoder = None

        if self.model_type in TOKEN_DECISION_HEAD_VARIANTS:
            if self.model_type == "rpg_entity_token_decision_head_hypercond":
                token_interaction_input_dim = self.rpg_relation_dim
            elif self.model_type in (
                PUBLIC_TRANSFORMER_RELATION_PAIR_TOKEN_HEAD_VARIANTS
                | PUBLIC_TRANSFORMER_RELATION_PRIVATE_TOKEN_HEAD_VARIANTS
                | PUBLIC_TRANSFORMER_RELATION_DELTA_TOKEN_HEAD_VARIANTS
            ):
                token_interaction_input_dim = 3 * self.rpg_relation_dim
            else:
                token_interaction_input_dim = 2 * self.rpg_relation_dim
            if self.model_type in RPG_POLICY_RELATION_FUSION_HEAD_VARIANTS:
                self.policy_relation_decision_fuser = nn.Sequential(
                    nn.Linear(self.hidden_dim + self.rpg_relation_dim, self.rpg_relation_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.rpg_relation_dim, self.rpg_relation_dim),
                )
            else:
                self.policy_relation_decision_fuser = None
            self.token_ego_bottleneck_w = nn.Linear(self.cond_dim, self.rpg_relation_dim * self.hidden_dim)
            self.token_ego_bottleneck_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.token_ego_out_w = nn.Linear(self.cond_dim, self.hidden_dim * self.rpg_n_ego_actions)
            self.token_ego_out_b = nn.Linear(self.cond_dim, self.rpg_n_ego_actions)
            self.token_interaction_input_dim = token_interaction_input_dim
            self.token_interaction_out_w = nn.Linear(self.cond_dim, self.token_interaction_input_dim)
            self.token_interaction_out_b = nn.Linear(self.cond_dim, 1)
        else:
            self.token_ego_bottleneck_w = None
            self.token_ego_bottleneck_b = None
            self.token_ego_out_w = None
            self.token_ego_out_b = None
            self.policy_relation_decision_fuser = None
            self.token_interaction_input_dim = None
            self.token_interaction_out_w = None
            self.token_interaction_out_b = None

        if self.model_type in (
            RPG_POST_TARGET_SELECTION_VARIANTS
            | PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS
            | PUBLIC_TRANSFORMER_TARGET_SELECTION_VARIANTS
        ):
            target_source_dim = (
                self.rpg_relation_dim
                if self.model_type in PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS
                else self.hidden_dim
            )
            self.rpg_target_selector = nn.Sequential(
                nn.Linear(target_source_dim + self.cond_dim + self.rpg_relation_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.rpg_target_selector = None

        if self.model_type in {
            "rpg_private_enemy_token_interaction_hypercond",
            *PUBLIC_TRANSFORMER_PRIVATE_HEAD_INPUT_VARIANTS,
        } | PUBLIC_TRANSFORMER_RELATION_PRIVATE_TOKEN_HEAD_VARIANTS:
            self.rpg_private_enemy_token_encoder = nn.Sequential(
                nn.Linear(4, self.rpg_relation_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.rpg_relation_dim, self.rpg_relation_dim),
            )
        else:
            self.rpg_private_enemy_token_encoder = None

        if self.model_type == "rpg_delta_enemy_token_interaction_hypercond" or (
            self.model_type in PUBLIC_TRANSFORMER_RELATION_DELTA_TOKEN_HEAD_VARIANTS
        ):
            self.rpg_delta_enemy_token_encoder = nn.Sequential(
                nn.Linear(self.rpg_obs_layout["enemy_feat_dim"], self.rpg_relation_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.rpg_relation_dim, self.rpg_relation_dim),
            )
        else:
            self.rpg_delta_enemy_token_encoder = None

        if self.model_type in {
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_COARSE_HEAD_VARIANTS,
        }:
            self.self_fine_condition_encoder = nn.Sequential(
                nn.Linear(self.rpg_obs_layout["move_dim"] + self.rpg_obs_layout["own_dim"], self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
        else:
            self.self_fine_condition_encoder = None

        if self.model_type in {
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_action_edge_public_pred_coarse_q_fine_gate_head",
        }:
            self.relation_coarse_bottleneck_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
            self.relation_coarse_bottleneck_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.relation_coarse_out_w = nn.Linear(self.cond_dim, self.hidden_dim * self.n_actions)
            self.relation_coarse_out_b = nn.Linear(self.cond_dim, self.n_actions)
        else:
            self.relation_coarse_bottleneck_w = None
            self.relation_coarse_bottleneck_b = None
            self.relation_coarse_out_w = None
            self.relation_coarse_out_b = None

        if self.model_type == "rpg_relation_coarse_self_fine_head":
            self.self_fine_out_w = nn.Linear(self.cond_dim, self.hidden_dim * self.n_actions)
            self.self_fine_out_b = nn.Linear(self.cond_dim, self.n_actions)
            nn.init.zeros_(self.self_fine_out_w.weight)
            nn.init.zeros_(self.self_fine_out_w.bias)
            nn.init.zeros_(self.self_fine_out_b.weight)
            nn.init.zeros_(self.self_fine_out_b.bias)
        else:
            self.self_fine_out_w = None
            self.self_fine_out_b = None

        if self.model_type in {
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_action_edge_public_pred_coarse_fine_four_layer_head",
        }:
            self.relation_coarse_layer1_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
            self.relation_coarse_layer1_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.relation_coarse_layer2_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
            self.relation_coarse_layer2_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.self_fine_layer3_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
            self.self_fine_layer3_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.self_fine_layer4_w = nn.Linear(self.cond_dim, self.hidden_dim * self.n_actions)
            self.self_fine_layer4_b = nn.Linear(self.cond_dim, self.n_actions)
        else:
            self.relation_coarse_layer1_w = None
            self.relation_coarse_layer1_b = None
            self.relation_coarse_layer2_w = None
            self.relation_coarse_layer2_b = None
            self.self_fine_layer3_w = None
            self.self_fine_layer3_b = None
            self.self_fine_layer4_w = None
            self.self_fine_layer4_b = None

        if self.model_type in {
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_action_edge_public_pred_coarse_q_fine_gate_head",
        }:
            self.self_fine_gate_bottleneck_w = nn.Linear(self.cond_dim, self.hidden_dim * self.hidden_dim)
            self.self_fine_gate_bottleneck_b = nn.Linear(self.cond_dim, self.hidden_dim)
            self.self_fine_gate_out_w = nn.Linear(self.cond_dim, self.hidden_dim * self.n_actions)
            self.self_fine_gate_out_b = nn.Linear(self.cond_dim, self.n_actions)
            for module in (
                self.self_fine_gate_bottleneck_w,
                self.self_fine_gate_bottleneck_b,
                self.self_fine_gate_out_w,
                self.self_fine_gate_out_b,
            ):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        else:
            self.self_fine_gate_bottleneck_w = None
            self.self_fine_gate_bottleneck_b = None
            self.self_fine_gate_out_w = None
            self.self_fine_gate_out_b = None

        if self.model_type == "rpg_relation_prototype_single_head":
            self.prototype_head_hypernet = MLPHyperParameterGenerator(
                embed_dim=self.cond_dim,
                output_dims=[
                    (self.hidden_dim, self.hidden_dim),
                    (self.hidden_dim, self.n_actions),
                ],
                hyper_hidden_dim=self.hyper_mlp_hidden_dim,
            )
        else:
            self.prototype_head_hypernet = None

        if self.model_type in {
            "rpg_public_hyper_private_input_single_head",
            "rpg_private_hyper_public_input_single_head",
            *ACTION_EDGE_PUBLIC_PRED_SINGLE_HEAD_VARIANTS,
        }:
            private_source_dim = self._public_private_private_source_dim()
            if self.model_type in {
                "rpg_public_hyper_private_input_single_head",
                "rpg_private_hyper_public_input_single_head",
            }:
                public_source_dim = self._public_private_public_source_dim()
                self.public_single_condition_encoder = nn.Sequential(
                    nn.Linear(public_source_dim, self.cond_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.cond_dim, self.cond_dim),
                )
            else:
                self.public_single_condition_encoder = None
            self.private_single_condition_encoder = nn.Sequential(
                nn.Linear(private_source_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
            self.public_private_single_head_hypernet = MLPHyperParameterGenerator(
                embed_dim=self.cond_dim,
                output_dims=[
                    (self.hidden_dim + self.cond_dim, self.hidden_dim),
                    (self.hidden_dim, self.n_actions),
                ],
                hyper_hidden_dim=self.hyper_mlp_hidden_dim,
            )
        else:
            self.public_single_condition_encoder = None
            self.private_single_condition_encoder = None
            self.public_private_single_head_hypernet = None

        if self.model_type in ACTION_EDGE_REL_PRIVATE_VARIANTS:
            private_source_dim = self._public_private_private_source_dim()
            self.private_single_condition_encoder = nn.Sequential(
                nn.Linear(private_source_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
            self.relation_private_condition_encoder = nn.Sequential(
                nn.Linear(self.cond_dim * 2, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
            if self.model_type in ACTION_EDGE_REL_PRIVATE_SINGLE_HEAD_VARIANTS:
                self.relation_private_single_head_hypernet = MLPHyperParameterGenerator(
                    embed_dim=self.cond_dim,
                    output_dims=[
                        (self.hidden_dim, self.hidden_dim),
                        (self.hidden_dim, self.n_actions),
                    ],
                    hyper_hidden_dim=self.hyper_mlp_hidden_dim,
                )
            else:
                self.relation_private_single_head_hypernet = None
        else:
            self.relation_private_condition_encoder = None
            self.relation_private_single_head_hypernet = None

        if self.model_type == "rpg_public_delta_aux_hypercond":
            self.public_delta_target_dim = self._public_delta_target_dim()
            self.public_delta_predictor = nn.Sequential(
                nn.Linear(self.rpg_relation_dim, self.rpg_relation_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.rpg_relation_dim, self.public_delta_target_dim),
            )
        else:
            self.public_delta_target_dim = 0
            self.public_delta_predictor = None

        if self.model_type == "rpg_relation_distill_hypercond":
            state_shape = getattr(args, "state_shape", 0)
            if isinstance(state_shape, (tuple, list)):
                self.teacher_state_dim = int(math.prod(state_shape))
            else:
                self.teacher_state_dim = int(state_shape)
            self.teacher_agent_embeddings = nn.Embedding(self.n_agents, self.cond_dim)
            nn.init.orthogonal_(self.teacher_agent_embeddings.weight)
            self.relation_teacher_encoder = nn.Sequential(
                nn.Linear(self.teacher_state_dim + self.cond_dim, self.cond_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
        else:
            self.teacher_state_dim = None
            self.teacher_agent_embeddings = None
            self.relation_teacher_encoder = None

        self.latest_condition = None
        self.latest_aux_loss = None
        self.latest_aux_stats = {}
        self.latest_teacher_q = None
        self.latest_generated_interaction_head = None
        self.latest_generated_interaction_head_graph = None
        self.capture_semantic_parameter_graph = False
        self.capture_generated_interaction_head = bool(
            getattr(args, "save_battle_trace", False)
        )
        self.latest_route_logits = None
        self.latest_route_indices = None
        self.latest_graph_adj = None
        self.latest_graph_nodes = None
        self.latest_relation_ally_attn = None
        self.latest_relation_enemy_attn = None

    def init_hidden(self):
        self.public_memory_obs = None
        hidden_size = self.hidden_dim
        if self.model_type in {
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_flat_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_CAPTURER_VARIANTS,
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            *ACTION_EDGE_REL_PRIVATE_VARIANTS,
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_relation_prototype_single_head",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
            *GRF_PUBLIC_TRANSFORMER_VARIANTS,
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
            "shield_bits_ally": shield_bits_ally,
            "shield_bits_enemy": shield_bits_enemy,
            "unit_type_bits": unit_type_bits,
            "obs_all_health": env_args.get("obs_all_health", True),
            "obs_own_health": env_args.get("obs_own_health", True),
            "obs_last_action": env_args.get("obs_last_action", False),
            "obs_timestep_number": env_args.get("obs_timestep_number", False),
        }

    def _public_private_public_source_dim(self):
        layout = self.rpg_obs_layout
        self_public_dim = 1 + layout["unit_type_bits"]
        ally_public_dim = 1 + layout["unit_type_bits"]
        enemy_public_dim = 1 + layout["unit_type_bits"]
        if layout["obs_own_health"]:
            self_public_dim += 1 + layout["shield_bits_ally"]
        if layout["obs_all_health"]:
            ally_public_dim += 1 + layout["shield_bits_ally"]
            enemy_public_dim += 1 + layout["shield_bits_enemy"]
        return (
            self_public_dim
            + layout["n_allies"] * ally_public_dim
            + layout["n_enemies"] * enemy_public_dim
        )

    def _public_private_private_source_dim(self):
        layout = self.rpg_obs_layout
        return layout["move_dim"] + 4 * layout["n_allies"] + 4 * layout["n_enemies"]

    def _public_private_public_features(self, context):
        obs = context["obs"]
        move_feat, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        del move_feat
        layout = self.rpg_obs_layout
        batch_size, n_agents, _ = obs.shape

        self_parts = [own_feat.new_ones(batch_size, n_agents, 1)]
        own_idx = 0
        if layout["obs_own_health"]:
            self_parts.append(own_feat[:, :, own_idx : own_idx + 1])
            own_idx += 1
            if layout["shield_bits_ally"] > 0:
                self_parts.append(own_feat[:, :, own_idx : own_idx + 1])
                own_idx += 1
        if layout["unit_type_bits"] > 0:
            self_parts.append(own_feat[:, :, own_idx : own_idx + layout["unit_type_bits"]])
        self_public = th.cat(self_parts, dim=-1)

        ally_mask = ally_feat.abs().sum(dim=-1, keepdim=True) > 0
        ally_parts = [ally_mask.float()]
        ally_idx = 4
        if layout["obs_all_health"]:
            ally_parts.append(ally_feat[:, :, :, ally_idx : ally_idx + 1])
            ally_idx += 1
            if layout["shield_bits_ally"] > 0:
                ally_parts.append(ally_feat[:, :, :, ally_idx : ally_idx + 1])
                ally_idx += 1
        if layout["unit_type_bits"] > 0:
            ally_parts.append(ally_feat[:, :, :, ally_idx : ally_idx + layout["unit_type_bits"]])
        ally_public = th.cat(ally_parts, dim=-1) * ally_mask.float()

        enemy_mask = enemy_feat.abs().sum(dim=-1, keepdim=True) > 0
        enemy_parts = [enemy_mask.float()]
        enemy_idx = 4
        if layout["obs_all_health"]:
            enemy_parts.append(enemy_feat[:, :, :, enemy_idx : enemy_idx + 1])
            enemy_idx += 1
            if layout["shield_bits_enemy"] > 0:
                enemy_parts.append(enemy_feat[:, :, :, enemy_idx : enemy_idx + 1])
                enemy_idx += 1
        if layout["unit_type_bits"] > 0:
            enemy_parts.append(enemy_feat[:, :, :, enemy_idx : enemy_idx + layout["unit_type_bits"]])
        enemy_public = th.cat(enemy_parts, dim=-1) * enemy_mask.float()

        return th.cat(
            [
                self_public.reshape(batch_size, n_agents, -1),
                ally_public.reshape(batch_size, n_agents, -1),
                enemy_public.reshape(batch_size, n_agents, -1),
            ],
            dim=-1,
        )

    def _public_delta_target_dim(self):
        layout = self.rpg_obs_layout
        target_dim = 0
        if layout["obs_own_health"]:
            target_dim += 1 + layout["shield_bits_ally"]
        if layout["obs_all_health"]:
            target_dim += layout["n_allies"] * (1 + layout["shield_bits_ally"])
            target_dim += layout["n_enemies"] * (1 + layout["shield_bits_enemy"])
        return target_dim

    def _public_delta_values_and_mask(self, obs, next_obs, next_obs_mask):
        layout = self.rpg_obs_layout
        _, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        _, next_enemy_feat, next_ally_feat, next_own_feat = self._split_rpg_obs(next_obs)
        batch_size, n_agents, _ = obs.shape
        next_valid = next_obs_mask.bool()
        target_parts = []
        mask_parts = []

        if layout["obs_own_health"]:
            self_dim = 1 + layout["shield_bits_ally"]
            self_delta = next_own_feat[:, :, :self_dim] - own_feat[:, :, :self_dim]
            target_parts.append(self_delta)
            mask_parts.append(next_valid.unsqueeze(-1).expand_as(self_delta))

        if layout["obs_all_health"]:
            ally_dim = 1 + layout["shield_bits_ally"]
            ally_idx = 4
            ally_current = ally_feat[:, :, :, ally_idx : ally_idx + ally_dim]
            ally_next = next_ally_feat[:, :, :, ally_idx : ally_idx + ally_dim]
            ally_delta = ally_next - ally_current
            ally_visible = ally_feat.abs().sum(dim=-1) > 0
            ally_next_visible = next_ally_feat.abs().sum(dim=-1) > 0
            ally_mask = (ally_visible & ally_next_visible & next_valid.unsqueeze(-1)).unsqueeze(-1)
            target_parts.append(ally_delta.reshape(batch_size, n_agents, -1))
            mask_parts.append(ally_mask.expand_as(ally_delta).reshape(batch_size, n_agents, -1))

            enemy_dim = 1 + layout["shield_bits_enemy"]
            enemy_idx = 4
            enemy_current = enemy_feat[:, :, :, enemy_idx : enemy_idx + enemy_dim]
            enemy_next = next_enemy_feat[:, :, :, enemy_idx : enemy_idx + enemy_dim]
            enemy_delta = enemy_next - enemy_current
            enemy_visible = enemy_feat.abs().sum(dim=-1) > 0
            enemy_next_visible = next_enemy_feat.abs().sum(dim=-1) > 0
            enemy_mask = (enemy_visible & enemy_next_visible & next_valid.unsqueeze(-1)).unsqueeze(-1)
            target_parts.append(enemy_delta.reshape(batch_size, n_agents, -1))
            mask_parts.append(enemy_mask.expand_as(enemy_delta).reshape(batch_size, n_agents, -1))

        if not target_parts:
            empty = obs.new_zeros(batch_size, n_agents, 0)
            return empty, empty.bool()
        return th.cat(target_parts, dim=-1), th.cat(mask_parts, dim=-1)

    def _public_delta_aux_loss(self, relation_hidden, context):
        next_obs = context.get("next_obs")
        next_obs_mask = context.get("next_obs_mask")
        if next_obs is None or next_obs_mask is None or self.public_delta_predictor is None:
            zero = relation_hidden.new_zeros(())
            return zero, {}

        target, valid_mask = self._public_delta_values_and_mask(context["obs"], next_obs, next_obs_mask)
        pred = self.public_delta_predictor(relation_hidden)
        valid = valid_mask.float()
        if valid.sum() <= 0:
            zero = pred.new_zeros(())
            return zero, {
                "public_delta_loss_raw": zero.detach(),
                "public_delta_mask_frac": zero.detach(),
                "public_delta_target_abs": zero.detach(),
                "public_delta_pred_abs": zero.detach(),
            }

        loss_each = F.smooth_l1_loss(pred, target, reduction="none")
        loss = (loss_each * valid).sum() / valid.sum().clamp(min=1.0)
        valid_denom = valid.sum().clamp(min=1.0)
        stats = {
            "public_delta_loss_raw": loss.detach(),
            "public_delta_mask_frac": valid.mean().detach(),
            "public_delta_target_abs": ((target.abs() * valid).sum() / valid_denom).detach(),
            "public_delta_pred_abs": ((pred.abs() * valid).sum() / valid_denom).detach(),
        }
        return loss, stats

    def _public_private_private_features(self, context):
        move_feat, enemy_feat, ally_feat, _ = self._split_rpg_obs(context["obs"])
        batch_size, n_agents, _ = move_feat.shape
        return th.cat(
            [
                move_feat.reshape(batch_size, n_agents, -1),
                ally_feat[:, :, :, :4].reshape(batch_size, n_agents, -1),
                enemy_feat[:, :, :, :4].reshape(batch_size, n_agents, -1),
            ],
            dim=-1,
        )

    def _init_grf_relation_capturer(self):
        if getattr(self.args, "env", None) not in {
            "academy_pass_and_shoot_with_keeper",
            "academy_3_vs_1_with_keeper",
            "academy_counterattack_easy",
        }:
            raise ValueError(
                "{} currently supports GoMARL GRF academy envs only.".format(self.model_type)
            )
        self.rpg_obs_layout = None
        self.rpg_relation_capturer = GRFPublicPrivateBiasTransformerCapturer(
            n_agents=self.n_agents,
            relation_dim=self.rpg_relation_dim,
            output_dim=self.cond_dim,
            num_heads=self.public_transformer_heads,
            num_layers=self.public_transformer_layers,
            use_absolute_public=self.model_type
            in {
                "grf_abs_public_private_bias_transformer_hypercond",
                "grf_abs_public_private_bias_transformer_two_layer_head_hypercond",
                "grf_abs_public_private_bias_transformer_linear_head_hypercond",
                "grf_abs_public_private_bias_transformer_decision_maker_hypercond",
            },
        )

    def _init_grf_decision_maker_head(self):
        if self.n_actions != 19:
            raise ValueError(
                "{} assumes the default GRF 19-action space, got n_actions={}.".format(
                    self.model_type, self.n_actions
                )
            )

        # GRF actions are not target-indexed. We split them by semantic role:
        # local control, teammate interaction (passes), and opponent/goal interaction.
        self.register_buffer(
            "grf_ego_action_idx",
            th.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 17, 18], dtype=th.long),
            persistent=False,
        )
        self.register_buffer(
            "grf_ally_action_idx",
            th.tensor([9, 10, 11], dtype=th.long),
            persistent=False,
        )
        self.register_buffer(
            "grf_opponent_action_idx",
            th.tensor([12, 16], dtype=th.long),
            persistent=False,
        )

        self.grf_ego_head_input_dim = self.hidden_dim + self.rpg_relation_dim
        self.grf_ally_head_input_dim = self.hidden_dim + 2 * self.rpg_relation_dim
        self.grf_opponent_head_input_dim = self.hidden_dim + 2 * self.rpg_relation_dim
        self.grf_decision_hyper = nn.ModuleDict()
        for branch, input_dim, output_dim in (
            ("ego", self.grf_ego_head_input_dim, int(self.grf_ego_action_idx.numel())),
            ("ally", self.grf_ally_head_input_dim, int(self.grf_ally_action_idx.numel())),
            ("opponent", self.grf_opponent_head_input_dim, int(self.grf_opponent_action_idx.numel())),
        ):
            self.grf_decision_hyper[f"{branch}_w1"] = nn.Linear(
                self.cond_dim, input_dim * self.hidden_dim
            )
            self.grf_decision_hyper[f"{branch}_b1"] = nn.Linear(self.cond_dim, self.hidden_dim)
            self.grf_decision_hyper[f"{branch}_w2"] = nn.Linear(
                self.cond_dim, self.hidden_dim * output_dim
            )
            self.grf_decision_hyper[f"{branch}_b2"] = nn.Linear(self.cond_dim, output_dim)

    def _init_rpg_relation_capturer(self):
        self.rpg_obs_layout = self._build_rpg_obs_layout()
        capturer_cls = RPGInspiredRelationCapturer
        if self.model_type == "two_graph_gat_hypercond":
            capturer_cls = TwoGraphGATRelationCapturer
        elif self.model_type == "hetero_gat_hypercond":
            capturer_cls = HeteroGATRelationCapturer
        elif self.model_type == "rpg_public_relation_hypercond":
            capturer_cls = PublicRPGRelationCapturer
        elif self.model_type in PUBLIC_TRANSFORMER_CAPTURER_VARIANTS:
            capturer_cls = PublicTransformerRelationCapturer
        elif self.model_type == "rpg_semantic_selfattn_relation_hypercond":
            capturer_cls = SemanticSelfAttentionRelationCapturer
        elif self.model_type == "rpg_entity_selfattn_relation_hypercond":
            capturer_cls = EntitySelfAttentionRelationCapturer
        elif self.model_type == "rpg_topk_entity_relation_hypercond":
            capturer_cls = TopKEntitySelfAttentionRelationCapturer
        elif self.model_type in RPG_PRE_RELATION_SELECTION_VARIANTS:
            capturer_cls = PreSelectRPGRelationCapturer
        elif self.model_type in {
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            *ACTION_EDGE_REL_PRIVATE_VARIANTS,
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
        }:
            capturer_cls = ActionEdgeGraphRelationCapturer
        elif self.model_type == "rpg_delta_relation_hypercond":
            layout_obs_dim = (
                self.rpg_obs_layout["move_dim"]
                + self.rpg_obs_layout["n_enemies"] * self.rpg_obs_layout["enemy_feat_dim"]
                + self.rpg_obs_layout["n_allies"] * self.rpg_obs_layout["ally_feat_dim"]
                + self.rpg_obs_layout["own_dim"]
            )
            self.rpg_relation_capturer = DeltaObservationRelationCapturer(
                move_dim=self.rpg_obs_layout["move_dim"],
                own_dim=self.rpg_obs_layout["own_dim"],
                ally_feat_dim=self.rpg_obs_layout["ally_feat_dim"],
                enemy_feat_dim=self.rpg_obs_layout["enemy_feat_dim"],
                relation_dim=self.rpg_relation_dim,
                output_dim=self.cond_dim,
                obs_dim=layout_obs_dim,
            )
            return

        capturer_kwargs = {
            "move_dim": self.rpg_obs_layout["move_dim"],
            "own_dim": self.rpg_obs_layout["own_dim"],
            "ally_feat_dim": self.rpg_obs_layout["ally_feat_dim"],
            "enemy_feat_dim": self.rpg_obs_layout["enemy_feat_dim"],
            "relation_dim": self.rpg_relation_dim,
            "output_dim": self.cond_dim,
        }
        if capturer_cls is PublicRPGRelationCapturer:
            capturer_kwargs.update(
                unit_type_bits=self.rpg_obs_layout["unit_type_bits"],
                shield_bits_ally=self.rpg_obs_layout["shield_bits_ally"],
                shield_bits_enemy=self.rpg_obs_layout["shield_bits_enemy"],
                obs_all_health=self.rpg_obs_layout["obs_all_health"],
                obs_own_health=self.rpg_obs_layout["obs_own_health"],
                obs_last_action=self.rpg_obs_layout["obs_last_action"],
                n_actions=self.n_actions,
            )
        elif capturer_cls is PublicTransformerRelationCapturer:
            capturer_kwargs.update(
                unit_type_bits=self.rpg_obs_layout["unit_type_bits"],
                shield_bits_ally=self.rpg_obs_layout["shield_bits_ally"],
                shield_bits_enemy=self.rpg_obs_layout["shield_bits_enemy"],
                obs_all_health=self.rpg_obs_layout["obs_all_health"],
                obs_own_health=self.rpg_obs_layout["obs_own_health"],
                obs_last_action=self.rpg_obs_layout["obs_last_action"],
                n_actions=self.n_actions,
                n_allies=self.rpg_obs_layout["n_allies"],
                n_enemies=self.rpg_obs_layout["n_enemies"],
                mode=PUBLIC_TRANSFORMER_MODE_BY_MODEL[self.model_type],
                num_heads=self.public_transformer_heads,
                num_layers=self.public_transformer_layers,
                use_encoded_enemy_tokens=self.model_type
                in {
                    "rpg_public_private_bias_past_delta_token_transformer_enemy_slot_hypercond",
                    "rpg_public_private_token_past_delta_bias_transformer_enemy_slot_hypercond",
                    "rpg_public_private_bias_transformer_slot_token_head_hypercond",
                },
                merge_friendly_public_side=self.model_type in PUBLIC_TRANSFORMER_FRIEND_MERGED_VARIANTS,
                private_owner_side=self.model_type
                in (
                    {
                        "rpg_public_private_owner_bias_transformer_hypercond",
                        "rpg_public_private_selfattn_bias_transformer_hypercond",
                    }
                    | PUBLIC_TRANSFORMER_SIMPLE_BIAS_FAMILY
                ),
                private_bias_style=(
                    "pair_mlp_no_side"
                    if self.model_type == "rpg_public_private_owner_bias_transformer_hypercond"
                    else "simple"
                    if self.model_type in PUBLIC_TRANSFORMER_SIMPLE_BIAS_FAMILY
                    else "selfattn_simple"
                    if self.model_type == "rpg_public_private_selfattn_bias_transformer_hypercond"
                    else "pair_mlp"
                ),
                semantic_router_mode=SEMANTIC_ROUTER_MODE_BY_MODEL.get(self.model_type),
                semantic_router_ema=self.semantic_router_ema,
                semantic_router_threshold=self.semantic_router_threshold,
                semantic_router_temperature=self.semantic_router_temperature,
                semantic_router_warmup_steps=self.semantic_router_warmup_steps,
                semantic_router_freeze_steps=self.semantic_router_freeze_steps,
            )
        elif capturer_cls is SemanticSelfAttentionRelationCapturer:
            capturer_kwargs.update(
                unit_type_bits=self.rpg_obs_layout["unit_type_bits"],
                shield_bits_ally=self.rpg_obs_layout["shield_bits_ally"],
                shield_bits_enemy=self.rpg_obs_layout["shield_bits_enemy"],
                obs_all_health=self.rpg_obs_layout["obs_all_health"],
                obs_own_health=self.rpg_obs_layout["obs_own_health"],
                obs_last_action=self.rpg_obs_layout["obs_last_action"],
                obs_timestep_number=self.rpg_obs_layout["obs_timestep_number"],
                n_actions=self.n_actions,
            )
        elif capturer_cls is TopKEntitySelfAttentionRelationCapturer:
            capturer_kwargs.update(topk=self.relation_topk)
        elif capturer_cls is PreSelectRPGRelationCapturer:
            capturer_kwargs.update(
                selection_mode=(
                    "topk"
                    if self.model_type == "rpg_pre_topk_entity_relation_hypercond"
                    else "threshold"
                ),
                topk=self.relation_topk,
                threshold=self.pre_relation_threshold,
            )
        elif capturer_cls is ActionEdgeGraphRelationCapturer:
            layout_obs_dim = (
                self.rpg_obs_layout["move_dim"]
                + self.rpg_obs_layout["n_enemies"] * self.rpg_obs_layout["enemy_feat_dim"]
                + self.rpg_obs_layout["n_allies"] * self.rpg_obs_layout["ally_feat_dim"]
                + self.rpg_obs_layout["own_dim"]
            )
            capturer_kwargs.update(
                obs_dim=layout_obs_dim,
                n_agents=self.n_agents,
                n_actions=self.n_actions,
                n_enemies=self.rpg_obs_layout["n_enemies"],
                unit_type_bits=self.rpg_obs_layout["unit_type_bits"],
                shield_bits_ally=self.rpg_obs_layout["shield_bits_ally"],
                shield_bits_enemy=self.rpg_obs_layout["shield_bits_enemy"],
                obs_all_health=self.rpg_obs_layout["obs_all_health"],
                obs_own_health=self.rpg_obs_layout["obs_own_health"],
                graph_encoder_type={
                    "rpg_action_edge_rgcn_hypercond": "rgcn",
                    "rpg_action_edge_egcn_hypercond": "egcn",
                    "rpg_action_edge_egcn_plus_public_pred_hypercond": "egcn_plus",
                    "rpg_action_edge_graphormer_relation_private_single_head": "graphormer",
                    "rpg_action_edge_graphit_relation_private_single_head": "graphit",
                    "rpg_action_edge_edgeset_relation_private_single_head": "edgeset",
                    "rpg_action_edge_motif_transformer_relation_private_single_head": "motif_transformer",
                }.get(self.model_type, "pool"),
                use_oracle_edges=self.model_type in {
                    "rpg_action_edge_oracle_graph_hypercond",
                    "rpg_action_edge_oracle_no_self_hypercond",
                    "rpg_action_edge_prev_oracle_graph_hypercond",
                },
                oracle_edge_mode=(
                    "previous" if self.model_type == "rpg_action_edge_prev_oracle_graph_hypercond" else "current"
                ),
                predictor_input_mode=(
                    "public"
                    if self.model_type in {
                        "rpg_action_edge_public_pred_hypercond",
                        *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
                        *ACTION_EDGE_REL_PRIVATE_VARIANTS,
                        "rpg_action_edge_public_memory_hypercond",
                        "rpg_action_edge_global_public_pred_hypercond",
                        "rpg_action_edge_egcn_plus_public_pred_hypercond",
                    }
                    else "full"
                ),
                use_public_memory=self.model_type == "rpg_action_edge_public_memory_hypercond",
                return_target_context=self.model_type
                in {"rpg_action_edge_target_context_hypercond", "rpg_action_edge_egcn_plus_public_pred_hypercond"},
                no_self_identity=self.model_type == "rpg_action_edge_oracle_no_self_hypercond",
            )
        self.rpg_relation_capturer = capturer_cls(**capturer_kwargs)

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

    def _split_sc2_state(self, state):
        if getattr(self.args, "env_args", {}).get("obs_instead_of_state", False):
            raise ValueError(
                "rpg_global_filled_obs_hypercond requires the standard SMAC global state, "
                "but env_args.obs_instead_of_state=True."
            )
        layout = self.rpg_obs_layout
        batch_size = state.size(0)
        state_ally_dim = 4 + layout["shield_bits_ally"] + layout["unit_type_bits"]
        state_enemy_dim = 3 + layout["shield_bits_enemy"] + layout["unit_type_bits"]

        idx = 0
        ally_total = self.n_agents * state_ally_dim
        ally_state = state[:, idx : idx + ally_total].view(batch_size, self.n_agents, state_ally_dim)
        idx += ally_total

        enemy_total = layout["n_enemies"] * state_enemy_dim
        enemy_state = state[:, idx : idx + enemy_total].view(batch_size, layout["n_enemies"], state_enemy_dim)
        idx += enemy_total

        last_action = None
        env_args = getattr(self.args, "env_args", {})
        if env_args.get("state_last_action", True):
            action_total = self.n_agents * self.n_actions
            if state.size(-1) >= idx + action_total:
                last_action = state[:, idx : idx + action_total].view(batch_size, self.n_agents, self.n_actions)
        return ally_state, enemy_state, last_action

    def _build_global_filled_obs(self, context):
        state = context.get("state")
        if state is None:
            return context["obs"]

        obs = context["obs"]
        move_feat, local_enemy, local_ally, own_feat = self._split_rpg_obs(obs)
        ally_state, enemy_state, state_last_action = self._split_sc2_state(state.reshape(state.size(0), -1))
        layout = self.rpg_obs_layout
        batch_size, n_agents, _ = move_feat.shape
        device = obs.device

        self_state = ally_state[:, :n_agents]
        self_alive = self_state[:, :, 0] > 0
        self_x = self_state[:, :, 2]
        self_y = self_state[:, :, 3]

        enemy_alive = enemy_state[:, :, 0] > 0
        enemy_dx = enemy_state[:, None, :, 1] - self_x[:, :, None]
        enemy_dy = enemy_state[:, None, :, 2] - self_y[:, :, None]
        enemy_dist = th.sqrt(enemy_dx.pow(2) + enemy_dy.pow(2)).clamp(max=10.0)
        enemy_valid = (self_alive[:, :, None] & enemy_alive[:, None, :]).float()

        enemy_feat = local_enemy.clone()
        # Keep the local attack-availability flag; fill only relational entity
        # information from centralized state so action availability semantics do
        # not become privileged.
        enemy_feat[:, :, :, 1] = enemy_dist * enemy_valid
        enemy_feat[:, :, :, 2] = enemy_dx * enemy_valid
        enemy_feat[:, :, :, 3] = enemy_dy * enemy_valid
        obs_idx = 4
        state_idx = 3
        if layout["obs_all_health"]:
            enemy_feat[:, :, :, obs_idx] = enemy_state[:, None, :, 0] * enemy_valid
            obs_idx += 1
            if layout["shield_bits_enemy"] > 0:
                enemy_feat[:, :, :, obs_idx] = enemy_state[:, None, :, state_idx] * enemy_valid
                obs_idx += 1
                state_idx += 1
        if layout["unit_type_bits"] > 0:
            enemy_type = enemy_state[:, None, :, state_idx : state_idx + layout["unit_type_bits"]]
            enemy_feat[:, :, :, obs_idx : obs_idx + layout["unit_type_bits"]] = enemy_type * enemy_valid.unsqueeze(-1)

        ally_feat = local_ally.clone()
        ally_ids = th.arange(n_agents, device=device)
        other_agent_ids = []
        for agent_id in range(n_agents):
            other_agent_ids.append(ally_ids[ally_ids != agent_id])
        other_agent_ids = th.stack(other_agent_ids, dim=0)

        gathered_ally = ally_state[:, other_agent_ids]
        ally_alive = gathered_ally[:, :, :, 0] > 0
        ally_dx = gathered_ally[:, :, :, 2] - self_x[:, :, None]
        ally_dy = gathered_ally[:, :, :, 3] - self_y[:, :, None]
        ally_dist = th.sqrt(ally_dx.pow(2) + ally_dy.pow(2)).clamp(max=10.0)
        ally_valid = (self_alive[:, :, None] & ally_alive).float()

        ally_feat[:, :, :, 0] = ally_valid
        ally_feat[:, :, :, 1] = ally_dist * ally_valid
        ally_feat[:, :, :, 2] = ally_dx * ally_valid
        ally_feat[:, :, :, 3] = ally_dy * ally_valid
        obs_idx = 4
        state_idx = 4
        if layout["obs_all_health"]:
            ally_feat[:, :, :, obs_idx] = gathered_ally[:, :, :, 0] * ally_valid
            obs_idx += 1
            if layout["shield_bits_ally"] > 0:
                ally_feat[:, :, :, obs_idx] = gathered_ally[:, :, :, state_idx] * ally_valid
                obs_idx += 1
                state_idx += 1
        if layout["unit_type_bits"] > 0:
            ally_type = gathered_ally[:, :, :, state_idx : state_idx + layout["unit_type_bits"]]
            ally_feat[:, :, :, obs_idx : obs_idx + layout["unit_type_bits"]] = ally_type * ally_valid.unsqueeze(-1)
            obs_idx += layout["unit_type_bits"]
        if layout["obs_last_action"] and state_last_action is not None:
            gathered_action = state_last_action[:, other_agent_ids]
            ally_feat[:, :, :, obs_idx : obs_idx + self.n_actions] = gathered_action * ally_valid.unsqueeze(-1)

        return th.cat(
            [
                move_feat.reshape(batch_size, n_agents, -1),
                enemy_feat.reshape(batch_size, n_agents, -1),
                ally_feat.reshape(batch_size, n_agents, -1),
                own_feat.reshape(batch_size, n_agents, -1),
            ],
            dim=-1,
        )

    def _build_global_public_filled_obs(self, context, obs_key="obs", state_key="state"):
        state = context.get(state_key)
        obs = context.get(obs_key)
        if state is None or obs is None:
            return obs if obs is not None else context["obs"]

        move_feat, local_enemy, local_ally, own_feat = self._split_rpg_obs(obs)
        ally_state, enemy_state, _ = self._split_sc2_state(state.reshape(state.size(0), -1))
        layout = self.rpg_obs_layout
        batch_size, n_agents, _ = move_feat.shape
        device = obs.device

        self_state = ally_state[:, :n_agents]
        self_alive = self_state[:, :, 0] > 0

        own_public = own_feat.clone()
        own_idx = 0
        state_idx = 0
        if layout["obs_own_health"]:
            own_public[:, :, own_idx] = self_state[:, :, state_idx] * self_alive.float()
            own_idx += 1
            state_idx = 4
            if layout["shield_bits_ally"] > 0:
                own_public[:, :, own_idx : own_idx + layout["shield_bits_ally"]] = (
                    self_state[:, :, state_idx : state_idx + layout["shield_bits_ally"]]
                    * self_alive.unsqueeze(-1).float()
                )
                own_idx += layout["shield_bits_ally"]
                state_idx += layout["shield_bits_ally"]
        else:
            state_idx = 4 + layout["shield_bits_ally"]
        if layout["unit_type_bits"] > 0:
            own_public[:, :, own_idx : own_idx + layout["unit_type_bits"]] = (
                self_state[:, :, state_idx : state_idx + layout["unit_type_bits"]]
                * self_alive.unsqueeze(-1).float()
            )

        enemy_alive = enemy_state[:, :, 0] > 0
        enemy_valid = (self_alive[:, :, None] & enemy_alive[:, None, :]).float()
        enemy_feat = local_enemy.clone()
        obs_idx = 4
        state_idx = 3
        if layout["obs_all_health"]:
            enemy_feat[:, :, :, obs_idx] = enemy_state[:, None, :, 0] * enemy_valid
            obs_idx += 1
            if layout["shield_bits_enemy"] > 0:
                enemy_feat[:, :, :, obs_idx : obs_idx + layout["shield_bits_enemy"]] = (
                    enemy_state[:, None, :, state_idx : state_idx + layout["shield_bits_enemy"]]
                    * enemy_valid.unsqueeze(-1)
                )
                obs_idx += layout["shield_bits_enemy"]
                state_idx += layout["shield_bits_enemy"]
        if layout["unit_type_bits"] > 0:
            enemy_feat[:, :, :, obs_idx : obs_idx + layout["unit_type_bits"]] = (
                enemy_state[:, None, :, state_idx : state_idx + layout["unit_type_bits"]]
                * enemy_valid.unsqueeze(-1)
            )

        ally_feat = local_ally.clone()
        ally_ids = th.arange(n_agents, device=device)
        other_agent_ids = []
        for agent_id in range(n_agents):
            other_agent_ids.append(ally_ids[ally_ids != agent_id])
        other_agent_ids = th.stack(other_agent_ids, dim=0)
        gathered_ally = ally_state[:, other_agent_ids]
        ally_alive = gathered_ally[:, :, :, 0] > 0
        ally_valid = (self_alive[:, :, None] & ally_alive).float()

        obs_idx = 4
        state_idx = 4
        if layout["obs_all_health"]:
            ally_feat[:, :, :, obs_idx] = gathered_ally[:, :, :, 0] * ally_valid
            obs_idx += 1
            if layout["shield_bits_ally"] > 0:
                ally_feat[:, :, :, obs_idx : obs_idx + layout["shield_bits_ally"]] = (
                    gathered_ally[:, :, :, state_idx : state_idx + layout["shield_bits_ally"]]
                    * ally_valid.unsqueeze(-1)
                )
                obs_idx += layout["shield_bits_ally"]
                state_idx += layout["shield_bits_ally"]
        if layout["unit_type_bits"] > 0:
            ally_feat[:, :, :, obs_idx : obs_idx + layout["unit_type_bits"]] = (
                gathered_ally[:, :, :, state_idx : state_idx + layout["unit_type_bits"]]
                * ally_valid.unsqueeze(-1)
            )

        return th.cat(
            [
                move_feat.reshape(batch_size, n_agents, -1),
                enemy_feat.reshape(batch_size, n_agents, -1),
                ally_feat.reshape(batch_size, n_agents, -1),
                own_public.reshape(batch_size, n_agents, -1),
            ],
            dim=-1,
        )

    def _build_public_memory_filled_obs(self, obs):
        if obs is None:
            return obs
        if self.public_memory_obs is None or self.public_memory_obs.shape != obs.shape:
            self.public_memory_obs = th.zeros_like(obs)

        prev_memory = self.public_memory_obs.to(device=obs.device, dtype=obs.dtype)
        move_feat, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        mem_move, mem_enemy, mem_ally, mem_own = self._split_rpg_obs(prev_memory)

        enemy_mask = enemy_feat.abs().sum(dim=-1, keepdim=True) > 0
        ally_mask = ally_feat.abs().sum(dim=-1, keepdim=True) > 0
        self_mask = own_feat.abs().sum(dim=-1, keepdim=True) > 0

        filled_enemy = th.where(enemy_mask, enemy_feat, mem_enemy)
        filled_ally = th.where(ally_mask, ally_feat, mem_ally)
        filled_own = th.where(self_mask, own_feat, mem_own)
        filled_obs = th.cat(
            [
                move_feat.reshape(obs.size(0), self.n_agents, -1),
                filled_enemy.reshape(obs.size(0), self.n_agents, -1),
                filled_ally.reshape(obs.size(0), self.n_agents, -1),
                filled_own.reshape(obs.size(0), self.n_agents, -1),
            ],
            dim=-1,
        )
        self.public_memory_obs = filled_obs.detach()
        return filled_obs

    def _build_local_structured_condition(self, hidden, context):
        condition = self.local_condition_encoder(self._build_local_source(hidden, context))
        _, enemy_feat, _, _ = self._split_rpg_obs(context["obs"])
        enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
        enemy_tokens = self.rpg_relation_capturer.enemy_encoder(enemy_feat) * enemy_mask.unsqueeze(-1).float()
        return condition, enemy_tokens, enemy_mask

    def _build_rpg_condition(self, context, relation_hidden, test_mode=False):
        if self.model_type == "rpg_global_filled_obs_hypercond" and not test_mode:
            obs = self._build_global_filled_obs(context)
        elif self.model_type in PUBLIC_TRANSFORMER_GLOBAL_PUBLIC_VARIANTS and (
            not test_mode or self.model_type in PUBLIC_TRANSFORMER_EVAL_GLOBAL_VARIANTS
        ):
            obs = self._build_global_public_filled_obs(context, obs_key="obs", state_key="state")
        elif self.model_type in PUBLIC_TRANSFORMER_MEMORY_EVAL_VARIANTS and test_mode:
            obs = self._build_public_memory_filled_obs(context["obs"])
        else:
            obs = context["obs"]
        move_feat, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        if isinstance(self.rpg_relation_capturer, PublicTransformerRelationCapturer):
            if self.model_type in PUBLIC_TRANSFORMER_GLOBAL_PUBLIC_VARIANTS and (
                not test_mode or self.model_type in PUBLIC_TRANSFORMER_EVAL_GLOBAL_VARIANTS
            ):
                prev_obs = self._build_global_public_filled_obs(
                    context, obs_key="prev_obs", state_key="prev_state"
                )
            elif self.model_type in PUBLIC_TRANSFORMER_MEMORY_EVAL_VARIANTS and test_mode:
                prev_obs = context.get("prev_obs")
            else:
                prev_obs = context.get("prev_obs")
            next_obs = context.get("next_obs")
            if prev_obs is None:
                prev_obs = th.zeros_like(obs)
            if next_obs is None:
                next_obs = th.zeros_like(obs)
            prev_move, prev_enemy, prev_ally, prev_own = self._split_rpg_obs(prev_obs)
            next_move, next_enemy, next_ally, next_own = self._split_rpg_obs(next_obs)
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
                prev_self_feat=th.cat([prev_move, prev_own], dim=-1),
                prev_ally_feat=prev_ally,
                prev_enemy_feat=prev_enemy,
                next_self_feat=th.cat([next_move, next_own], dim=-1),
                next_ally_feat=next_ally,
                next_enemy_feat=next_enemy,
                next_obs_mask=context.get("next_obs_mask"),
            )
        else:
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
        self.latest_aux_stats.update(
            getattr(self.rpg_relation_capturer, "latest_aux_stats", {})
        )
        return condition, new_relation_hidden, enemy_tokens, enemy_mask

    def _build_action_edge_graph_condition(self, context, relation_hidden, test_mode=False):
        obs = context["obs"]
        prev_obs = context.get("prev_obs")
        if prev_obs is None:
            prev_obs = th.zeros_like(obs)
        if self.model_type == "rpg_action_edge_global_public_pred_hypercond" and not test_mode:
            obs = self._build_global_filled_obs({"obs": obs, "state": context.get("state")})
            prev_state = context.get("prev_state")
            if prev_state is not None:
                prev_obs = self._build_global_filled_obs({"obs": prev_obs, "state": prev_state})
        move_feat, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        target_actions = None if test_mode else context.get("action_targets")
        action_target_mask = None if test_mode else context.get("action_target_mask")
        prev_action_targets = None
        if not test_mode and context.get("prev_action") is not None:
            prev_action_targets = context["prev_action"].argmax(dim=-1)
        (
            condition,
            new_relation_hidden,
            ally_attn,
            enemy_attn,
            enemy_tokens,
            enemy_mask,
            action_loss,
        ) = self.rpg_relation_capturer(
            self_feat=self_feat,
            ally_feat=ally_feat,
            enemy_feat=enemy_feat,
            prev_relation_hidden=relation_hidden,
            obs=obs,
            prev_obs=prev_obs,
            target_actions=target_actions,
            action_target_mask=action_target_mask,
            prev_action_targets=prev_action_targets,
        )
        self.latest_relation_ally_attn = ally_attn.detach()
        self.latest_relation_enemy_attn = enemy_attn.detach()
        return condition, new_relation_hidden, enemy_tokens, enemy_mask, action_loss

    def _build_rpg_delta_condition(self, context):
        obs = context["obs"]
        prev_obs = context.get("prev_obs")
        if prev_obs is None:
            prev_obs = th.zeros_like(obs)
        move_feat, enemy_feat, ally_feat, own_feat = self._split_rpg_obs(obs)
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        delta_obs = obs - prev_obs
        condition, ally_attn, enemy_attn, enemy_tokens, enemy_mask = self.rpg_relation_capturer(
            self_feat=self_feat,
            ally_feat=ally_feat,
            enemy_feat=enemy_feat,
            delta_obs=delta_obs,
        )
        self.latest_relation_ally_attn = ally_attn.detach()
        self.latest_relation_enemy_attn = enemy_attn.detach()
        return condition, enemy_tokens, enemy_mask

    def _build_global_graph_condition(self, context):
        move_feat, enemy_feat, _, own_feat = self._split_rpg_obs(context["obs"])
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        return self.global_graph_relation_encoder(self_feat, enemy_feat)

    def _build_relation_teacher_condition(self, context):
        state = context.get("state")
        if state is None:
            return None
        batch_size = state.size(0)
        flat_state = state.reshape(batch_size, -1)
        if flat_state.size(-1) != self.teacher_state_dim:
            raise ValueError(
                "rpg_relation_distill_hypercond expected state_dim={}, got {}.".format(
                    self.teacher_state_dim, flat_state.size(-1)
                )
            )
        state_rep = flat_state.unsqueeze(1).expand(-1, self.n_agents, -1)
        agent_ids = th.arange(self.n_agents, device=state.device).view(1, self.n_agents).expand(batch_size, -1)
        agent_embed = self.teacher_agent_embeddings(agent_ids)
        return self.relation_teacher_encoder(th.cat([state_rep, agent_embed], dim=-1))

    def _relation_distill_loss(self, student_condition, teacher_condition):
        student = F.normalize(student_condition, p=2, dim=-1)
        teacher = F.normalize(teacher_condition.detach(), p=2, dim=-1)
        return F.mse_loss(student, teacher)

    def _route_from_logits(self, route_logits, test_mode):
        route_num = self.route_codebook.size(0)
        if test_mode:
            route_index = route_logits.argmax(dim=-1)
            route_weight = F.one_hot(route_index, num_classes=route_num).float()
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
            "local_linear_interaction_hypercond",
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_flat_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_CAPTURER_VARIANTS,
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            *ACTION_EDGE_REL_PRIVATE_VARIANTS,
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_delta_relation_hypercond",
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_relation_prototype_single_head",
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

    def _apply_grf_linear_head(self, hidden, condition):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = condition.reshape(batch_size * n_agents, -1)
        out_w = self.grf_linear_head_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        out_b = self.grf_linear_head_b(flat_condition).view(batch_size * n_agents, 1, self.n_actions)
        q = th.bmm(flat_hidden, out_w) + out_b
        return q.view(batch_size, n_agents, self.n_actions)

    def _apply_grf_generated_branch(self, branch, branch_input, condition, output_dim):
        batch_size, n_agents, input_dim = branch_input.shape
        flat_input = branch_input.reshape(batch_size * n_agents, 1, input_dim)
        flat_condition = condition.reshape(batch_size * n_agents, -1)

        w1 = self.grf_decision_hyper[f"{branch}_w1"](flat_condition).view(
            batch_size * n_agents, input_dim, self.hidden_dim
        )
        b1 = self.grf_decision_hyper[f"{branch}_b1"](flat_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        w2 = self.grf_decision_hyper[f"{branch}_w2"](flat_condition).view(
            batch_size * n_agents, self.hidden_dim, output_dim
        )
        b2 = self.grf_decision_hyper[f"{branch}_b2"](flat_condition).view(
            batch_size * n_agents, 1, output_dim
        )

        mid = F.elu(th.bmm(flat_input, w1) + b1)
        q = th.bmm(mid, w2) + b2
        return q.view(batch_size, n_agents, output_dim)

    def _apply_grf_decision_maker_head(self, hidden, condition):
        self_token = getattr(self.rpg_relation_capturer, "latest_self_token", None)
        ally_tokens = getattr(self.rpg_relation_capturer, "latest_ally_tokens", None)
        opponent_tokens = getattr(self.rpg_relation_capturer, "latest_opponent_tokens", None)
        if self_token is None or ally_tokens is None or opponent_tokens is None:
            raise RuntimeError(
                "{} requires GRF transformer entity tokens from the relation capturer.".format(
                    self.model_type
                )
            )

        if ally_tokens.size(2) > 0:
            ally_context = ally_tokens.mean(dim=2)
        else:
            ally_context = self_token.new_zeros(self_token.shape)
        if opponent_tokens.size(2) > 0:
            opponent_context = opponent_tokens.mean(dim=2)
        else:
            opponent_context = self_token.new_zeros(self_token.shape)

        ego_input = th.cat([hidden, self_token], dim=-1)
        ally_input = th.cat([hidden, self_token, ally_context], dim=-1)
        opponent_input = th.cat([hidden, self_token, opponent_context], dim=-1)

        q = hidden.new_zeros(hidden.size(0), hidden.size(1), self.n_actions)
        q[:, :, self.grf_ego_action_idx] = self._apply_grf_generated_branch(
            "ego", ego_input, condition, int(self.grf_ego_action_idx.numel())
        )
        q[:, :, self.grf_ally_action_idx] = self._apply_grf_generated_branch(
            "ally", ally_input, condition, int(self.grf_ally_action_idx.numel())
        )
        q[:, :, self.grf_opponent_action_idx] = self._apply_grf_generated_branch(
            "opponent", opponent_input, condition, int(self.grf_opponent_action_idx.numel())
        )
        return q

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

    def _apply_generated_mlp_head(self, hidden, condition, generator):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = condition.reshape(batch_size * n_agents, -1)
        weights, biases = generator(flat_condition)
        generated_head = th.cat(
            [
                item.reshape(batch_size * n_agents, -1)
                for pair in zip(weights, biases)
                for item in pair
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)

        current = flat_hidden
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            current = th.bmm(current, weight) + bias
            if layer_idx != len(weights) - 1:
                current = F.elu(current)
        return current.view(batch_size, n_agents, self.n_actions)

    def _apply_generated_mlp_head_from_input(self, head_input, condition, generator):
        batch_size, n_agents, _ = head_input.shape
        flat_input = head_input.reshape(batch_size * n_agents, 1, -1)
        flat_condition = condition.reshape(batch_size * n_agents, -1)
        weights, biases = generator(flat_condition)
        generated_head = th.cat(
            [
                item.reshape(batch_size * n_agents, -1)
                for pair in zip(weights, biases)
                for item in pair
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)

        current = flat_input
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            current = th.bmm(current, weight) + bias
            if layer_idx != len(weights) - 1:
                current = F.elu(current)
        return current.view(batch_size, n_agents, self.n_actions)

    def _apply_public_private_single_head(self, hidden, context):
        public_condition = self.public_single_condition_encoder(
            self._public_private_public_features(context)
        )
        private_condition = self.private_single_condition_encoder(
            self._public_private_private_features(context)
        )
        if self.model_type == "rpg_public_hyper_private_input_single_head":
            generator_condition = public_condition
            input_condition = private_condition
        else:
            generator_condition = private_condition
            input_condition = public_condition
        self.latest_condition = generator_condition.detach()
        head_input = th.cat([hidden, input_condition], dim=-1)
        return self._apply_generated_mlp_head_from_input(
            head_input, generator_condition, self.public_private_single_head_hypernet
        )

    def _apply_action_edge_public_pred_single_head(self, hidden, relation_condition, context):
        private_condition = self.private_single_condition_encoder(
            self._public_private_private_features(context)
        )
        if self.model_type == "rpg_action_edge_public_pred_public_hyper_private_input_single_head":
            generator_condition = relation_condition
            input_condition = private_condition
        else:
            generator_condition = private_condition
            input_condition = relation_condition
        self.latest_condition = generator_condition.detach()
        head_input = th.cat([hidden, input_condition], dim=-1)
        return self._apply_generated_mlp_head_from_input(
            head_input, generator_condition, self.public_private_single_head_hypernet
        )

    def _action_edge_relation_private_condition(self, relation_condition, context):
        private_condition = self.private_single_condition_encoder(
            self._public_private_private_features(context)
        )
        return self.relation_private_condition_encoder(
            th.cat([relation_condition, private_condition], dim=-1)
        )

    def _apply_action_edge_relation_private_single_head(self, hidden, relation_condition, context):
        generator_condition = self._action_edge_relation_private_condition(relation_condition, context)
        self.latest_condition = generator_condition.detach()
        return self._apply_generated_mlp_head(
            hidden, generator_condition, self.relation_private_single_head_hypernet
        )

    def _apply_relation_coarse_self_fine_head(self, hidden, relation_condition, context):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)

        move_feat, _, _, own_feat = self._split_rpg_obs(context["obs"])
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        self_condition = self.self_fine_condition_encoder(self_feat).reshape(batch_size * n_agents, -1)

        coarse_bottleneck_w = self.relation_coarse_bottleneck_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        coarse_bottleneck_b = self.relation_coarse_bottleneck_b(flat_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        coarse_out_w = self.relation_coarse_out_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        coarse_out_b = self.relation_coarse_out_b(flat_condition).view(
            batch_size * n_agents, 1, self.n_actions
        )

        fine_out_w = self.self_fine_out_w(self_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        fine_out_b = self.self_fine_out_b(self_condition).view(
            batch_size * n_agents, 1, self.n_actions
        )
        out_w = coarse_out_w + self.self_fine_delta_scale * fine_out_w
        out_b = coarse_out_b + self.self_fine_delta_scale * fine_out_b

        mid = F.elu(th.bmm(flat_hidden, coarse_bottleneck_w) + coarse_bottleneck_b)
        q = th.bmm(mid, out_w) + out_b
        generated_head = th.cat(
            [
                coarse_bottleneck_w.reshape(batch_size * n_agents, -1),
                coarse_bottleneck_b.reshape(batch_size * n_agents, -1),
                coarse_out_w.reshape(batch_size * n_agents, -1),
                coarse_out_b.reshape(batch_size * n_agents, -1),
                fine_out_w.reshape(batch_size * n_agents, -1),
                fine_out_b.reshape(batch_size * n_agents, -1),
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
        return q.view(batch_size, n_agents, self.n_actions)

    def _self_fine_condition(self, context, batch_size, n_agents):
        move_feat, _, _, own_feat = self._split_rpg_obs(context["obs"])
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        return self.self_fine_condition_encoder(self_feat).reshape(batch_size * n_agents, -1)

    def _apply_relation_coarse_fine_four_layer_head(self, hidden, relation_condition, context):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)
        self_condition = self._self_fine_condition(context, batch_size, n_agents)

        coarse_l1_w = self.relation_coarse_layer1_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        coarse_l1_b = self.relation_coarse_layer1_b(flat_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        coarse_l2_w = self.relation_coarse_layer2_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        coarse_l2_b = self.relation_coarse_layer2_b(flat_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        fine_l3_w = self.self_fine_layer3_w(self_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        fine_l3_b = self.self_fine_layer3_b(self_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        fine_l4_w = self.self_fine_layer4_w(self_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        fine_l4_b = self.self_fine_layer4_b(self_condition).view(
            batch_size * n_agents, 1, self.n_actions
        )

        current = F.elu(th.bmm(flat_hidden, coarse_l1_w) + coarse_l1_b)
        current = F.elu(th.bmm(current, coarse_l2_w) + coarse_l2_b)
        current = F.elu(th.bmm(current, fine_l3_w) + fine_l3_b)
        q = th.bmm(current, fine_l4_w) + fine_l4_b
        generated_head = th.cat(
            [
                coarse_l1_w.reshape(batch_size * n_agents, -1),
                coarse_l1_b.reshape(batch_size * n_agents, -1),
                coarse_l2_w.reshape(batch_size * n_agents, -1),
                coarse_l2_b.reshape(batch_size * n_agents, -1),
                fine_l3_w.reshape(batch_size * n_agents, -1),
                fine_l3_b.reshape(batch_size * n_agents, -1),
                fine_l4_w.reshape(batch_size * n_agents, -1),
                fine_l4_b.reshape(batch_size * n_agents, -1),
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
        return q.view(batch_size, n_agents, self.n_actions)

    def _apply_relation_coarse_q_fine_gate_head(self, hidden, relation_condition, context):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)
        self_condition = self._self_fine_condition(context, batch_size, n_agents)

        coarse_bottleneck_w = self.relation_coarse_bottleneck_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        coarse_bottleneck_b = self.relation_coarse_bottleneck_b(flat_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        coarse_out_w = self.relation_coarse_out_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        coarse_out_b = self.relation_coarse_out_b(flat_condition).view(
            batch_size * n_agents, 1, self.n_actions
        )
        coarse_mid = F.elu(th.bmm(flat_hidden, coarse_bottleneck_w) + coarse_bottleneck_b)
        coarse_q = th.bmm(coarse_mid, coarse_out_w) + coarse_out_b

        gate_bottleneck_w = self.self_fine_gate_bottleneck_w(self_condition).view(
            batch_size * n_agents, self.hidden_dim, self.hidden_dim
        )
        gate_bottleneck_b = self.self_fine_gate_bottleneck_b(self_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        gate_out_w = self.self_fine_gate_out_w(self_condition).view(
            batch_size * n_agents, self.hidden_dim, self.n_actions
        )
        gate_out_b = self.self_fine_gate_out_b(self_condition).view(
            batch_size * n_agents, 1, self.n_actions
        )
        gate_mid = F.elu(th.bmm(flat_hidden, gate_bottleneck_w) + gate_bottleneck_b)
        gate = th.sigmoid(th.bmm(gate_mid, gate_out_w) + gate_out_b)
        q = coarse_q + self.self_fine_gate_scale * th.log((2.0 * gate).clamp(min=1e-6))

        generated_head = th.cat(
            [
                coarse_bottleneck_w.reshape(batch_size * n_agents, -1),
                coarse_bottleneck_b.reshape(batch_size * n_agents, -1),
                coarse_out_w.reshape(batch_size * n_agents, -1),
                coarse_out_b.reshape(batch_size * n_agents, -1),
                gate_bottleneck_w.reshape(batch_size * n_agents, -1),
                gate_bottleneck_b.reshape(batch_size * n_agents, -1),
                gate_out_w.reshape(batch_size * n_agents, -1),
                gate_out_b.reshape(batch_size * n_agents, -1),
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
        return q.view(batch_size, n_agents, self.n_actions)

    def _apply_relation_prototype_single_head(self, hidden, relation_condition, test_mode):
        route_logits = self.route_logits_head(relation_condition)
        prototype_condition = self._route_from_logits(route_logits, test_mode=test_mode)
        self.latest_condition = prototype_condition.detach()
        return self._apply_generated_mlp_head(hidden, prototype_condition, self.prototype_head_hypernet)

    def _linear_generated_interaction(self, flat_interaction_input, flat_condition, batch_size, n_agents):
        interaction_out_w = self.rpg_interaction_out_w(flat_condition).view(
            batch_size * n_agents, self.rpg_interaction_input_dim, 1
        )
        interaction_out_b = self.rpg_interaction_out_b(flat_condition).view(batch_size * n_agents, 1, 1)
        capture_parameter_graph = (
            SEMANTIC_ROUTER_MODE_BY_MODEL.get(self.model_type)
            == "parameter_sensitivity"
            and self.capture_semantic_parameter_graph
        )
        generated_head = None
        if (
            self.capture_generated_interaction_head
            or capture_parameter_graph
            or self.model_type in PUBLIC_TRANSFORMER_SMOOTH_HEAD_VARIANTS
        ):
            generated_head = th.cat(
                [
                    interaction_out_w.reshape(batch_size * n_agents, -1),
                    interaction_out_b.reshape(batch_size * n_agents, -1),
                ],
                dim=-1,
            )
        if self.capture_generated_interaction_head and generated_head is not None:
            self.latest_generated_interaction_head = generated_head.detach().view(
                batch_size, n_agents, -1
            )
        if capture_parameter_graph:
            self.latest_generated_interaction_head_graph = generated_head.view(
                batch_size, n_agents, -1
            )
        q_attack = th.bmm(flat_interaction_input, interaction_out_w) + interaction_out_b
        return q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"]), generated_head

    def _add_aux_loss(self, loss):
        if loss is None:
            return
        self.latest_aux_loss = loss if self.latest_aux_loss is None else self.latest_aux_loss + loss

    def _q_residual_generated_interaction(self, flat_interaction_input, flat_condition, batch_size, n_agents):
        q_dynamic, generated_head = self._linear_generated_interaction(
            flat_interaction_input, flat_condition, batch_size, n_agents
        )
        q_base = self.rpg_interaction_base_scorer(flat_interaction_input).squeeze(-1)
        q_base = q_base.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"])
        gate = th.sigmoid(self.rpg_residual_interaction_gate(flat_condition)).view(batch_size, n_agents, 1)
        if th.is_grad_enabled():
            self.latest_aux_stats["residual_interaction_gate"] = gate.mean().detach()
        return q_base + gate * q_dynamic, generated_head

    def _param_residual_generated_interaction(self, flat_interaction_input, flat_condition, batch_size, n_agents):
        batch_agents = batch_size * n_agents
        delta_w = self.rpg_interaction_out_w(flat_condition).view(
            batch_agents, self.rpg_interaction_input_dim, 1
        )
        delta_b = self.rpg_interaction_out_b(flat_condition).view(batch_agents, 1, 1)
        gate = th.sigmoid(self.rpg_residual_interaction_gate(flat_condition)).view(batch_agents, 1, 1)

        base_w = self.rpg_interaction_base_scorer.weight.t().unsqueeze(0).expand_as(delta_w)
        base_b = self.rpg_interaction_base_scorer.bias.view(1, 1, 1)
        interaction_out_w = base_w + gate * delta_w
        interaction_out_b = base_b + gate * delta_b
        generated_head = th.cat(
            [
                interaction_out_w.reshape(batch_agents, -1),
                interaction_out_b.reshape(batch_agents, -1),
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
        q_attack = th.bmm(flat_interaction_input, interaction_out_w) + interaction_out_b
        residual_head = None
        if th.is_grad_enabled():
            self.latest_aux_stats["residual_interaction_gate"] = gate.mean().detach()
            residual_head = th.cat(
                [
                    (gate * delta_w).reshape(batch_agents, -1),
                    (gate * delta_b).reshape(batch_agents, -1),
                ],
                dim=-1,
            )
            self.latest_aux_stats["residual_interaction_param_norm"] = residual_head.norm(dim=-1).mean().detach()
        return q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"]), generated_head, residual_head

    def _linear_generated_interaction_selected(
        self,
        interaction_input,
        relation_condition,
        selected_mask,
        batch_size,
        n_agents,
    ):
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)
        condition_rep = flat_condition.unsqueeze(1).expand(
            -1, self.rpg_obs_layout["n_enemies"], -1
        ).reshape(-1, flat_condition.size(-1))
        flat_input = interaction_input.reshape(-1, self.rpg_interaction_input_dim)
        flat_mask = selected_mask.reshape(-1).bool()

        q_flat = flat_input.new_zeros(flat_input.size(0))
        if flat_mask.any():
            selected_input = flat_input[flat_mask].unsqueeze(1)
            selected_condition = condition_rep[flat_mask]
            interaction_out_w = self.rpg_interaction_out_w(selected_condition).view(
                selected_input.size(0), self.rpg_interaction_input_dim, 1
            )
            interaction_out_b = self.rpg_interaction_out_b(selected_condition).view(
                selected_input.size(0), 1, 1
            )
            selected_q = th.bmm(selected_input, interaction_out_w) + interaction_out_b
            q_flat[flat_mask] = selected_q.view(-1)

        self.latest_generated_interaction_head = None
        return q_flat.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"])

    def _head_smoothness_loss(self, flat_condition, generated_head, normalize_head=True):
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
        rel_dist = 1.0 - th.matmul(condition, condition.transpose(0, 1))
        rel_dist = rel_dist.masked_fill(th.eye(sample_size, device=rel_dist.device).bool(), float("inf"))
        knn = min(self.smooth_head_knn, sample_size - 1)
        neighbor_idx = rel_dist.topk(k=knn, largest=False, dim=-1).indices
        neighbor_head = head.index_select(0, neighbor_idx.reshape(-1)).view(sample_size, knn, -1)
        if normalize_head:
            head = F.normalize(head, p=2, dim=-1)
            neighbor_head = F.normalize(neighbor_head, p=2, dim=-1)
            head_sim = (head.unsqueeze(1) * neighbor_head).sum(dim=-1)
            return (1.0 - head_sim).mean()
        return (head.unsqueeze(1) - neighbor_head).pow(2).mean()

    def _private_enemy_tokens(self, context, enemy_mask):
        if context is None:
            raise ValueError("{} requires context for private enemy tokens.".format(self.model_type))
        _, enemy_feat, _, _ = self._split_rpg_obs(context["obs"])
        private_feat = enemy_feat[:, :, :, :4]
        return self.rpg_private_enemy_token_encoder(private_feat) * enemy_mask.unsqueeze(-1).float()

    def _delta_enemy_tokens(self, context, enemy_mask):
        if context is None:
            raise ValueError("{} requires context for delta enemy tokens.".format(self.model_type))
        _, enemy_feat, _, _ = self._split_rpg_obs(context["obs"])
        prev_obs = context.get("prev_obs")
        if prev_obs is None:
            prev_enemy_feat = th.zeros_like(enemy_feat)
        else:
            _, prev_enemy_feat, _, _ = self._split_rpg_obs(prev_obs)
        prev_enemy_mask = prev_enemy_feat.abs().sum(dim=-1) > 0
        valid = enemy_mask.bool() & prev_enemy_mask.bool()
        delta_feat = (enemy_feat - prev_enemy_feat) * valid.unsqueeze(-1).float()
        return self.rpg_delta_enemy_token_encoder(delta_feat) * valid.unsqueeze(-1).float()

    def _interaction_enemy_features(self, enemy_tokens, enemy_mask, context, hidden=None):
        if self.model_type == "rpg_no_enemy_token_interaction_hypercond":
            return th.zeros_like(enemy_tokens)
        if self.model_type == "rpg_private_enemy_token_interaction_hypercond":
            return self._private_enemy_tokens(context, enemy_mask)
        if self.model_type == "rpg_delta_enemy_token_interaction_hypercond":
            delta_tokens = self._delta_enemy_tokens(context, enemy_mask)
            return th.cat([enemy_tokens, delta_tokens], dim=-1)
        if self.model_type in PUBLIC_TRANSFORMER_PAIR_INTERACTION_VARIANTS:
            if hidden is None or self.rpg_hidden_pair_encoder is None:
                raise ValueError("{} requires policy hidden for pair interaction features.".format(self.model_type))
            hidden_pair = self.rpg_hidden_pair_encoder(hidden)
            hidden_pair = hidden_pair.unsqueeze(2).expand_as(enemy_tokens)
            pair_tokens = hidden_pair * enemy_tokens
            return th.cat([enemy_tokens, pair_tokens], dim=-1)
        if self.model_type in PUBLIC_TRANSFORMER_PAIR_CONCAT_INTERACTION_VARIANTS:
            if hidden is None or self.rpg_hidden_pair_encoder is None or self.rpg_pair_concat_encoder is None:
                raise ValueError("{} requires policy hidden for concat pair interaction features.".format(self.model_type))
            hidden_pair = self.rpg_hidden_pair_encoder(hidden)
            hidden_pair = hidden_pair.unsqueeze(2).expand_as(enemy_tokens)
            pair_input = th.cat([hidden_pair, enemy_tokens], dim=-1)
            pair_tokens = self.rpg_pair_concat_encoder(pair_input)
            return pair_tokens * enemy_mask.unsqueeze(-1).float()
        if self.model_type in PUBLIC_TRANSFORMER_PRIVATE_HEAD_INPUT_VARIANTS:
            private_tokens = self._private_enemy_tokens(context, enemy_mask)
            return th.cat([enemy_tokens, private_tokens], dim=-1)
        return enemy_tokens

    def _target_selection_mask(self, source, relation_condition, enemy_tokens, enemy_mask):
        batch_size, n_agents, n_enemies, _ = enemy_tokens.shape
        source_rep = source.unsqueeze(2).expand(-1, -1, n_enemies, -1)
        cond_rep = relation_condition.unsqueeze(2).expand(-1, -1, n_enemies, -1)
        selector_input = th.cat([source_rep, cond_rep, enemy_tokens], dim=-1)
        scores = self.rpg_target_selector(selector_input).squeeze(-1)
        valid_mask = enemy_mask.bool()
        masked_scores = scores.masked_fill(~valid_mask, _neg_inf_like(scores))
        valid_any = valid_mask.any(dim=-1, keepdim=True)

        if self.model_type in {
            "rpg_post_topk_enemy_interaction_hypercond",
            "rpg_public_private_bias_transformer_topk_hypercond",
            "rpg_global_public_private_bias_past_delta_token_transformer_topk_hypercond",
        } | PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS:
            k = max(1, min(self.target_topk, n_enemies))
            _, indices = th.topk(masked_scores, k=k, dim=-1)
            gathered_valid = th.gather(valid_mask, dim=-1, index=indices)
            selected = scores.new_zeros(scores.shape)
            selected.scatter_(dim=-1, index=indices, src=gathered_valid.float())
            selected = selected.bool()
        else:
            selected = (th.sigmoid(scores) >= self.target_threshold) & valid_mask
            selected_any = selected.any(dim=-1, keepdim=True)
            fallback_idx = masked_scores.argmax(dim=-1, keepdim=True)
            fallback = th.zeros_like(selected)
            fallback.scatter_(dim=-1, index=fallback_idx, src=valid_any)
            selected = th.where(selected_any, selected, fallback)

        selected = selected & valid_mask
        valid_count = valid_mask.float().sum().clamp(min=1.0)
        self.latest_aux_stats["target_select_frac"] = (selected.float().sum() / valid_count).detach()
        return selected

    def _rpg_self_token_from_context(self, context):
        if context is None:
            raise ValueError("{} requires context for self-token decision inputs.".format(self.model_type))
        move_feat, _, _, own_feat = self._split_rpg_obs(context["obs"])
        self_feat = th.cat([move_feat, own_feat], dim=-1)
        if not hasattr(self.rpg_relation_capturer, "self_encoder"):
            raise ValueError("{} requires an RPG-style capturer with self_encoder.".format(self.model_type))
        return self.rpg_relation_capturer.self_encoder(self_feat)

    def _apply_rpg_token_decision_head(
        self,
        relation_condition,
        self_token,
        enemy_tokens,
        enemy_mask,
        relation_hidden,
        policy_hidden=None,
        context=None,
    ):
        batch_size, n_agents, _ = relation_condition.shape
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)

        if self.model_type in RPG_POLICY_RELATION_FUSION_HEAD_VARIANTS:
            if relation_hidden is None or policy_hidden is None:
                raise ValueError("{} requires policy_hidden and relation_hidden.".format(self.model_type))
            fusion_input = th.cat([policy_hidden, relation_hidden], dim=-1)
            fused_source = self.policy_relation_decision_fuser(fusion_input)
            ego_source = fused_source
            interaction_anchor = fused_source
        elif self.model_type in RELATION_TOKEN_DECISION_HEAD_VARIANTS:
            if relation_hidden is None:
                raise ValueError("{} requires relation_hidden as decision input.".format(self.model_type))
            ego_source = relation_hidden
            interaction_anchor = relation_hidden
        else:
            ego_source = self_token
            interaction_anchor = self_token

        flat_ego_source = ego_source.reshape(batch_size * n_agents, 1, self.rpg_relation_dim)
        ego_bottleneck_w = self.token_ego_bottleneck_w(flat_condition).view(
            batch_size * n_agents, self.rpg_relation_dim, self.hidden_dim
        )
        ego_bottleneck_b = self.token_ego_bottleneck_b(flat_condition).view(
            batch_size * n_agents, 1, self.hidden_dim
        )
        ego_out_w = self.token_ego_out_w(flat_condition).view(
            batch_size * n_agents, self.hidden_dim, self.rpg_n_ego_actions
        )
        ego_out_b = self.token_ego_out_b(flat_condition).view(
            batch_size * n_agents, 1, self.rpg_n_ego_actions
        )
        ego_mid = F.elu(th.bmm(flat_ego_source, ego_bottleneck_w) + ego_bottleneck_b)
        q_ego = th.bmm(ego_mid, ego_out_w) + ego_out_b
        q_ego = q_ego.view(batch_size, n_agents, self.rpg_n_ego_actions)

        if self.model_type == "rpg_entity_token_decision_head_hypercond":
            interaction_input = enemy_tokens
        elif self.model_type in PUBLIC_TRANSFORMER_RELATION_PAIR_TOKEN_HEAD_VARIANTS:
            anchor_rep = interaction_anchor.unsqueeze(2).expand(
                -1, -1, self.rpg_obs_layout["n_enemies"], -1
            )
            pair_token = anchor_rep * enemy_tokens
            interaction_input = th.cat([anchor_rep, enemy_tokens, pair_token], dim=-1)
        elif self.model_type in PUBLIC_TRANSFORMER_RELATION_PRIVATE_TOKEN_HEAD_VARIANTS:
            anchor_rep = interaction_anchor.unsqueeze(2).expand(
                -1, -1, self.rpg_obs_layout["n_enemies"], -1
            )
            private_tokens = self._private_enemy_tokens(context, enemy_mask)
            interaction_input = th.cat([anchor_rep, enemy_tokens, private_tokens], dim=-1)
        elif self.model_type in PUBLIC_TRANSFORMER_RELATION_DELTA_TOKEN_HEAD_VARIANTS:
            anchor_rep = interaction_anchor.unsqueeze(2).expand(
                -1, -1, self.rpg_obs_layout["n_enemies"], -1
            )
            delta_tokens = self._delta_enemy_tokens(context, enemy_mask)
            interaction_input = th.cat([anchor_rep, enemy_tokens, delta_tokens], dim=-1)
        else:
            anchor_rep = interaction_anchor.unsqueeze(2).expand(
                -1, -1, self.rpg_obs_layout["n_enemies"], -1
            )
            interaction_input = th.cat([anchor_rep, enemy_tokens], dim=-1)

        flat_interaction_input = interaction_input.reshape(
            batch_size * n_agents,
            self.rpg_obs_layout["n_enemies"],
            self.token_interaction_input_dim,
        )
        interaction_out_w = self.token_interaction_out_w(flat_condition).view(
            batch_size * n_agents, self.token_interaction_input_dim, 1
        )
        interaction_out_b = self.token_interaction_out_b(flat_condition).view(
            batch_size * n_agents, 1, 1
        )
        q_attack = th.bmm(flat_interaction_input, interaction_out_w) + interaction_out_b
        q_attack = q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"])

        generated_head = th.cat(
            [
                ego_bottleneck_w.reshape(batch_size * n_agents, -1),
                ego_bottleneck_b.reshape(batch_size * n_agents, -1),
                ego_out_w.reshape(batch_size * n_agents, -1),
                ego_out_b.reshape(batch_size * n_agents, -1),
                interaction_out_w.reshape(batch_size * n_agents, -1),
                interaction_out_b.reshape(batch_size * n_agents, -1),
            ],
            dim=-1,
        )
        self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
        q_attack = q_attack.masked_fill(~enemy_mask.bool(), 0.0)
        return th.cat([q_ego, q_attack], dim=-1)

    def _apply_rpg_structured_maker(self, hidden, relation_condition, enemy_tokens, enemy_mask, context=None):
        batch_size, n_agents, _ = hidden.shape
        flat_hidden = hidden.reshape(batch_size * n_agents, 1, self.hidden_dim)
        flat_condition = relation_condition.reshape(batch_size * n_agents, -1)
        compute_training_aux = th.is_grad_enabled()
        generated_ego_head = None

        if self.model_type in {"rpg_fixed_structured_maker", "rpg_fixed_linear_structured_maker"}:
            q_ego = self.rpg_ego_maker(th.cat([hidden, relation_condition], dim=-1))
        else:
            ego_input = hidden
            ego_input_dim = self.rpg_ego_input_dim or self.hidden_dim
            if self.model_type in PUBLIC_TRANSFORMER_SLOT_TOKEN_HEAD_VARIANTS:
                self_token = getattr(self.rpg_relation_capturer, "latest_encoded_self_token", None)
                if self_token is None:
                    raise ValueError(
                        f"{self.model_type} requires encoded Transformer self slots before ego head computation."
                    )
                ego_input = th.cat([hidden, self_token], dim=-1)
            flat_ego_input = ego_input.reshape(batch_size * n_agents, 1, ego_input_dim)
            ego_bottleneck_w = self.rpg_ego_bottleneck_w(flat_condition).view(
                batch_size * n_agents, ego_input_dim, self.hidden_dim
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
            if (
                SEMANTIC_ROUTER_MODE_BY_MODEL.get(self.model_type)
                == "parameter_sensitivity"
                and self.capture_semantic_parameter_graph
            ):
                generated_ego_head = th.cat(
                    [
                        ego_bottleneck_w.reshape(batch_size * n_agents, -1),
                        ego_bottleneck_b.reshape(batch_size * n_agents, -1),
                        ego_out_w.reshape(batch_size * n_agents, -1),
                        ego_out_b.reshape(batch_size * n_agents, -1),
                    ],
                    dim=-1,
                )

            if self.model_type in PUBLIC_TRANSFORMER_PARAM_RESIDUAL_HEAD_VARIANTS:
                gate = th.sigmoid(self.rpg_residual_ego_gate(flat_condition)).view(
                    batch_size * n_agents, 1, 1
                )
                base_w1 = self.rpg_ego_base_maker[0].weight.t().unsqueeze(0).expand_as(ego_bottleneck_w)
                base_b1 = self.rpg_ego_base_maker[0].bias.view(1, 1, self.hidden_dim)
                base_w2 = self.rpg_ego_base_maker[2].weight.t().unsqueeze(0).expand_as(ego_out_w)
                base_b2 = self.rpg_ego_base_maker[2].bias.view(1, 1, self.rpg_n_ego_actions)
                if compute_training_aux:
                    ego_residual_parts = [
                        gate * ego_bottleneck_w,
                        gate * ego_bottleneck_b,
                        gate * ego_out_w,
                        gate * ego_out_b,
                    ]
                ego_bottleneck_w = base_w1 + gate * ego_bottleneck_w
                ego_bottleneck_b = base_b1 + gate * ego_bottleneck_b
                ego_out_w = base_w2 + gate * ego_out_w
                ego_out_b = base_b2 + gate * ego_out_b
                if compute_training_aux:
                    self.latest_aux_stats["residual_ego_gate"] = gate.mean().detach()
                    ego_residual_norm = th.stack([part.pow(2).mean() for part in ego_residual_parts]).mean()
                    self.latest_aux_stats["residual_ego_param_norm"] = ego_residual_norm.sqrt().detach()
                    if self.model_type in PUBLIC_TRANSFORMER_RESIDUAL_L2_HEAD_VARIANTS:
                        self._add_aux_loss(self.stable_residual_l2_coef * ego_residual_norm)

            ego_mid = F.elu(th.bmm(flat_ego_input, ego_bottleneck_w) + ego_bottleneck_b)
            q_ego = th.bmm(ego_mid, ego_out_w) + ego_out_b
            if self.model_type in PUBLIC_TRANSFORMER_Q_RESIDUAL_HEAD_VARIANTS:
                gate = th.sigmoid(self.rpg_residual_ego_gate(flat_condition)).view(batch_size, n_agents, 1)
                q_ego_base = self.rpg_ego_base_maker(ego_input)
                q_ego = q_ego_base + gate * q_ego.view(batch_size, n_agents, self.rpg_n_ego_actions)
                if compute_training_aux:
                    self.latest_aux_stats["residual_ego_gate"] = gate.mean().detach()
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
        elif self.model_type in {
            "local_linear_interaction_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_RELATION_VARIANTS,
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            "rpg_action_edge_public_pred_relation_private_decision_maker",
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_delta_relation_hypercond",
        }:
            interaction_enemy_features = self._interaction_enemy_features(enemy_tokens, enemy_mask, context, hidden=hidden)
            interaction_input = th.cat([hidden_rep, interaction_enemy_features], dim=-1)
            if self.model_type in PUBLIC_TRANSFORMER_TARGET_SELECTION_VARIANTS:
                target_mask = self._target_selection_mask(hidden, relation_condition, enemy_tokens, enemy_mask)
                q_attack = self._linear_generated_interaction_selected(
                    interaction_input, relation_condition, target_mask, batch_size, n_agents
                )
            else:
                flat_interaction_input = interaction_input.reshape(
                    batch_size * n_agents, self.rpg_obs_layout["n_enemies"], self.rpg_interaction_input_dim
                )
                if self.model_type in PUBLIC_TRANSFORMER_Q_RESIDUAL_HEAD_VARIANTS:
                    q_attack, generated_head = self._q_residual_generated_interaction(
                        flat_interaction_input, flat_condition, batch_size, n_agents
                    )
                elif self.model_type in PUBLIC_TRANSFORMER_PARAM_RESIDUAL_HEAD_VARIANTS:
                    q_attack, generated_head, residual_head = self._param_residual_generated_interaction(
                        flat_interaction_input, flat_condition, batch_size, n_agents
                    )
                    if compute_training_aux and self.model_type in PUBLIC_TRANSFORMER_RESIDUAL_L2_HEAD_VARIANTS:
                        self._add_aux_loss(self.stable_residual_l2_coef * residual_head.pow(2).mean())
                else:
                    q_attack, generated_head = self._linear_generated_interaction(
                        flat_interaction_input, flat_condition, batch_size, n_agents
                    )
                if compute_training_aux and self.model_type in PUBLIC_TRANSFORMER_SMOOTH_HEAD_VARIANTS:
                    smooth_target = (
                        residual_head
                        if self.model_type in PUBLIC_TRANSFORMER_RESIDUAL_SMOOTH_HEAD_VARIANTS
                        else generated_head
                    )
                    smooth_loss = self._head_smoothness_loss(
                        flat_condition,
                        smooth_target,
                        normalize_head=self.model_type not in PUBLIC_TRANSFORMER_RESIDUAL_SMOOTH_HEAD_VARIANTS,
                    )
                    self.latest_aux_stats["head_smooth_loss_raw"] = smooth_loss.detach()
                    self._add_aux_loss(self.smooth_head_loss_coef * smooth_loss)
            if self.model_type in RPG_POST_TARGET_SELECTION_VARIANTS:
                target_mask = self._target_selection_mask(hidden, relation_condition, enemy_tokens, enemy_mask)
                q_attack = q_attack.masked_fill(~target_mask, 0.0)
        elif self.model_type == "rpg_flat_interaction_hypercond":
            interaction_out_w = self.rpg_interaction_out_w(flat_condition).view(
                batch_size * n_agents, self.hidden_dim, self.rpg_obs_layout["n_enemies"]
            )
            interaction_out_b = self.rpg_interaction_out_b(flat_condition).view(
                batch_size * n_agents, 1, self.rpg_obs_layout["n_enemies"]
            )
            generated_head = th.cat(
                [
                    interaction_out_w.reshape(batch_size * n_agents, -1),
                    interaction_out_b.reshape(batch_size * n_agents, -1),
                ],
                dim=-1,
            )
            self.latest_generated_interaction_head = generated_head.detach().view(batch_size, n_agents, -1)
            q_attack = th.bmm(flat_hidden, interaction_out_w) + interaction_out_b
            q_attack = q_attack.view(batch_size, n_agents, self.rpg_obs_layout["n_enemies"])
        elif self.model_type == "rpg_private_interaction_input_hypercond":
            if context is None:
                raise ValueError("rpg_private_interaction_input_hypercond requires context for private MLP inputs.")
            move_feat, raw_enemy_feat, _, _ = self._split_rpg_obs(context["obs"])
            move_rep = move_feat.unsqueeze(2).expand(-1, -1, self.rpg_obs_layout["n_enemies"], -1)
            target_private_feat = raw_enemy_feat[:, :, :, :4]
            private_input = th.cat([move_rep, target_private_feat], dim=-1)
            interaction_input = self.rpg_interaction_encoder(private_input)
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
            if compute_training_aux:
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
        if (
            SEMANTIC_ROUTER_MODE_BY_MODEL.get(self.model_type)
            == "parameter_sensitivity"
            and self.capture_semantic_parameter_graph
            and generated_ego_head is not None
            and self.latest_generated_interaction_head_graph is not None
        ):
            interaction_graph = self.latest_generated_interaction_head_graph.reshape(
                batch_size * n_agents, -1
            )
            self.latest_generated_interaction_head_graph = th.cat(
                [generated_ego_head, interaction_graph], dim=-1
            ).view(batch_size, n_agents, -1)
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
        self.latest_aux_stats = {}
        self.latest_teacher_q = None
        self.latest_generated_interaction_head = None
        self.latest_generated_interaction_head_graph = None

        if hidden_state is None:
            hidden_state = self.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)
        if self.model_type in {
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_linear_interaction_hypercond",
            "rpg_flat_interaction_hypercond",
            "rpg_public_relation_hypercond",
            "rpg_private_interaction_input_hypercond",
            "rpg_global_filled_obs_hypercond",
            "rpg_relation_distill_hypercond",
            "rpg_public_delta_aux_hypercond",
            *PUBLIC_TRANSFORMER_CAPTURER_VARIANTS,
            "rpg_residual_interaction_hypercond",
            "rpg_film_interaction_hypercond",
            "rpg_moe_interaction_head",
            "rpg_smooth_linear_interaction_hypercond",
            "rpg_semantic_selfattn_relation_hypercond",
            "rpg_entity_selfattn_relation_hypercond",
            "rpg_topk_entity_relation_hypercond",
            *RPG_TARGETWISE_ABLATION_VARIANTS,
            *TOKEN_DECISION_HEAD_VARIANTS,
            "rpg_action_edge_graph_hypercond",
            "rpg_action_edge_rgcn_hypercond",
            "rpg_action_edge_egcn_hypercond",
            "rpg_action_edge_egcn_plus_public_pred_hypercond",
            "rpg_action_edge_oracle_graph_hypercond",
            "rpg_action_edge_oracle_no_self_hypercond",
            "rpg_action_edge_prev_oracle_graph_hypercond",
            "rpg_action_edge_public_pred_hypercond",
            *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
            *ACTION_EDGE_REL_PRIVATE_VARIANTS,
            "rpg_action_edge_public_memory_hypercond",
            "rpg_action_edge_global_public_pred_hypercond",
            "rpg_action_edge_target_context_hypercond",
            "rpg_action_edge_coarse_private_fine_gate_hypercond",
            "rpg_relation_coarse_self_fine_head",
            "rpg_relation_coarse_fine_four_layer_head",
            "rpg_relation_coarse_q_fine_gate_head",
            "rpg_relation_prototype_single_head",
            "rpg_fixed_structured_maker",
            "rpg_fixed_linear_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
            *GRF_PUBLIC_TRANSFORMER_VARIANTS,
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
            if self.model_type in {"local_structured_hypercond", "local_linear_interaction_hypercond"}:
                condition, enemy_tokens, enemy_mask = self._build_local_structured_condition(hidden, context)
                self.latest_condition = condition.detach()
                q = self._apply_rpg_structured_maker(hidden, condition, enemy_tokens, enemy_mask)
                next_hidden = hidden
            elif self.model_type in GRF_PUBLIC_TRANSFORMER_VARIANTS:
                relation_condition, next_relation_hidden = self.rpg_relation_capturer(
                    context["obs"], relation_hidden_state
                )
                self.latest_condition = relation_condition.detach()
                if self.model_type in GRF_DECISION_MAKER_VARIANTS:
                    q = self._apply_grf_decision_maker_head(hidden, relation_condition)
                elif self.model_type in GRF_LINEAR_HEAD_VARIANTS:
                    q = self._apply_grf_linear_head(hidden, relation_condition)
                else:
                    q = self._apply_dynamic_head(hidden, relation_condition)
                next_hidden = th.cat([hidden, next_relation_hidden], dim=-1)
            elif self.model_type in {
                "rpg_action_edge_graph_hypercond",
                "rpg_action_edge_rgcn_hypercond",
                "rpg_action_edge_egcn_hypercond",
                "rpg_action_edge_egcn_plus_public_pred_hypercond",
                "rpg_action_edge_oracle_graph_hypercond",
                "rpg_action_edge_oracle_no_self_hypercond",
                "rpg_action_edge_prev_oracle_graph_hypercond",
                "rpg_action_edge_public_pred_hypercond",
                *ACTION_EDGE_PUBLIC_PRED_HEAD_VARIANTS,
                *ACTION_EDGE_REL_PRIVATE_VARIANTS,
                "rpg_action_edge_public_memory_hypercond",
                "rpg_action_edge_global_public_pred_hypercond",
                "rpg_action_edge_target_context_hypercond",
                "rpg_action_edge_coarse_private_fine_gate_hypercond",
            }:
                relation_condition, next_relation_hidden, enemy_tokens, enemy_mask, action_loss = (
                    self._build_action_edge_graph_condition(context, relation_hidden_state, test_mode=test_mode)
                )
                self.latest_condition = relation_condition.detach()
                self.latest_aux_stats = getattr(self.rpg_relation_capturer, "latest_aux_stats", {})
                if th.is_grad_enabled() and not test_mode:
                    self.latest_aux_loss = self.action_pred_loss_coef * action_loss
                if self.model_type in ACTION_EDGE_REL_PRIVATE_SINGLE_HEAD_VARIANTS:
                    q = self._apply_action_edge_relation_private_single_head(hidden, relation_condition, context)
                elif self.model_type == "rpg_action_edge_public_pred_relation_private_decision_maker":
                    relation_condition = self._action_edge_relation_private_condition(relation_condition, context)
                    self.latest_condition = relation_condition.detach()
                    q = self._apply_rpg_structured_maker(hidden, relation_condition, enemy_tokens, enemy_mask)
                elif self.model_type in ACTION_EDGE_PUBLIC_PRED_SINGLE_HEAD_VARIANTS:
                    q = self._apply_action_edge_public_pred_single_head(hidden, relation_condition, context)
                elif self.model_type == "rpg_action_edge_public_pred_coarse_fine_four_layer_head":
                    q = self._apply_relation_coarse_fine_four_layer_head(hidden, relation_condition, context)
                elif self.model_type in {
                    "rpg_action_edge_coarse_private_fine_gate_hypercond",
                    "rpg_action_edge_public_pred_coarse_q_fine_gate_head",
                }:
                    q = self._apply_relation_coarse_q_fine_gate_head(hidden, relation_condition, context)
                else:
                    if self.model_type in {
                        "rpg_action_edge_target_context_hypercond",
                        "rpg_action_edge_egcn_plus_public_pred_hypercond",
                    }:
                        graph_enemy_tokens = getattr(self.rpg_relation_capturer, "latest_enemy_graph_tokens", None)
                        if graph_enemy_tokens is not None:
                            enemy_tokens = enemy_tokens + graph_enemy_tokens
                    q = self._apply_rpg_structured_maker(hidden, relation_condition, enemy_tokens, enemy_mask)
                next_hidden = th.cat([hidden, next_relation_hidden], dim=-1)
            elif self.model_type in {
                "rpg_public_hyper_private_input_single_head",
                "rpg_private_hyper_public_input_single_head",
            }:
                q = self._apply_public_private_single_head(hidden, context)
                next_hidden = hidden
            elif self.model_type == "rpg_delta_relation_hypercond":
                relation_condition, enemy_tokens, enemy_mask = self._build_rpg_delta_condition(context)
                self.latest_condition = relation_condition.detach()
                q = self._apply_rpg_structured_maker(hidden, relation_condition, enemy_tokens, enemy_mask)
                next_hidden = hidden
            elif self.model_type in {
                "rpg_relation_hypercond",
                "rpg_relation_route",
                "rpg_structured_hypercond",
                "rpg_full_structured_hypercond",
                "rpg_readout_structured_hypercond",
                "rpg_linear_interaction_hypercond",
                "rpg_flat_interaction_hypercond",
                "rpg_public_relation_hypercond",
                "rpg_private_interaction_input_hypercond",
                "rpg_global_filled_obs_hypercond",
                "rpg_relation_distill_hypercond",
                "rpg_public_delta_aux_hypercond",
                *PUBLIC_TRANSFORMER_CAPTURER_VARIANTS,
                "rpg_residual_interaction_hypercond",
                "rpg_film_interaction_hypercond",
                "rpg_moe_interaction_head",
                "rpg_smooth_linear_interaction_hypercond",
                "rpg_semantic_selfattn_relation_hypercond",
                "rpg_entity_selfattn_relation_hypercond",
                "rpg_topk_entity_relation_hypercond",
                *RPG_TARGETWISE_ABLATION_VARIANTS,
                *TOKEN_DECISION_HEAD_VARIANTS,
                "rpg_relation_coarse_self_fine_head",
                "rpg_relation_coarse_fine_four_layer_head",
                "rpg_relation_coarse_q_fine_gate_head",
                "rpg_relation_prototype_single_head",
                "rpg_fixed_structured_maker",
                "rpg_fixed_linear_structured_maker",
                "two_graph_gat_hypercond",
                "hetero_gat_hypercond",
            }:
                relation_condition, next_relation_hidden, enemy_tokens, enemy_mask = self._build_rpg_condition(
                    context, relation_hidden_state, test_mode=test_mode
                )
                if self.model_type in TOKEN_DECISION_HEAD_VARIANTS:
                    condition = relation_condition
                    self.latest_condition = condition.detach()
                    self_token = (
                        None
                        if self.model_type in RELATION_TOKEN_DECISION_HEAD_VARIANTS
                        or self.model_type in RPG_POLICY_RELATION_FUSION_HEAD_VARIANTS
                        else self._rpg_self_token_from_context(context)
                    )
                    q = self._apply_rpg_token_decision_head(
                        relation_condition,
                        self_token,
                        enemy_tokens,
                        enemy_mask,
                        next_relation_hidden,
                        policy_hidden=hidden,
                        context=context,
                    )
                    if self.model_type in PUBLIC_TRANSFORMER_RELATION_TOKEN_TOPK_VARIANTS:
                        target_mask = self._target_selection_mask(
                            next_relation_hidden, relation_condition, enemy_tokens, enemy_mask
                        )
                        q_ego = q[:, :, : self.rpg_n_ego_actions]
                        q_attack = q[:, :, self.rpg_n_ego_actions :].masked_fill(~target_mask, 0.0)
                        q = th.cat([q_ego, q_attack], dim=-1)
                elif self.model_type in {
                    "rpg_relation_hypercond",
                    "two_graph_gat_hypercond",
                    "hetero_gat_hypercond",
                    *PUBLIC_TRANSFORMER_SINGLE_HEAD_VARIANTS,
                }:
                    condition = relation_condition
                    self.latest_condition = condition.detach()
                    q = self._apply_dynamic_head(hidden, condition)
                    if (
                        self.model_type in PUBLIC_TRANSFORMER_FUTURE_DELTA_ALL_VARIANTS
                        and th.is_grad_enabled()
                        and not test_mode
                    ):
                        aux_loss = getattr(self.rpg_relation_capturer, "latest_aux_loss", None)
                        if aux_loss is not None:
                            self.latest_aux_loss = self.public_transformer_delta_loss_coef * aux_loss
                            self.latest_aux_stats.update(
                                getattr(self.rpg_relation_capturer, "latest_aux_stats", {})
                            )
                elif self.model_type == "rpg_relation_route":
                    route_logits = self.route_logits_head(relation_condition)
                    condition = self._route_from_logits(route_logits, test_mode=test_mode)
                    self.latest_condition = condition.detach()
                    q = self._apply_dynamic_head(hidden, condition)
                else:
                    condition = relation_condition
                    self.latest_condition = condition.detach()
                    if self.model_type == "rpg_relation_coarse_self_fine_head":
                        q = self._apply_relation_coarse_self_fine_head(hidden, relation_condition, context)
                    elif self.model_type == "rpg_relation_coarse_fine_four_layer_head":
                        q = self._apply_relation_coarse_fine_four_layer_head(hidden, relation_condition, context)
                    elif self.model_type == "rpg_relation_coarse_q_fine_gate_head":
                        q = self._apply_relation_coarse_q_fine_gate_head(hidden, relation_condition, context)
                    elif self.model_type == "rpg_relation_prototype_single_head":
                        q = self._apply_relation_prototype_single_head(hidden, relation_condition, test_mode)
                    else:
                        q = self._apply_rpg_structured_maker(
                            hidden, relation_condition, enemy_tokens, enemy_mask, context=context
                        )
                    if (
                        self.model_type in PUBLIC_TRANSFORMER_FUTURE_DELTA_VARIANTS
                        and th.is_grad_enabled()
                        and not test_mode
                    ):
                        aux_loss = getattr(self.rpg_relation_capturer, "latest_aux_loss", None)
                        if aux_loss is not None:
                            self.latest_aux_loss = self.public_transformer_delta_loss_coef * aux_loss
                            self.latest_aux_stats.update(
                                getattr(self.rpg_relation_capturer, "latest_aux_stats", {})
                            )
                    if (
                        self.model_type == "rpg_public_delta_aux_hypercond"
                        and th.is_grad_enabled()
                        and not test_mode
                        and context.get("next_obs_mask") is not None
                        and context["next_obs_mask"].bool().any()
                    ):
                        public_delta_loss, public_delta_stats = self._public_delta_aux_loss(
                            next_relation_hidden, context
                        )
                        self.latest_aux_loss = self.public_delta_loss_coef * public_delta_loss
                        self.latest_aux_stats.update(public_delta_stats)
                    if (
                        self.model_type == "rpg_relation_distill_hypercond"
                        and th.is_grad_enabled()
                        and not test_mode
                    ):
                        teacher_condition = self._build_relation_teacher_condition(context)
                        if teacher_condition is not None:
                            self.latest_teacher_q = self._apply_rpg_structured_maker(
                                hidden, teacher_condition, enemy_tokens, enemy_mask, context=context
                            )
                            self.latest_aux_loss = self.relation_distill_coef * self._relation_distill_loss(
                                relation_condition, teacher_condition
                            )
                next_hidden = th.cat([hidden, next_relation_hidden], dim=-1)
            else:
                condition = self._build_condition(hidden, context, test_mode=test_mode)
                if self.model_type == "hypermarl_fullnet":
                    q = self._apply_full_hypermarl_head(hidden, condition)
                else:
                    q = self._apply_dynamic_head(hidden, condition)
                next_hidden = hidden

        return q, next_hidden
