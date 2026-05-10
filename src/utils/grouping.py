import ast


def _parse_manual_group(group_value):
    if group_value is None:
        return None
    if isinstance(group_value, str):
        return ast.literal_eval(group_value)
    return group_value


def _validate_group(group, n_agents):
    if group is None:
        raise ValueError("Resolved group is None.")

    flat = [agent for group_i in group for agent in group_i]
    if sorted(flat) != list(range(n_agents)):
        raise ValueError(
            "Invalid group {} for {} agents. Groups must partition agent ids [0, {}].".format(
                group, n_agents, n_agents - 1
            )
        )
    return [list(group_i) for group_i in group]


def resolve_group_config(args):
    n_agents = args.n_agents
    env_args = getattr(args, "env_args", {})
    if isinstance(env_args, dict):
        map_name = env_args.get("map_name", None)
    else:
        map_name = getattr(env_args, "map_name", None)
    group_mode = getattr(args, "group_mode", "dynamic")
    manual_group = _parse_manual_group(getattr(args, "group", None))

    if group_mode == "dynamic":
        if manual_group is None:
            return [list(range(n_agents))]
        return _validate_group(manual_group, n_agents)

    if group_mode == "one_group":
        return [list(range(n_agents))]

    if group_mode == "singletons":
        return [[i] for i in range(n_agents)]

    if group_mode == "manual":
        if manual_group is None:
            raise ValueError("`group_mode=manual` requires `group` to be set.")
        return _validate_group(manual_group, n_agents)

    if group_mode == "same_type":
        if map_name in ["MMM", "MMM2"]:
            return _validate_group([[0, 1], [2, 3, 4, 5, 6, 7, 8], [9]], n_agents)
        if map_name == "3s5z_vs_3s6z":
            return _validate_group([[0, 1, 2], [3, 4, 5, 6, 7]], n_agents)
        raise ValueError(
            "`group_mode=same_type` is only implemented for MMM/MMM2 and 3s5z_vs_3s6z, got map_name={}.".format(
                map_name
            )
        )

    raise ValueError(
        "Unknown `group_mode`: {}. Expected one of dynamic, one_group, singletons, same_type, manual.".format(
            group_mode
        )
    )


def uses_dynamic_grouping(args):
    if getattr(args, "group_mode", "dynamic") != "dynamic":
        return False

    group_head_mode = getattr(args, "group_head_mode", "latent")
    if isinstance(group_head_mode, str):
        group_head_mode = group_head_mode.replace("-", "_")

    no_group_modes = {
        "graph_input_fusion_node_embed_struct_only",
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
    }
    return group_head_mode not in no_group_modes
