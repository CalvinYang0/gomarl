import math

import torch as th


def pseudo_attention_graph(node_features):
    # node_features: [batch, time, n_agents, feat_dim] or [batch, n_agents, feat_dim]
    if node_features.dim() == 3:
        node_features = node_features.unsqueeze(1)

    hidden = th.nn.functional.normalize(node_features, p=2, dim=-1)
    d = hidden.size(-1)
    scores = th.einsum("btid,btjd->btij", hidden, hidden) / math.sqrt(max(d, 1))
    eye = th.eye(scores.size(-1), device=scores.device, dtype=th.bool).view(1, 1, scores.size(-1), scores.size(-1))
    scores = scores.masked_fill(eye, -1e9)
    attn = th.softmax(scores, dim=-1)
    attn = 0.5 * (attn + attn.transpose(-1, -2))
    attn = attn.masked_fill(eye, 0.0)
    return attn.mean(dim=(0, 1))


def local_subgraph_similarity_graph(node_features, neighbor_topk=None):
    base_graph = pseudo_attention_graph(node_features)
    n_agents = base_graph.size(0)
    if neighbor_topk is None:
        neighbor_topk = max(1, n_agents // 2)
    neighbor_topk = max(1, min(int(neighbor_topk), max(n_agents - 1, 1)))

    subgraph_vectors = []
    for agent_i in range(n_agents):
        row = base_graph[agent_i].clone()
        row[agent_i] = -1e9
        _, neighbor_indices = th.topk(row, k=neighbor_topk, dim=-1)

        ordered_nodes = th.cat(
            [
                th.tensor([agent_i], device=base_graph.device, dtype=th.long),
                neighbor_indices.long(),
            ],
            dim=0,
        )
        local_adj = base_graph.index_select(0, ordered_nodes).index_select(1, ordered_nodes)
        local_adj = 0.5 * (local_adj + local_adj.transpose(0, 1))
        local_adj.fill_diagonal_(0.0)
        local_vec = local_adj.reshape(-1)
        subgraph_vectors.append(local_vec)

    subgraph_vectors = th.stack(subgraph_vectors, dim=0)
    subgraph_vectors = th.nn.functional.normalize(subgraph_vectors, p=2, dim=-1)
    similarity = th.matmul(subgraph_vectors, subgraph_vectors.transpose(0, 1))
    similarity = 0.5 * (similarity + 1.0)
    similarity.fill_diagonal_(0.0)
    return similarity


def local_subgraph_fusion_graph(node_features, neighbor_topk=None):
    base_graph = pseudo_attention_graph(node_features)
    n_agents = base_graph.size(0)
    if neighbor_topk is None:
        neighbor_topk = max(1, n_agents // 2)
    neighbor_topk = max(1, min(int(neighbor_topk), max(n_agents - 1, 1)))

    fused = base_graph.new_zeros(base_graph.shape)
    counts = base_graph.new_zeros(base_graph.shape)

    for agent_i in range(n_agents):
        row = base_graph[agent_i].clone()
        row[agent_i] = -1e9
        _, neighbor_indices = th.topk(row, k=neighbor_topk, dim=-1)
        ordered_nodes = th.cat(
            [
                th.tensor([agent_i], device=base_graph.device, dtype=th.long),
                neighbor_indices.long(),
            ],
            dim=0,
        )

        local_adj = base_graph.index_select(0, ordered_nodes).index_select(1, ordered_nodes)
        local_adj = 0.5 * (local_adj + local_adj.transpose(0, 1))
        local_adj.fill_diagonal_(0.0)

        for local_u, global_u in enumerate(ordered_nodes.tolist()):
            for local_v, global_v in enumerate(ordered_nodes.tolist()):
                if global_u == global_v:
                    continue
                fused[global_u, global_v] += local_adj[local_u, local_v]
                counts[global_u, global_v] += 1.0

    fused = fused / counts.clamp(min=1.0)
    fused = fused.masked_fill(counts == 0, 0.0)
    fused = 0.5 * (fused + fused.transpose(0, 1))
    fused.fill_diagonal_(0.0)
    return fused


def sparsify_graph(adjacency, topk=None, threshold=0.0):
    n_agents = adjacency.size(0)
    if topk is None:
        topk = max(1, n_agents // 2)
    topk = max(1, min(int(topk), max(n_agents - 1, 1)))

    sparse = adjacency.new_zeros(adjacency.shape)
    for i in range(n_agents):
        row = adjacency[i].clone()
        row[i] = -1e9
        values, indices = th.topk(row, k=topk, dim=-1)
        valid = values > float(threshold)
        if valid.any():
            sparse[i, indices[valid]] = adjacency[i, indices[valid]]

    sparse = th.maximum(sparse, sparse.transpose(0, 1))
    sparse.fill_diagonal_(0.0)
    return sparse


def adjacency_to_groups(adjacency):
    n_agents = adjacency.size(0)
    visited = [False] * n_agents
    groups = []

    for start in range(n_agents):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component = []
        while queue:
            node = queue.pop(0)
            component.append(node)
            neighbors = (adjacency[node] > 0).nonzero(as_tuple=False).view(-1).tolist()
            for nxt in neighbors:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        groups.append(sorted(component))

    groups.sort(key=lambda group_i: group_i[0])
    return groups
