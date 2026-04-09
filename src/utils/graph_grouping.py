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
