import torch as th


def hidden_similarity_graph(agent_hidden):
    if agent_hidden.dim() != 4:
        raise ValueError("Expected agent_hidden with shape [batch, time, n_agents, hidden_dim].")

    b, t, n_agents, _ = agent_hidden.size()
    hidden = agent_hidden.reshape(b * t, n_agents, -1)
    hidden = th.nn.functional.normalize(hidden, p=2, dim=-1)
    sim = th.matmul(hidden, hidden.transpose(1, 2))
    sim = sim.mean(dim=0)
    sim = (sim + sim.transpose(0, 1)) / 2.0
    sim = (sim + 1.0) / 2.0
    sim.fill_diagonal_(0.0)
    return sim


def adjacency_to_groups(adj_matrix, threshold):
    n_agents = adj_matrix.size(0)
    visited = [False for _ in range(n_agents)]
    groups = []

    for start in range(n_agents):
        if visited[start]:
            continue

        stack = [start]
        component = []
        visited[start] = True

        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = th.where(adj_matrix[node] > threshold)[0].tolist()
            for neigh in neighbors:
                if not visited[neigh]:
                    visited[neigh] = True
                    stack.append(neigh)

        groups.append(sorted(component))

    groups.sort(key=lambda group_i: group_i[0])
    return groups


def group_assignment(group, n_agents):
    assignment = [-1 for _ in range(n_agents)]
    for group_id, group_i in enumerate(group):
        for agent_id in group_i:
            assignment[agent_id] = group_id
    return assignment
