import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _circle_layout(n_agents, radius=1.0):
    positions = []
    for idx in range(n_agents):
        angle = 2 * math.pi * idx / max(n_agents, 1)
        positions.append((radius * math.cos(angle), radius * math.sin(angle)))
    return positions


def _render_frame(group, viz_info, title):
    unit_names = viz_info.get("unit_names", [])
    alive = viz_info.get("alive", [])
    n_agents = len(unit_names)
    positions = _circle_layout(n_agents)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    ax.set_title(title)

    cmap = plt.get_cmap("tab10")
    assignment = [-1 for _ in range(n_agents)]
    for group_id, group_i in enumerate(group):
        for agent_id in group_i:
            if 0 <= agent_id < n_agents:
                assignment[agent_id] = group_id

    for group_id, group_i in enumerate(group):
        for i in range(len(group_i)):
            for j in range(i + 1, len(group_i)):
                a_i = group_i[i]
                a_j = group_i[j]
                if a_i >= n_agents or a_j >= n_agents:
                    continue
                x1, y1 = positions[a_i]
                x2, y2 = positions[a_j]
                ax.plot([x1, x2], [y1, y2], color=cmap(group_id % 10), alpha=0.55, linewidth=2)

    for agent_id, (x, y) in enumerate(positions):
        group_id = assignment[agent_id] if assignment[agent_id] >= 0 else 0
        color = cmap(group_id % 10)
        is_alive = bool(alive[agent_id]) if agent_id < len(alive) else True
        face_alpha = 0.95 if is_alive else 0.18
        edge_color = "black" if is_alive else "#666666"
        ax.scatter([x], [y], s=900, c=[color], alpha=face_alpha, edgecolors=edge_color, linewidths=2, zorder=3)
        if not is_alive:
            ax.text(x, y, "X", color="black", fontsize=16, ha="center", va="center", zorder=4)
        ax.text(
            x,
            y,
            unit_names[agent_id],
            ha="center",
            va="center",
            fontsize=8,
            color="white" if is_alive else "#333333",
            zorder=5,
        )

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def build_group_viz_frames(group_trace, group, map_name, max_frames=24):
    if not group_trace:
        return []

    if len(group_trace) > max_frames:
        indices = np.linspace(0, len(group_trace) - 1, max_frames).astype(int)
        sampled_trace = [group_trace[i] for i in indices]
    else:
        sampled_trace = group_trace

    frames = []
    for step_idx, item in enumerate(sampled_trace):
        if isinstance(item, dict):
            viz_info = item.get("viz_info")
            frame_group = item.get("group") if item.get("group") is not None else group
        else:
            viz_info = item
            frame_group = group
        if viz_info is None:
            continue
        title = "{} | t_ep={} | groups={}".format(map_name, step_idx, frame_group)
        frames.append(_render_frame(frame_group, viz_info, title))
    return frames


def _project_to_2d(points):
    if points.shape[1] == 1:
        return np.concatenate([points, np.zeros((points.shape[0], 1), dtype=points.dtype)], axis=1)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis


def build_role_scatter_image(group_trace, map_name):
    if not group_trace:
        return None

    feature_list = []
    prob_list = []
    prototypes = None

    for item in group_trace:
        if not isinstance(item, dict):
            continue
        role_features = item.get("role_features")
        role_probs = item.get("role_probs")
        role_prototypes = item.get("role_prototypes")
        if role_features is None or role_probs is None:
            continue
        if role_prototypes is not None:
            prototypes = np.asarray(role_prototypes, dtype=np.float32)
        role_features = np.asarray(role_features, dtype=np.float32)
        role_probs = np.asarray(role_probs, dtype=np.float32)
        feature_list.append(role_features)
        prob_list.append(role_probs)

    if not feature_list:
        return None

    feature_stack = np.stack(feature_list, axis=0)
    prob_stack = np.stack(prob_list, axis=0)
    points = feature_stack.mean(axis=0)
    mean_probs = prob_stack.mean(axis=0)
    colors = mean_probs.argmax(axis=-1)

    if prototypes is None:
        proto_ids = sorted(np.unique(colors).tolist())
        centroid_list = []
        for group_id in proto_ids:
            mask = colors == group_id
            if mask.any():
                centroid_list.append(points[mask].mean(axis=0))
        if not centroid_list:
            return None
        prototypes = np.stack(centroid_list, axis=0)
        proto_labels = ["centroid{}".format(group_id) for group_id in proto_ids]
    else:
        prototypes = np.asarray(prototypes, dtype=np.float32)
        proto_labels = ["role{}".format(i) for i in range(prototypes.shape[0])]

    combined = np.concatenate([points, prototypes], axis=0)
    projected = _project_to_2d(combined)
    point_xy = projected[: points.shape[0]]
    proto_xy = projected[points.shape[0]:]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for group_id in range(prototypes.shape[0]):
        mask = colors == group_id
        if mask.any():
            ax.scatter(
                point_xy[mask, 0],
                point_xy[mask, 1],
                s=22,
                alpha=0.45,
                c=[cmap(group_id % 10)],
                label="agents->role{}".format(group_id),
            )

    if points.shape[0] <= 12:
        for agent_id, (x, y) in enumerate(point_xy):
            ax.text(x, y, " a{}".format(agent_id), fontsize=8, ha="left", va="bottom")

    for group_id, (x, y) in enumerate(proto_xy):
        ax.scatter([x], [y], s=240, marker="X", c=[cmap(group_id % 10)], edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(x, y, " {}".format(proto_labels[group_id]), fontsize=10, ha="left", va="center")

    ax.set_title("{} | role scatter".format(map_name))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image
