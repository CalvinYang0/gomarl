import math

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
        ax.text(x, y, unit_names[agent_id], ha="center", va="center", fontsize=8, color="white" if is_alive else "#333333", zorder=5)

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
    for step_idx, viz_info in enumerate(sampled_trace):
        title = "{} | t_ep={} | groups={}".format(map_name, step_idx, group)
        frames.append(_render_frame(group, viz_info, title))
    return frames
