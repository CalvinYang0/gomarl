import json
import os
from math import ceil

import numpy as np


def tensor_to_list(value, batch_index=None):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
    if hasattr(value, "numpy"):
        value = value.numpy()
    if batch_index is not None:
        value = value[batch_index]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def model_diagnostics(mac, batch_index=0):
    return {
        "relation_condition": tensor_to_list(getattr(mac, "latest_condition", None), batch_index=batch_index),
        "generated_interaction_head": tensor_to_list(
            getattr(mac, "latest_generated_interaction_head", None), batch_index=batch_index
        ),
        "relation_ally_attn": tensor_to_list(
            getattr(mac, "latest_relation_ally_attn", None), batch_index=batch_index
        ),
        "relation_enemy_attn": tensor_to_list(
            getattr(mac, "latest_relation_enemy_attn", None), batch_index=batch_index
        ),
    }


def save_battle_trace(trace, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "{}_trace.json".format(prefix))
    with open(trace_path, "w") as f:
        json.dump(trace, f)
    return trace_path


def render_battle_trace(
    trace,
    output_dir,
    prefix,
    frame_stride=4,
    fps=6,
    make_video=True,
    similarity_sample_size=256,
):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    frames = trace.get("frames", [])
    if not frames:
        return paths

    overview_path = os.path.join(output_dir, "{}_battle_overview.png".format(prefix))
    _plot_battle_overview(frames, overview_path, plt)
    paths["battle_overview"] = overview_path

    similarity_path = os.path.join(output_dir, "{}_relation_head_similarity.png".format(prefix))
    if _plot_similarity(frames, similarity_path, plt, similarity_sample_size):
        paths["similarity"] = similarity_path

    alignment_path = os.path.join(output_dir, "{}_relation_head_alignment.png".format(prefix))
    if _plot_alignment_summary(frames, alignment_path, plt, similarity_sample_size):
        paths["alignment"] = alignment_path

    if make_video:
        video_path = _render_battle_intent_video(frames, output_dir, prefix, plt, frame_stride=frame_stride, fps=fps)
        if video_path is not None:
            paths["video"] = video_path
        relation_video_path = _render_relation_dynamics_video(
            frames,
            output_dir,
            prefix,
            plt,
            frame_stride=frame_stride,
            fps=fps,
            sample_size=similarity_sample_size,
        )
        if relation_video_path is not None:
            paths["relation_dynamics_video"] = relation_video_path

    return paths


def _plot_battle_overview(frames, output_path, plt):
    shown = _uniform_sample(frames, min(6, len(frames)))
    cols = 3
    rows = int(ceil(len(shown) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    bounds = _battle_bounds(frames)

    for ax, frame in zip(axes, shown):
        _draw_battle_frame(ax, frame, bounds)
    for ax in axes[len(shown) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_battle_intent_video(frames, output_dir, prefix, plt, frame_stride=1, fps=6):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    selected = frames[:: max(1, int(frame_stride))]
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])

    frame_dir = os.path.join(output_dir, "{}_frames".format(prefix))
    os.makedirs(frame_dir, exist_ok=True)
    bounds = _battle_bounds(frames)
    png_paths = []
    for idx, frame in enumerate(selected):
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        prev_frame = selected[idx - 1] if idx > 0 else None
        _draw_battle_frame(ax, frame, bounds, prev_frame=prev_frame)
        fig.tight_layout()
        png_path = os.path.join(frame_dir, "{:04d}.png".format(idx))
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        png_paths.append(png_path)

    images = [imageio.imread(path) for path in png_paths]
    mp4_path = os.path.join(output_dir, "{}_battle_intent.mp4".format(prefix))
    try:
        imageio.mimsave(mp4_path, images, fps=fps)
        return mp4_path
    except Exception:
        gif_path = os.path.join(output_dir, "{}_battle_intent.gif".format(prefix))
        try:
            imageio.mimsave(gif_path, images, fps=fps)
            return gif_path
        except Exception:
            return None


def _render_relation_dynamics_video(frames, output_dir, prefix, plt, frame_stride=1, fps=6, sample_size=256):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    relation_items = []
    for frame in frames:
        diagnostics = frame.get("diagnostics") or {}
        relation = diagnostics.get("relation_condition")
        if relation is None:
            continue
        relation = np.asarray(relation, dtype=np.float32)
        if relation.ndim != 2:
            continue
        head = diagnostics.get("generated_interaction_head")
        if head is not None:
            head = np.asarray(head, dtype=np.float32)
            if head.ndim != 2 or head.shape[0] != relation.shape[0]:
                head = None
        relation_items.append({"frame": frame, "relation": relation, "head": head})

    if len(relation_items) < 2:
        return None

    selected = relation_items[:: max(1, int(frame_stride))]
    if selected[-1] is not relation_items[-1]:
        selected.append(relation_items[-1])

    rel_all = np.concatenate([item["relation"] for item in relation_items], axis=0)
    rel_projector = _fit_pca2(rel_all)
    rel_coords = [rel_projector(item["relation"]) for item in relation_items]
    rel_bounds = _coord_bounds(rel_coords)

    has_head = all(item["head"] is not None for item in relation_items)
    if has_head:
        head_all = np.concatenate([item["head"] for item in relation_items], axis=0)
        head_projector = _fit_pca2(head_all)
        head_coords = [head_projector(item["head"]) for item in relation_items]
        head_bounds = _coord_bounds(head_coords)
    else:
        head_coords = None
        head_bounds = None

    item_to_index = {id(item): idx for idx, item in enumerate(relation_items)}
    frame_dir = os.path.join(output_dir, "{}_relation_frames".format(prefix))
    os.makedirs(frame_dir, exist_ok=True)
    png_paths = []

    for out_idx, item in enumerate(selected):
        item_idx = item_to_index[id(item)]
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
        _draw_projection_panel(
            axes[0, 0],
            rel_coords,
            item_idx,
            rel_bounds,
            title="Relation pattern in 2D",
            subtitle="Each point = one agent at current timestep",
        )
        _draw_agent_similarity_panel(
            axes[0, 1],
            item["relation"],
            title="Relation similarity between agents",
        )
        if has_head:
            _draw_projection_panel(
                axes[1, 0],
                head_coords,
                item_idx,
                head_bounds,
                title="Generated MLP head in 2D",
                subtitle="Agent head-parameter trajectory",
            )
            _draw_agent_similarity_panel(
                axes[1, 1],
                item["head"],
                title="MLP-head similarity between agents",
            )
        else:
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
            axes[1, 0].text(0.5, 0.5, "No generated MLP-head parameters\nfor this model.", ha="center", va="center")

        fig.suptitle("Relation/head dynamics at timestep {}".format(item["frame"].get("t", 0)), fontsize=15)
        fig.tight_layout()
        png_path = os.path.join(frame_dir, "{:04d}.png".format(out_idx))
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        png_paths.append(png_path)

    images = [imageio.imread(path) for path in png_paths]
    mp4_path = os.path.join(output_dir, "{}_relation_head_dynamics.mp4".format(prefix))
    try:
        imageio.mimsave(mp4_path, images, fps=fps)
        return mp4_path
    except Exception:
        gif_path = os.path.join(output_dir, "{}_relation_head_dynamics.gif".format(prefix))
        try:
            imageio.mimsave(gif_path, images, fps=fps)
            return gif_path
        except Exception:
            return None


def _fit_pca2(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    if centered.shape[0] < 2:
        components = np.zeros((2, centered.shape[1]), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2]
        if components.shape[0] == 1:
            components = np.concatenate([components, np.zeros_like(components)], axis=0)

    def project(values):
        projected = (np.asarray(values, dtype=np.float32) - mean).dot(components.T)
        if projected.shape[1] < 2:
            projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
        return projected[:, :2]

    return project


def _coord_bounds(coord_by_frame):
    coords = np.concatenate(coord_by_frame, axis=0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_pad = max((x_max - x_min) * 0.12, 1e-3)
    y_pad = max((y_max - y_min) * 0.12, 1e-3)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _draw_projection_panel(ax, coord_by_frame, frame_idx, bounds, title, subtitle):
    agent_count = coord_by_frame[frame_idx].shape[0]
    colors = _agent_colors(agent_count)
    all_coords = np.concatenate(coord_by_frame, axis=0)
    ax.scatter(all_coords[:, 0], all_coords[:, 1], s=7, color="#d1d5db", alpha=0.22, label="all timesteps")
    for agent_idx in range(agent_count):
        history = np.asarray([coords[agent_idx] for coords in coord_by_frame[: frame_idx + 1] if agent_idx < coords.shape[0]])
        if history.size > 0:
            ax.plot(history[:, 0], history[:, 1], color=colors[agent_idx], linewidth=1.6, alpha=0.58)
        current = coord_by_frame[frame_idx][agent_idx]
        ax.scatter([current[0]], [current[1]], s=95, color=colors[agent_idx], edgecolor="black", linewidth=0.7)
        ax.text(current[0], current[1], " A{}".format(agent_idx), fontsize=9, weight="bold", color=colors[agent_idx])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.text(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.78},
    )


def _draw_agent_similarity_panel(ax, values, title):
    sim = _cosine_sim(values)
    im = ax.imshow(sim, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    agent_count = values.shape[0]
    labels = ["A{}".format(idx) for idx in range(agent_count)]
    ax.set_xticks(np.arange(agent_count))
    ax.set_yticks(np.arange(agent_count))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=12)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")


def _agent_colors(agent_count):
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return [palette[idx % len(palette)] for idx in range(agent_count)]


def _draw_battle_frame(ax, frame, bounds, prev_frame=None):
    snapshot = frame.get("snapshot", {})
    allies = snapshot.get("allies", [])
    enemies = snapshot.get("enemies", [])
    actions = frame.get("actions") or []
    enemy_by_id = {unit["id"]: unit for unit in enemies}
    ally_by_tag = {unit["tag"]: unit for unit in allies}
    enemy_by_tag = {unit["tag"]: unit for unit in enemies}

    ax.set_facecolor("#f7f5ef")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.grid(True, color="#ddd6c8", linewidth=0.7, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Battle intent at timestep {}  (env step {})".format(frame.get("t", 0), frame.get("t_env", 0)), fontsize=12)

    for unit in enemies:
        _draw_unit(ax, unit, color="#c0392b", label_prefix="E", marker="s")
    for unit in allies:
        _draw_unit(ax, unit, color="#2166ac", label_prefix="A", marker="o")

    for agent_id, action in enumerate(actions):
        if agent_id >= len(allies):
            continue
        source = allies[agent_id]
        if not source.get("alive", False):
            continue
        _draw_action(ax, source, int(action), enemy_by_id, snapshot.get("n_actions_no_attack", 6))

    _draw_enemy_orders(ax, enemies, ally_by_tag)
    _draw_inferred_enemy_damage(ax, frame, prev_frame, enemies)
    _draw_intent_text(ax, frame, prev_frame, enemy_by_id, ally_by_tag, enemy_by_tag)

    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_unit(ax, unit, color, label_prefix, marker):
    alpha = 1.0 if unit.get("alive", False) else 0.22
    x, y = unit["x"], unit["y"]
    ax.scatter([x], [y], s=90, c=color, marker=marker, edgecolor="black", linewidth=0.7, alpha=alpha)
    ax.text(x + 0.25, y + 0.25, "{}{}".format(label_prefix, unit["id"]), color=color, fontsize=9, alpha=alpha)

    health_max = max(float(unit.get("health_max", 1.0)), 1e-6)
    health_ratio = max(0.0, min(1.0, float(unit.get("health", 0.0)) / health_max))
    bar_w = 1.3
    ax.plot([x - bar_w / 2, x + bar_w / 2], [y - 0.45, y - 0.45], color="#222222", linewidth=2.0, alpha=alpha)
    ax.plot(
        [x - bar_w / 2, x - bar_w / 2 + bar_w * health_ratio],
        [y - 0.45, y - 0.45],
        color="#2ca25f",
        linewidth=2.0,
        alpha=alpha,
    )


def _draw_action(ax, source, action, enemy_by_id, n_actions_no_attack):
    move_delta = {
        2: (0.0, 1.4),
        3: (0.0, -1.4),
        4: (1.4, 0.0),
        5: (-1.4, 0.0),
    }
    sx, sy = source["x"], source["y"]
    if action in move_delta:
        dx, dy = move_delta[action]
        ax.arrow(sx, sy, dx, dy, color="#2563eb", width=0.035, head_width=0.28, alpha=0.78)
    elif action >= n_actions_no_attack:
        target_id = action - n_actions_no_attack
        target = enemy_by_id.get(target_id)
        if target is None:
            return
        tx, ty = target["x"], target["y"]
        _annotate_arrow(ax, source, target, color="#f97316", linestyle="-", linewidth=2.2, alpha=0.88)
        ax.text(
            (sx + tx) / 2,
            (sy + ty) / 2,
            "A{}->E{}".format(source["id"], target_id),
            color="#9a3412",
            fontsize=8,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
        )


def _draw_enemy_orders(ax, enemies, ally_by_tag):
    for enemy in enemies:
        if not enemy.get("alive", False):
            continue
        for order in enemy.get("orders", []):
            target = ally_by_tag.get(order.get("target_unit_tag"))
            if target is None or not target.get("alive", False):
                continue
            _annotate_arrow(ax, enemy, target, color="#dc2626", linestyle="-", linewidth=2.0, alpha=0.72)
            ax.text(
                (enemy["x"] + target["x"]) / 2,
                (enemy["y"] + target["y"]) / 2,
                "E{}->A{}".format(enemy["id"], target["id"]),
                color="#991b1b",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
            )


def _draw_inferred_enemy_damage(ax, frame, prev_frame, enemies):
    if prev_frame is None:
        return
    prev_allies = {
        unit["id"]: unit
        for unit in (prev_frame.get("snapshot", {}) or {}).get("allies", [])
    }
    current_allies = (frame.get("snapshot", {}) or {}).get("allies", [])
    for ally in current_allies:
        prev_ally = prev_allies.get(ally["id"])
        if prev_ally is None:
            continue
        hp_now = float(ally.get("health", 0.0)) + float(ally.get("shield", 0.0))
        hp_prev = float(prev_ally.get("health", 0.0)) + float(prev_ally.get("shield", 0.0))
        if hp_prev - hp_now <= 1e-4:
            continue
        nearest_enemy = _nearest_alive_unit(ally, enemies)
        if nearest_enemy is None:
            continue
        _annotate_arrow(ax, nearest_enemy, ally, color="#7f1d1d", linestyle="--", linewidth=1.4, alpha=0.55)


def _annotate_arrow(ax, source, target, color, linestyle="-", linewidth=1.8, alpha=0.8):
    ax.annotate(
        "",
        xy=(target["x"], target["y"]),
        xytext=(source["x"], source["y"]),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "alpha": alpha,
            "shrinkA": 8,
            "shrinkB": 8,
        },
    )


def _nearest_alive_unit(source, candidates):
    alive = [unit for unit in candidates if unit.get("alive", False)]
    if not alive:
        return None
    return min(alive, key=lambda unit: (unit["x"] - source["x"]) ** 2 + (unit["y"] - source["y"]) ** 2)


def _draw_intent_text(ax, frame, prev_frame, enemy_by_id, ally_by_tag, enemy_by_tag):
    snapshot = frame.get("snapshot", {})
    allies = snapshot.get("allies", [])
    enemies = snapshot.get("enemies", [])
    actions = frame.get("actions") or []
    n_actions_no_attack = snapshot.get("n_actions_no_attack", 6)

    ally_lines = []
    for agent_id, action in enumerate(actions):
        if agent_id >= len(allies):
            continue
        ally_lines.append("A{}: {}".format(agent_id, _action_text(int(action), enemy_by_id, n_actions_no_attack)))

    enemy_lines = []
    for enemy in enemies:
        target_texts = []
        for order in enemy.get("orders", []):
            target_tag = order.get("target_unit_tag")
            target = ally_by_tag.get(target_tag)
            prefix = "A"
            if target is None:
                target = enemy_by_tag.get(target_tag)
                prefix = "E"
            if target is not None:
                target_texts.append("{}{}".format(prefix, target["id"]))
        if target_texts:
            enemy_lines.append("E{}: target {}".format(enemy["id"], ",".join(target_texts)))

    damage_lines = _damage_text(prev_frame, frame)
    text = "Ally chosen actions\n{}\n\nEnemy visible orders\n{}\n\nObserved ally damage\n{}".format(
        "\n".join(ally_lines[:10]) if ally_lines else "None",
        "\n".join(enemy_lines[:10]) if enemy_lines else "No raw order / not visible",
        "\n".join(damage_lines[:6]) if damage_lines else "None",
    )
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )


def _action_text(action, enemy_by_id, n_actions_no_attack):
    move_names = {0: "noop", 1: "stop", 2: "move north", 3: "move south", 4: "move east", 5: "move west"}
    if action in move_names:
        return move_names[action]
    target_id = action - n_actions_no_attack
    if target_id in enemy_by_id:
        return "attack E{}".format(target_id)
    return "attack target {}".format(target_id)


def _damage_text(prev_frame, frame):
    if prev_frame is None:
        return []
    prev_allies = {
        unit["id"]: unit
        for unit in (prev_frame.get("snapshot", {}) or {}).get("allies", [])
    }
    lines = []
    for ally in (frame.get("snapshot", {}) or {}).get("allies", []):
        prev_ally = prev_allies.get(ally["id"])
        if prev_ally is None:
            continue
        hp_now = float(ally.get("health", 0.0)) + float(ally.get("shield", 0.0))
        hp_prev = float(prev_ally.get("health", 0.0)) + float(prev_ally.get("shield", 0.0))
        delta = hp_prev - hp_now
        if delta > 1e-4:
            lines.append("A{} -{:.1f} hp".format(ally["id"], delta))
    return lines


def _plot_similarity(frames, output_path, plt, sample_size):
    relation, head = _collect_relation_and_head(frames)
    if relation is None:
        return False

    relation, head = _sample_aligned(relation, head, sample_size)
    rel_sim = _cosine_sim(relation)
    has_head = head is not None and len(head) == len(relation)

    if has_head:
        head_sim = _cosine_sim(head)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    im0 = axes[0].imshow(rel_sim, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0].set_title("Relation-pattern cosine similarity")
    axes[0].set_xlabel("agent-time sample")
    axes[0].set_ylabel("agent-time sample")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if has_head:
        im1 = axes[1].imshow(head_sim, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axes[1].set_title("Generated-head cosine similarity")
        axes[1].set_xlabel("agent-time sample")
        axes[1].set_ylabel("agent-time sample")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        rel_dist, head_dist = _upper_triangle_distances(rel_sim, head_sim)
        corr = _rank_corr(rel_dist, head_dist)
        axes[2].scatter(rel_dist, head_dist, s=8, alpha=0.35, color="#4a5568")
        axes[2].set_title("Distance alignment, rank corr={:.3f}".format(corr))
        axes[2].set_xlabel("relation distance")
        axes[2].set_ylabel("head-parameter distance")
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "No generated interaction head\nfor this model/checkpoint.",
            ha="center",
            va="center",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return True


def _plot_alignment_summary(frames, output_path, plt, sample_size):
    relation, head = _collect_relation_and_head(frames)
    if relation is None or head is None:
        return False

    relation, head = _sample_aligned(relation, head, sample_size)
    if relation.shape[0] < 3:
        return False

    rel_sim = _cosine_sim(relation)
    head_sim = _cosine_sim(head)
    rel_dist, head_dist = _upper_triangle_distances(rel_sim, head_sim)
    corr = _rank_corr(rel_dist, head_dist)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    hb = ax.hexbin(
        rel_dist,
        head_dist,
        gridsize=46,
        mincnt=1,
        cmap="Blues",
        linewidths=0.0,
        alpha=0.92,
    )
    fig.colorbar(hb, ax=ax, label="number of sample pairs")

    bin_x, mean_y = _binned_mean(rel_dist, head_dist, bins=12)
    ax.plot(bin_x, mean_y, color="#d7301f", linewidth=2.7, marker="o", markersize=4.5, label="average trend")

    ax.set_title("Similar relations generate similar decision heads", fontsize=15, pad=14)
    ax.set_xlabel("Relation-pattern difference\n0 = same relation, larger = more different", fontsize=11)
    ax.set_ylabel("Generated MLP-head parameter difference\n0 = same head, larger = more different", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left")

    explanation = (
        "Each point/bin compares two agent-time samples.\n"
        "Upward trend means the hypernetwork changes the MLP head\n"
        "when the relation pattern changes.\n"
        "Rank correlation = {:.3f}".format(corr)
    )
    ax.text(
        0.03,
        0.97,
        explanation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.88},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _collect_relation_and_head(frames):
    relation_rows = []
    head_rows = []
    has_any_head = False
    for frame in frames:
        diagnostics = frame.get("diagnostics") or {}
        relation = diagnostics.get("relation_condition")
        if relation is None:
            continue
        relation = np.asarray(relation, dtype=np.float32)
        if relation.ndim != 2:
            continue

        head = diagnostics.get("generated_interaction_head")
        if head is not None:
            head = np.asarray(head, dtype=np.float32)
            if head.ndim == 2 and head.shape[0] == relation.shape[0]:
                has_any_head = True
            else:
                head = None

        for agent_idx in range(relation.shape[0]):
            relation_rows.append(relation[agent_idx])
            if head is not None:
                head_rows.append(head[agent_idx])
            else:
                head_rows.append(None)

    if not relation_rows:
        return None, None

    relation_arr = np.stack(relation_rows, axis=0)
    if not has_any_head:
        return relation_arr, None

    keep = [idx for idx, row in enumerate(head_rows) if row is not None]
    if len(keep) < 2:
        return relation_arr, None
    return relation_arr[keep], np.stack([head_rows[idx] for idx in keep], axis=0)


def _sample_aligned(relation, head, sample_size):
    sample_size = int(sample_size)
    if sample_size > 0 and relation.shape[0] > sample_size:
        indices = np.linspace(0, relation.shape[0] - 1, num=sample_size).astype(np.int64)
        relation = relation[indices]
        if head is not None:
            head = head[indices]
    return relation, head


def _cosine_sim(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(denom, 1e-8)
    return np.matmul(x, x.T)


def _upper_triangle_distances(rel_sim, head_sim):
    idx = np.triu_indices(rel_sim.shape[0], k=1)
    return 1.0 - rel_sim[idx], 1.0 - head_sim[idx]


def _binned_mean(x, y, bins=12):
    if len(x) == 0:
        return np.asarray([]), np.asarray([])
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = []
    valid_centers = []
    for left, right, center in zip(edges[:-1], edges[1:], centers):
        if right == edges[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if np.any(mask):
            valid_centers.append(center)
            means.append(float(np.mean(y[mask])))
    return np.asarray(valid_centers), np.asarray(means)


def _rank_corr(x, y):
    if len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 1e-8:
        return 0.0
    return float((rx * ry).sum() / denom)


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    return ranks


def _battle_bounds(frames):
    xs = []
    ys = []
    for frame in frames:
        snapshot = frame.get("snapshot", {})
        for unit in snapshot.get("allies", []) + snapshot.get("enemies", []):
            xs.append(unit["x"])
            ys.append(unit["y"])
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    pad = 3.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _uniform_sample(items, count):
    if count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=count).astype(np.int64)
    return [items[int(idx)] for idx in indices]
