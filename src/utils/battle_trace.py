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
        video_path = _render_video(frames, output_dir, prefix, plt, frame_stride=frame_stride, fps=fps)
        if video_path is not None:
            paths["video"] = video_path

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


def _render_video(frames, output_dir, prefix, plt, frame_stride=4, fps=6):
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
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        _draw_battle_frame(ax, frame, bounds)
        fig.tight_layout()
        png_path = os.path.join(frame_dir, "{:04d}.png".format(idx))
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        png_paths.append(png_path)

    images = [imageio.imread(path) for path in png_paths]
    mp4_path = os.path.join(output_dir, "{}_battle.mp4".format(prefix))
    try:
        imageio.mimsave(mp4_path, images, fps=fps)
        return mp4_path
    except Exception:
        gif_path = os.path.join(output_dir, "{}_battle.gif".format(prefix))
        try:
            imageio.mimsave(gif_path, images, fps=fps)
            return gif_path
        except Exception:
            return None


def _draw_battle_frame(ax, frame, bounds):
    snapshot = frame.get("snapshot", {})
    allies = snapshot.get("allies", [])
    enemies = snapshot.get("enemies", [])
    actions = frame.get("actions") or []
    enemy_by_id = {unit["id"]: unit for unit in enemies}

    ax.set_facecolor("#f7f5ef")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.grid(True, color="#ddd6c8", linewidth=0.7, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("t={}  env={}".format(frame.get("t", 0), frame.get("t_env", 0)), fontsize=10)

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
        ax.arrow(sx, sy, dx, dy, color="#2b6cb0", width=0.025, head_width=0.22, alpha=0.65)
    elif action >= n_actions_no_attack:
        target_id = action - n_actions_no_attack
        target = enemy_by_id.get(target_id)
        if target is None:
            return
        tx, ty = target["x"], target["y"]
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops={"arrowstyle": "->", "color": "#9e2f2f", "lw": 1.1, "alpha": 0.72},
        )


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
