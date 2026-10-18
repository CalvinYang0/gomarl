"""Branch-separated mixer attribution plots; no model or score recomputation."""
import copy
import math

from utils.grouping_diagnostics import membership


def prepare_group_score_snapshot(snapshot):
    """Recover the exact scoring partition, never the post-migration partition.

    Existing flat scores are owner-branch scores, not scores for both branches.
    Splitting them by their recorded ownership is lossless; it does not measure
    the counterfactual score in the other mixer or change ranking/training.
    """
    result = copy.deepcopy(snapshot)
    names, current = membership(result)
    migration = result.get("migration")
    evaluated = result.get("score_important_membership")
    if migration is not None:
        before, after = migration["important_before"], migration["important_after"]
        if list(after) != current:
            raise ValueError("Migration/current partition mismatch")
        if evaluated is not None and list(evaluated) != list(before):
            raise ValueError("Scoring/migration partition mismatch")
        evaluated = before
        source = "recorded pre-migration partition"
    elif evaluated is not None:
        source = "recorded scoring partition"
    elif "migration" in result:
        # Native snapshots always store migration=None when no regroup occurs.
        evaluated = current
        source = "recorded partition; no migration at this snapshot"
    else:
        raise ValueError("No scoring-partition provenance; cannot safely split legacy scores")
    if len(evaluated) != len(names) or any(x not in (0, 1) for x in evaluated):
        raise ValueError("Invalid scoring partition")
    masks = [[int(x) for x in evaluated], [int(not x) for x in evaluated]]
    grouped = {}
    for key in ("qchange", "tderror", "normalized_removal_gain",
                "qtot_variance_difference", "ranking_ema"):
        if key not in result:
            continue
        values = result[key]
        if len(values) != len(names) or any(not math.isfinite(x) for x in values):
            raise ValueError("Invalid score vector: " + key)
        grouped[key] = [[value if keep else 0.0 for value, keep in zip(values, mask)]
                        for mask in masks]
    if "qchange" not in grouped or "tderror" not in grouped:
        raise ValueError("Missing saved mixer Q-change/TD scores")
    result.update(group_score_schema_version=3, score_important_membership=list(evaluated),
                  score_masks=masks, group_scores=grouped, score_partition_source=source,
                  score_partition_note="Heatmap and bars use the same scoring partition; "
                      "masks_or_probability and membership tables retain the post-migration partition",
                  group_score_note="Each slot is measured only in its owning mixer. "
                      "Out-of-group entries are structural zeros, not evidence of low importance.")
    return result


def group_score_panels(snapshot):
    grouped = snapshot["group_scores"]
    if snapshot["variant"] == "group_variance":
        metrics = [("ranking_ema", "EMA variance-difference ranking (historical)"),
                   ("qtot_variance_difference", "Var(Q_tot) - Var(Q_tot_replaced)")]
    else:
        metrics = [("qchange", "mean abs Q_tot change")]
        metrics.append(("normalized_removal_gain", "normalized masking benefit")
                       if snapshot["variant"] == "group_calibration" else
                       ("tderror", "positive mean TD error increase"))
    return [(group, key, label, grouped[key][group])
            for key, label in metrics for group in range(2)]


def render_group_score_snapshot(snapshot, path, title=None):
    """Five panels: exact scoring masks + separate scores for each branch."""
    import matplotlib.pyplot as plt
    result = prepare_group_score_snapshot(snapshot)
    names = result["state_slot_names"]
    panels = group_score_panels(result)
    fig, axes = plt.subplots(5, 1, sharex=True,
                             figsize=(max(14, len(names) * .20), 11), constrained_layout=True)
    try:
        pixel = axes[0].imshow(result["score_masks"], aspect="auto", vmin=0, vmax=1,
                               cmap="Reds", interpolation="nearest")
        axes[0].set_yticks([0, 1], labels=["important mask at scoring", "unimportant mask at scoring"])
        fig.colorbar(pixel, ax=axes[0], ticks=[0, 1])
        for axis, (group, key, label, values) in zip(axes[1:], panels):
            mask = result["score_masks"][group]
            for j, keep in enumerate(mask):
                if not keep:
                    axis.axvspan(j - .5, j + .5, color="0.94", zorder=0)
            axis.bar(range(len(names)), values, color=("tab:blue", "tab:orange")[group])
            axis.axhline(0, color="black", linewidth=.5)
            axis.set_ylabel(("important" if group == 0 else "unimportant") + " group\n" + label,
                            fontsize=8)
        # Equal scales within each metric make the two branches comparable visually.
        for a, b in ((axes[1], axes[2]), (axes[3], axes[4])):
            limits = (min(a.get_ylim()[0], b.get_ylim()[0]), max(a.get_ylim()[1], b.get_ylim()[1]))
            a.set_ylim(*limits)
            b.set_ylim(*limits)
        axes[-1].set_xticks(range(len(names)), labels=names, rotation=90, fontsize=6)
        axes[-1].set_xlim(-.5, len(names) - .5)
        title = title or "{} / {} / seed={} / t_env={}".format(
            result.get("map_name", "unknown map"), result["variant"], result.get("seed", "?"), result["t_env"])
        if str(result.get("source_run_name", "")).startswith("SYNTHETIC"):
            title = "SYNTHETIC QA — NOT EXPERIMENT DATA\n" + title
        fig.suptitle(title + "\nMasks and scores: " + result["score_partition_source"] +
                     "; gray = excluded from this mixer", fontsize=11)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)
    return result
