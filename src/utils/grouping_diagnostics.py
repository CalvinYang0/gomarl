"""Readable, source-identifiable partition snapshots shared by both grouping paths."""


def membership(snapshot):
    names = snapshot.get("state_slot_names", snapshot.get("slot_names"))
    rows = snapshot["masks_or_probability"]
    if not names or len(rows) != 2 or any(len(row) != len(names) for row in rows):
        raise ValueError("Invalid partition dimensions")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate slot names")
    if not snapshot["row_labels"][0].startswith("important"):
        raise ValueError("Not a grouping snapshot")
    if any(a not in (0, 1) or b != 1 - a for a, b in zip(*rows)):
        raise ValueError("Groups must be binary, disjoint and exhaustive")
    return names, [bool(x) for x in rows[0]]


def enrich_snapshot(snapshot, args, logger):
    snapshot.update(schema_version=2, source_run_name=getattr(args, "wandb_run_name", None) or args.name,
                    source_run_id=getattr(getattr(logger, "wandb", None), "id", None),
                    seed=int(args.seed), map_name=args.env_args.get("map_name"),
                    unique_token=args.unique_token)
    if snapshot["row_labels"][0].startswith("important"):
        names, selected = membership(snapshot)
        snapshot["important_slot_names"] = [n for n, yes in zip(names, selected) if yes]
        snapshot["unimportant_slot_names"] = [n for n, yes in zip(names, selected) if not yes]


def log_group_details(snapshot, logger, prefix, image, snapshot_path):
    """Native training history: image plus readable membership and latest summary."""
    if not logger.use_wandb or not snapshot["row_labels"][0].startswith("important"):
        return
    names, selected = membership(snapshot)
    scores = snapshot.get("ranking_ema", snapshot.get("score", [None] * len(names)))
    if len(scores) != len(names):
        raise ValueError("Ranking/slot dimension mismatch")
    import json
    from hashlib import sha256
    # W&B 0.18.7 may query the server to serialize a Table even in offline
    # mode. Save its documented run-file format directly instead, so an
    # offline training process never needs a network request for membership.
    table = dict(columns=["slot", "group", "ranking_score"],
                 data=[[n, "important" if yes else "unimportant", score]
                       for n, yes, score in zip(names, selected, scores)])
    path = snapshot_path.parent / ("membership_" + snapshot_path.stem + ".table.json")
    path.write_text(json.dumps(table, allow_nan=False))
    logger.wandb_module.save(str(path), base_path=str(path.parent))
    raw = path.read_bytes()
    descriptor = dict(_type="table-file", path=path.name, sha256=sha256(raw).hexdigest(),
                      size=len(raw), ncols=3, nrows=len(names))
    t = snapshot["t_env"]
    logger._update_wandb_buffer(prefix + "/grouping_heatmap", image, t)
    logger._update_wandb_buffer(prefix + "/grouping_membership", descriptor, t)
    logger.wandb.summary.update({prefix + "/grouping_t_env": t,
        prefix + "/important_slot_names": snapshot["important_slot_names"],
        prefix + "/unimportant_slot_names": snapshot["unimportant_slot_names"]})
