#!/usr/bin/env python3
"""Show the latest slot-level semantic routes from OzSTAR job logs."""

import argparse
import getpass
import re
import subprocess
from pathlib import Path


ROUTE_MARKER = "Semantic slot route |"
ROUTE_RE = re.compile(
    r"t_env=(?P<t_env>\d+) \| TOKEN=\[(?P<token>.*?)\] "
    r"\| BIAS=\[(?P<bias>.*?)\] \| frozen=(?P<frozen>\d+) "
    r"\| version=(?P<version>\d+)"
)

LABELS = (
    ("obscons", "OBS"),
    ("tempstable", "TEMP"),
    ("gradimp", "GIMP"),
    ("gradcons", "GCONS"),
    ("paramsens", "PSENS"),
    ("counterfact", "CF"),
)


def active_job_ids():
    result = subprocess.run(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%A"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def last_route_line(path, chunk_size=64 * 1024):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        remainder = b""
        marker = ROUTE_MARKER.encode()
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size) + remainder
            lines = data.split(b"\n")
            remainder = lines[0]
            for line in reversed(lines[1:]):
                if marker in line:
                    return line.decode("utf-8", errors="replace")
        if marker in remainder:
            return remainder.decode("utf-8", errors="replace")
    return None


def split_names(value):
    return [name for name in value.split(",") if name]


def label_for(path, job_id):
    stem = path.stem
    for needle, label in LABELS:
        if needle in stem:
            return label
    return job_id


def route_for_job(log_dir, job_id):
    candidates = list(log_dir.glob("*_{}.out".format(job_id)))
    candidates += list(log_dir.glob("*_{}.err".format(job_id)))
    best = None
    for path in candidates:
        line = last_route_line(path)
        if line is None:
            continue
        match = ROUTE_RE.search(line)
        if match is None:
            continue
        record = {
            "job": job_id,
            "label": label_for(path, job_id),
            "path": path,
            "t_env": int(match.group("t_env")),
            "frozen": int(match.group("frozen")),
            "version": int(match.group("version")),
            "token": split_names(match.group("token")),
            "bias": split_names(match.group("bias")),
        }
        if best is None or record["t_env"] > best["t_env"]:
            best = record
    return best


def slot_group(name):
    match = re.match(r"(ally|enemy)_(\d+)_", name)
    if match:
        return "{}_{}".format(match.group(1), match.group(2))
    return "self"


def slot_sort_key(name):
    feature_order = {
        "move_north": 0,
        "move_south": 1,
        "move_east": 2,
        "move_west": 3,
        "visible": 4,
        "attack_available": 4,
        "distance": 5,
        "relative_x": 6,
        "relative_y": 7,
        "health": 8,
        "shield": 9,
    }
    match = re.match(r"(ally|enemy)_(\d+)_(.*)", name)
    if match:
        side_order = 1 if match.group(1) == "ally" else 2
        entity_index = int(match.group(2))
        feature = match.group(3)
    elif name.startswith("opponent_"):
        match = re.match(r"opponent_(\d+)_(.*)", name)
        side_order = 2
        entity_index = int(match.group(1))
        feature = match.group(2)
    elif name.startswith("ball_"):
        side_order = 3
        entity_index = 0
        feature = name[5:]
    else:
        side_order = 0
        entity_index = 0
        feature = name[5:] if name.startswith("self_") else name
    feature_rank = feature_order.get(feature)
    if feature_rank is None:
        if feature.startswith("unit_type_"):
            feature_rank = 10
        elif feature.startswith("last_action_"):
            feature_rank = 11
        else:
            feature_rank = 12
    return side_order, entity_index, feature_rank, feature


def print_flat_mask(route):
    """Print a stage-two config value in the capturer's internal slot order."""
    mapping = {name: 1 for name in route["token"]}
    mapping.update({name: 0 for name in route["bias"]})
    if any(name.startswith("self_position_") for name in mapping):
        # GRF compact-observation order differs from the display grouping.
        axis_rank = {"x": 0, "y": 1, "z": 2}

        def grf_internal_key(name):
            match = re.match(r"self_position_(x|y)$", name)
            if match:
                return 0, 0, axis_rank[match.group(1)]
            match = re.match(r"ally_(\d+)_relative_(x|y)$", name)
            if match:
                return 1, int(match.group(1)), axis_rank[match.group(2)]
            match = re.match(r"self_direction_(x|y)$", name)
            if match:
                return 2, 0, axis_rank[match.group(1)]
            match = re.match(r"ally_(\d+)_direction_(x|y)$", name)
            if match:
                return 3, int(match.group(1)), axis_rank[match.group(2)]
            match = re.match(r"opponent_(\d+)_relative_(x|y)$", name)
            if match:
                return 4, int(match.group(1)), axis_rank[match.group(2)]
            match = re.match(r"opponent_(\d+)_direction_(x|y)$", name)
            if match:
                return 5, int(match.group(1)), axis_rank[match.group(2)]
            ball_rank = {
                "ball_relative_x": 0,
                "ball_relative_y": 1,
                "ball_height": 2,
                "ball_direction_x": 3,
                "ball_direction_y": 4,
                "ball_direction_z": 5,
            }
            if name in ball_rank:
                return 6, 0, ball_rank[name]
            raise ValueError("Unknown GRF semantic slot: {}".format(name))

        ordered_names = sorted(mapping, key=grf_internal_key)
    else:
        ordered_names = sorted(mapping, key=slot_sort_key)
    # A compact bit-string survives Slurm --export; commas do not.
    mask = "".join(str(mapping[name]) for name in ordered_names)
    print(mask)


def feature_name(name, group):
    prefix = "{}_".format(group)
    if name.startswith(prefix):
        return name[len(prefix):]
    if group == "self" and name.startswith("self_"):
        return name[5:]
    return name


def print_job_route(route):
    mapping = {name: "TOKEN" for name in route["token"]}
    mapping.update({name: "BIAS" for name in route["bias"]})
    groups = {}
    for name in sorted(mapping, key=slot_sort_key):
        groups.setdefault(slot_group(name), []).append(name)

    print("\n" + "=" * 88)
    print("{} | job={} | t_env={} | frozen={} | version={}".format(
        route["label"],
        route["job"],
        route["t_env"],
        route["frozen"],
        route["version"],
    ))
    print("log: {}".format(route["path"]))
    print("=" * 88)
    for group, names in groups.items():
        token = [feature_name(name, group) for name in names if mapping[name] == "TOKEN"]
        bias = [feature_name(name, group) for name in names if mapping[name] == "BIAS"]
        print("\n[{}]".format(group))
        print("  TOKEN : {}".format(", ".join(token) if token else "-"))
        print("  BIAS  : {}".format(", ".join(bias) if bias else "-"))


def print_binary_mask(route):
    """Print the exact shared 0/1 route tensor reconstructed from the log."""
    mapping = {name: 1 for name in route["token"]}
    mapping.update({name: 0 for name in route["bias"]})
    groups = {}
    for name in sorted(mapping, key=slot_sort_key):
        groups.setdefault(slot_group(name), []).append(name)

    print("\n" + "=" * 88)
    print("{} | job={} | t_env={} | frozen={} | version={}".format(
        route["label"],
        route["job"],
        route["t_env"],
        route["frozen"],
        route["version"],
    ))
    print("Shared mask: 1=Transformer TOKEN, 0=Simple BIAS")
    print("Broadcast shape: self [1,1,D], ally [1,1,N_ally,D], enemy [1,1,N_enemy,D]")
    print("All agents receive this same mask; only their observation values differ.")
    print("=" * 88)
    for group, names in groups.items():
        fields = [feature_name(name, group) for name in names]
        values = [str(mapping[name]) for name in names]
        print("\n[{}]".format(group))
        print("  fields: [{}]".format(", ".join(fields)))
        print("  mask:   [{}]".format(", ".join(values)))


def print_route_matrix(routes):
    print("\nLATEST ROUTE SUMMARY")
    print("{:<8} {:<10} {:>10} {:>7} {:>8} {:>7} {:>7}".format(
        "LABEL", "JOB", "T_ENV", "FROZEN", "VERSION", "TOKEN", "BIAS"
    ))
    for route in routes:
        print("{:<8} {:<10} {:>10} {:>7} {:>8} {:>7} {:>7}".format(
            route["label"],
            route["job"],
            route["t_env"],
            route["frozen"],
            route["version"],
            len(route["token"]),
            len(route["bias"]),
        ))

    slots = []
    seen = set()
    for route in routes:
        for name in route["token"] + route["bias"]:
            if name not in seen:
                seen.add(name)
                slots.append(name)

    slots.sort(key=slot_sort_key)

    route_maps = []
    for route in routes:
        mapping = {name: "T" for name in route["token"]}
        mapping.update({name: "B" for name in route["bias"]})
        route_maps.append(mapping)

    slot_width = max(28, max(len(name) for name in slots))
    labels = [route["label"] for route in routes]
    print("\nSLOT ROUTE MATRIX  (T=Transformer token, B=Simple bias)")
    print("{:<{width}}  {}".format(
        "SLOT", "  ".join("{:>6}".format(label) for label in labels), width=slot_width
    ))
    print("-" * (slot_width + 2 + 8 * len(labels)))

    previous_group = None
    for slot in slots:
        group = slot_group(slot)
        if previous_group is not None and group != previous_group:
            print()
        values = [mapping.get(slot, "-") for mapping in route_maps]
        print("{:<{width}}  {}".format(
            slot, "  ".join("{:>6}".format(value) for value in values), width=slot_width
        ))
        previous_group = group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_ids", nargs="*", help="Slurm job IDs; defaults to active jobs")
    parser.add_argument("--log-dir", default="ozstar_logs")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="compare jobs side by side instead of printing one job per section",
    )
    parser.add_argument(
        "--raw-mask",
        action="store_true",
        help="print the exact shared 1/0 route mask in self/ally/enemy tensor order",
    )
    parser.add_argument(
        "--flat-mask",
        action="store_true",
        help="print only the stage-two 0/1 bit-string",
    )
    args = parser.parse_args()

    job_ids = args.job_ids or active_job_ids()
    log_dir = Path(args.log_dir)
    routes = [route_for_job(log_dir, job_id) for job_id in job_ids]
    missing = [job_id for job_id, route in zip(job_ids, routes) if route is None]
    routes = [route for route in routes if route is not None]

    if missing:
        print("No route summary yet: {}".format(", ".join(missing)))
    if not routes:
        raise SystemExit("No semantic slot routes found.")

    output_modes = sum((args.matrix, args.raw_mask, args.flat_mask))
    if output_modes > 1:
        parser.error("--matrix, --raw-mask and --flat-mask are mutually exclusive")
    if args.matrix:
        print_route_matrix(routes)
    elif args.raw_mask:
        for route in routes:
            print_binary_mask(route)
    elif args.flat_mask:
        if len(routes) != 1:
            parser.error("--flat-mask requires exactly one job with a route summary")
        print_flat_mask(routes[0])
    else:
        for route in routes:
            print_job_route(route)


if __name__ == "__main__":
    main()
