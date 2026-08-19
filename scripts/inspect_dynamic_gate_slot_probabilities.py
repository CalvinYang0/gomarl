#!/usr/bin/env python3
"""Print the latest per-slot Linear/Attention gate probabilities from W&B."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


LINEAR_PREFIX = "dynamic_gate_linear_probability_slot_"
ATTENTION_PREFIX = "dynamic_gate_attention_probability_slot_"


def scalar_value(value_json):
    try:
        value = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def resolve_run_file(path):
    path = Path(path).resolve()
    if path.is_dir():
        files = list(path.glob("run-*.wandb"))
        if len(files) != 1:
            raise SystemExit(
                "expected one run-*.wandb in {}, found {}".format(
                    path, len(files)
                )
            )
        path = files[0]
    if not path.is_file():
        raise SystemExit("not a W&B run file: {}".format(path))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="offline-run directory or run-*.wandb file")
    args = parser.parse_args()

    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    path = resolve_run_file(args.run)
    latest = {"linear": {}, "attention": {}}
    slot_order = []
    last_history_step = None

    scanner = DataStore()
    scanner.open_for_scan(str(path))
    try:
        while True:
            data = scanner.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if not record.HasField("history"):
                continue
            if record.history.HasField("step"):
                last_history_step = record.history.step.num
            for item in record.history.item:
                branch = None
                slot_name = None
                if item.key.startswith(LINEAR_PREFIX):
                    branch = "linear"
                    slot_name = item.key[len(LINEAR_PREFIX) :]
                elif item.key.startswith(ATTENTION_PREFIX):
                    branch = "attention"
                    slot_name = item.key[len(ATTENTION_PREFIX) :]
                if branch is None:
                    continue
                value = scalar_value(item.value_json)
                if value is None:
                    continue
                if slot_name not in slot_order:
                    slot_order.append(slot_name)
                latest[branch][slot_name] = value
    finally:
        if hasattr(scanner, "close"):
            scanner.close()

    if not slot_order:
        raise SystemExit(
            "no per-slot dynamic-gate metrics found in {}\n"
            "This run was created before per-slot logging was added.".format(path)
        )

    print("file: {}".format(path))
    print("last_history_step: {}".format(last_history_step))
    print("\nCross-slot selectivity (latest recorded slot means):")
    for branch in ("linear", "attention"):
        values = [
            latest[branch][slot_name]
            for slot_name in slot_order
            if slot_name in latest[branch]
        ]
        if values:
            print(
                "  {:<9} mean={:6.2%} min={:6.2%} max={:6.2%} "
                "across-slot std={:6.2%}".format(
                    branch,
                    mean(values),
                    min(values),
                    max(values),
                    pstdev(values),
                )
            )
    print("\nPer-slot probabilities:")
    print(
        "{:<36} {:>12} {:>12} {:>12} {:>12}".format(
            "slot", "linear keep", "linear drop", "attn keep", "attn drop"
        )
    )
    for slot_name in slot_order:
        linear = latest["linear"].get(slot_name)
        attention = latest["attention"].get(slot_name)
        linear_keep = "N/A" if linear is None else "{:.2%}".format(linear)
        linear_drop = "N/A" if linear is None else "{:.2%}".format(1.0 - linear)
        attention_keep = (
            "N/A" if attention is None else "{:.2%}".format(attention)
        )
        attention_drop = (
            "N/A" if attention is None else "{:.2%}".format(1.0 - attention)
        )
        print(
            "{:<36} {:>12} {:>12} {:>12} {:>12}".format(
                slot_name,
                linear_keep,
                linear_drop,
                attention_keep,
                attention_drop,
            )
        )


if __name__ == "__main__":
    main()
