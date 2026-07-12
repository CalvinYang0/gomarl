#!/usr/bin/env python3
"""Print the effective training history stored in one W&B offline run."""

import argparse
import json
from pathlib import Path


TRACKED_KEYS = {"t_env", "episode", "_runtime", "_timestamp"}


def scalar_value(value_json):
    try:
        value = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, (int, float)) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="offline-run directory or run-*.wandb file")
    args = parser.parse_args()

    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    path = Path(args.run).resolve()
    if path.is_dir():
        files = list(path.glob("run-*.wandb"))
        if len(files) != 1:
            raise SystemExit("expected one run-*.wandb in {}, found {}".format(path, len(files)))
        path = files[0]
    if not path.is_file():
        raise SystemExit("not a W&B run file: {}".format(path))

    history_records = 0
    last_step = None
    observed = {key: {"last": None, "max": None} for key in TRACKED_KEYS}
    run_ids = []
    exit_runtimes = []
    final_records = 0

    scanner = DataStore()
    scanner.open_for_scan(str(path))
    try:
        while True:
            data = scanner.scan_data()
            if data is None:
                break

            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)

            if record.HasField("run"):
                run_ids.append(record.run.run_id)
            elif record.HasField("exit"):
                exit_runtimes.append(record.exit.runtime)
            elif record.HasField("final"):
                final_records += 1
            elif record.HasField("history"):
                history_records += 1
                if record.history.HasField("step"):
                    last_step = record.history.step.num
                for item in record.history.item:
                    if item.key not in TRACKED_KEYS:
                        continue
                    value = scalar_value(item.value_json)
                    if value is None:
                        continue
                    observed[item.key]["last"] = value
                    old_max = observed[item.key]["max"]
                    observed[item.key]["max"] = value if old_max is None else max(old_max, value)
    finally:
        if hasattr(scanner, "close"):
            scanner.close()

    print("file: {}".format(path))
    print("size_bytes: {}".format(path.stat().st_size))
    print("modified: {}".format(path.stat().st_mtime_ns))
    print("run_ids: {}".format(", ".join(sorted(set(run_ids))) or "<none>"))
    print("history_records: {}".format(history_records))
    print("last_history_step: {}".format(last_step))
    for key in sorted(TRACKED_KEYS):
        print("{}: last={} max={}".format(key, observed[key]["last"], observed[key]["max"]))
    print("exit_runtimes_seconds: {}".format(exit_runtimes or "<none>"))
    print("final_records: {}".format(final_records))


if __name__ == "__main__":
    main()
