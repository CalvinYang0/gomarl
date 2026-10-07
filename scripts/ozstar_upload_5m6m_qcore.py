#!/usr/bin/env python3
"""Incremental scalar-only individual-run mirrors for six 5m6m diagnostics runs.

Original offline/cloud runs remain untouched. Upload core Q metrics plus other
training scalars, not images/checkpoints/88 diagnostic statistics. Includes
completed runs; stable per-source mirror IDs resume instead of making snapshots.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_linear_directkl_four_model_3seeds import local_run_index
from utils.value_diagnostics import CORE_VALUE_METRICS

NAMES = tuple("smac_5m6m_{}_10m_s{}_valuediag".format(model, seed)
              for model in ("linear_obs_baseline", "id_baseline") for seed in (1, 2, 3))


def keep_scalar(key, value):
    return (not key.startswith("_") and isinstance(value, (int, float))
            and math.isfinite(value)
            and (not key.startswith("test_value/") or key in CORE_VALUE_METRICS))


def records(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
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
            row = {}
            step = record.history.step.num if record.history.HasField("step") else None
            for item in record.history.item:
                key = ".".join(item.nested_key) if item.nested_key else item.key
                try:
                    value = json.loads(item.value_json)
                except (ValueError, TypeError):
                    continue
                if key == "_step" and step is None:
                    step = value
                if keep_scalar(key, value):
                    row[key] = value
            if isinstance(step, (int, float)) and step >= 0 and row:
                yield int(step), row
    finally:
        if hasattr(scanner, "close"):
            scanner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="hjh331-sjtu/gomarl")
    args = parser.parse_args()
    import wandb
    root = Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    output = root / "qcore_upload"
    output.mkdir(parents=True, exist_ok=True)
    wandb_root = Path(os.environ.get("WANDB_ROOT", str(root / "wandb")))
    entity, project = args.project.split("/", 1)
    index = local_run_index(wandb_root, set(NAMES))
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = output / "uploaded_sources.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for name in NAMES:
            paths = index.get(name, [])
            if not paths:
                print("MISSING offline run: " + name, flush=True)
                continue
            path = max(paths, key=lambda p: p.parent.name)
            source_id = path.stem[len("run-"):]
            stat = path.stat()
            fingerprint = [str(path), stat.st_size, stat.st_mtime_ns]
            state_key = args.project + ":" + source_id
            if state.get(state_key) == fingerprint:
                print("UNCHANGED, already uploaded: " + name, flush=True)
                continue
            mirror_id = hashlib.sha256((args.project + ":qcore-v1:" + source_id).encode()).hexdigest()[:16]
            # Cloud next-step is authoritative: interrupted uploads are safe to
            # retry; no local cursor can mark unacknowledged rows as uploaded.
            with wandb.init(entity=entity, project=project, id=mirror_id, resume="allow",
                            name=name + "_qcore", group="5m6m_individual_qcore",
                            job_type="filtered-training-history", mode="online", dir=str(output),
                            config={"source_run_id": source_id, "source_run_name": name,
                                    "filtered_copy": True, "q_metrics": sorted(CORE_VALUE_METRICS)}) as run:
                next_step = run.step
                count = 0
                for step, row in records(path):
                    if step < next_step:
                        continue
                    run.log(row, step=step)
                    count += 1
                print("{}: uploaded {} new scalar rows -> {}".format(name, count, run.url), flush=True)
            state[state_key] = fingerprint
            temporary = state_path.with_suffix(".tmp")
            with temporary.open("w") as handle:
                json.dump(state, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(state_path)


if __name__ == "__main__":
    main()
