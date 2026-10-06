#!/usr/bin/env python3
"""Read-only comparison of historical and threeway Counter baseline seed 1."""
import argparse
import json
from pathlib import Path

NAMES = (
    "grf_counter_paper_linear_singlehead_5m_s1",
    "grf_counter_singlehead_baseline_10m_s1_threeway",
)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path(
        "/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    parser.add_argument("--project", default="hjh331-sjtu/gomarl")
    args = parser.parse_args()
    candidates = {name: [] for name in NAMES}
    for path in (args.runtime_root / "results/sacred").glob("*/*/*/config.json"):
        try:
            config = json.loads(path.read_text())
            name = config.get("wandb_run_name")
            if name not in candidates:
                continue
            metadata = json.loads((path.parent / "run.json").read_text())
            candidates[name].append((str(metadata.get("start_time", "")),
                                     str(path), config, metadata))
        except (OSError, ValueError) as exc:
            print("WARNING", path, str(exc))
    selected = {}
    for name in NAMES:
        attempts = sorted(candidates[name], key=lambda item: item[0])
        print("\nRUN", name, "Sacred attempts:", len(attempts))
        for started, source, _, metadata in attempts:
            print(started, source, "status=" + str(metadata.get("status")))
        if attempts:
            started, source, config, metadata = attempts[-1]
            selected[name] = flatten(config)
            print("SELECTED", source)
            print("REPOSITORIES", json.dumps(metadata.get("experiment", {}).get(
                "repositories", []), ensure_ascii=False))
        else:
            # Deleted local records must not be silently reconstructed from
            # current defaults. Fetch the actual cloud run configuration.
            import wandb
            runs = list(wandb.Api(timeout=60).runs(args.project,
                filters={"display_name": name}, order="-created_at"))
            if not runs:
                print("MISSING: no local or cloud configuration")
                continue
            run = runs[0]
            selected[name] = flatten(dict(run.config))
            print("SELECTED CLOUD", run.id, run.created_at, run.url)
            print("COMMIT", getattr(run, "commit", None))
            print("Cloud attempts:", [(r.id, str(r.created_at)) for r in runs])
    if len(selected) == 2:
        old, new = (selected[name] for name in NAMES)
        print("\nACTUAL CONFIG DIFFERENCES (old -> new):")
        for key in sorted(set(old) | set(new)):
            a, b = old.get(key, "<absent>"), new.get(key, "<absent>")
            if a != b:
                print(key, json.dumps(a, ensure_ascii=False), "->",
                      json.dumps(b, ensure_ascii=False))
    else:
        print("\nINCOMPLETE: cannot establish equivalence without both configurations")
    print("\nNEW RUN WORKER RESTART EVIDENCE:")
    hits = 0
    for path in sorted((args.runtime_root / "ozstar_logs").glob(NAMES[1] + "_*")):
        if path.suffix not in {".out", ".err"}:
            continue
        print("SCANNING", path)
        with path.open(errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if "Restarting all environment workers" in line or "Environment rollout failed" in line:
                    hits += 1
                    print(str(path) + ":" + str(number), line.rstrip())
    print("Restart/error log lines:", hits,
          "(zero is inconclusive if logs were deleted)")


if __name__ == "__main__":
    main()
