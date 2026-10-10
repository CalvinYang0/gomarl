#!/usr/bin/env python3
"""Per-job final sync, verify and remove ONLY closed local W&B offline runs.

Preview by default. --apply permits final upload and irreversible local removal.
Never delete Sacred, Slurm logs, shared caches, or a live/unknown job's records.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

RUN_RE = re.compile(r"offline-run-([0-9]{8})_[0-9]{6}-([A-Za-z0-9]+)")
RUN_PATH_RE = re.compile(r"(?:/[^\s\"']+/)?wandb/offline-run-[0-9]{8}_[0-9]{6}-[A-Za-z0-9]+")
LOG_RE = re.compile(r".+_([0-9]+)\.(?:out|err)$")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}
SDK_FILES = {"config.yaml", "wandb-metadata.json", "wandb-summary.json",
             "requirements.txt", "conda-environment.yaml", "output.log"}


def command(args):
    return subprocess.check_output(args, text=True, timeout=60).strip()


def live_jobs():
    user = command(["id", "-un"])
    # Include pending, completing, suspended and requeued jobs, not only RUNNING.
    return {line.split("_", 1)[0].strip() for line in command(
        ["squeue", "-u", user, "-h", "-o", "%A"]).splitlines() if line.strip()}


def job_finished(job_id, repo):
    rows = command(["sacct", "-j", job_id, "-S", "1970-01-01", "-X", "-n", "-P",
                    "--format=JobIDRaw,State,WorkDir%4096"])
    found = []
    for line in rows.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) != 3 or fields[0].strip() != job_id:
            continue
        state = fields[1].strip().split()[0].rstrip("+")
        found.append(state in TERMINAL and Path(fields[2].strip()) == repo)
    return bool(found) and all(found)


def job_index(log_root, wandb_root, repo, progress=False):
    result = {}
    for number, path in enumerate(sorted(log_root.iterdir()), 1):
        if progress and number % 25 == 0:
            print("INDEX scanned {} log entries".format(number), flush=True)
        match = LOG_RE.fullmatch(path.name)
        if not match or path.is_symlink() or not path.is_file():
            continue
        with path.open(errors="replace") as handle:
            for line in handle:
                for run in RUN_PATH_RE.finditer(line):
                    path = Path(run.group(0))
                    if not path.is_absolute():
                        path = repo / path
                    if path.resolve().parent == wandb_root and RUN_RE.fullmatch(path.name):
                        result.setdefault(path.name, set()).add(match.group(1))
    return result


def snapshot(directory):
    """Never follow links; allow only the SDK's debug-core.log link itself."""
    result = {}
    def scan_error(error):
        raise error
    for parent, directories, files in os.walk(directory, followlinks=False, onerror=scan_error):
        for name in directories + files:
            path = Path(parent) / name
            key = str(path.relative_to(directory))
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode):
                if key != "logs/debug-core.log":
                    raise RuntimeError("Symlink in run directory; retain: " + str(path))
                # Core points this at a shared service log. Snapshot the link,
                # not its target (which may change for other running jobs).
                result[key] = (entry.st_size, entry.st_mtime_ns, entry.st_ino,
                               os.readlink(path))
            elif stat.S_ISREG(entry.st_mode):
                result[key] = (entry.st_size, entry.st_mtime_ns, entry.st_ino)
            elif not stat.S_ISDIR(entry.st_mode):
                raise RuntimeError("Special file in run directory; retain: " + str(path))
    return result


def records(data):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    store = DataStore()
    try:
        store.open_for_scan(str(data))
        while True:
            payload = store.scan_data()
            if payload is None:
                return
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            yield record
    finally:
        store.close()


def identity(data, run_id, entity, project):
    destination, name = None, None
    exited = False
    for record in records(data):
        if record.HasField("run"):
            run = record.run
            if (run.run_id != run_id or run.project != project or
                    (run.entity and run.entity != entity)):
                raise RuntimeError("Original run ID/destination differs; retain")
            destination = "/".join((run.entity or entity, run.project, run.run_id))
            name = run.display_name
        exited |= record.HasField("exit")
    # wandb-core 0.18.7 finishes with exit and has no legacy final record.
    if not destination or not exited:
        raise RuntimeError("Run not demonstrably closed (missing identity/exit); retain")
    return destination, name


def histories(data):
    pending, previous = None, -1
    for record in records(data):
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = item.key or ".".join(item.nested_key)
            row[key] = json.loads(item.value_json)
        step = row.get("_step")
        if step is None and record.history.HasField("step"):
            step = record.history.step.num
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step < previous:
            raise RuntimeError("Unresolved/non-monotonic history step; retain")
        row["_step"] = step
        if pending is not None and step != previous:
            yield pending
            pending = None
        if pending is None:
            pending = {}
        pending.update(row)
        previous = step
    if pending is not None:
        yield pending


def equal_value(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal_value(v, right[k]) for k, v in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal_value(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (math.isnan(left) and math.isnan(right)) or math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def verify_cloud(remote, data, directory, expected_name):
    if remote.name != expected_name or remote.state not in {"finished", "failed", "crashed", "killed"}:
        raise RuntimeError("Cloud identity/state not confirmed; retain")
    cloud = iter(remote.scan_history(page_size=1000))
    current = next(cloud, None)
    count = 0
    # Streaming full history comparison, not sampled history or summary alone.
    for row in histories(data):
        while current is not None and current.get("_step", -1) < row["_step"]:
            current = next(cloud, None)
        if current is None or current.get("_step") != row["_step"]:
            raise RuntimeError("Missing cloud history step " + str(row["_step"]))
        if any(k not in current or not equal_value(v, current[k]) for k, v in row.items()):
            raise RuntimeError("Cloud history differs at step " + str(row["_step"]))
        count += 1
    if not count:
        raise RuntimeError("No history available for verification; retain")
    files = {f.name: f for f in remote.files()}
    root = directory / "files"
    verified = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = str(path.relative_to(root))
        if name in SDK_FILES:
            continue  # SDK-managed config/summary/console are not byte-identical server copies.
        file = files.get(name)
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes = {digest.hexdigest(), base64.b64encode(digest.digest()).decode()}
        if file is None or file.size != path.stat().st_size or file.md5 not in hashes:
            raise RuntimeError("Cloud saved file/media missing or different: " + name)
        verified += 1
    return count, verified


def unchanged(before, after, allow_sync_metadata=False):
    # The 0.18.7 sender rewrites SDK config/summary/requirements during sync.
    # Permit those changes ONLY during that subprocess, never during verification.
    def keep(key):
        return not key.endswith(".wandb.synced") and not (
            allow_sync_metadata and key in {"files/" + name for name in SDK_FILES})
    return {k: v for k, v in before.items() if keep(k)} == {
        k: v for k, v in after.items() if keep(k)}


def process(directory, jobs, args, api_factory):
    if not jobs or jobs & live_jobs() or not all(job_finished(j, args.repo) for j in jobs):
        print("RETAIN live/unknown job: " + directory.name, flush=True)
        return 0
    if directory.is_symlink() or directory.resolve().parent != args.wandb_root:
        raise RuntimeError("Run directory escaped validated root")
    match = RUN_RE.fullmatch(directory.name)
    if not match or not directory.is_dir():
        raise RuntimeError("Not an ordinary offline-run directory")
    data = directory / ("run-" + match.group(2) + ".wandb")
    before = snapshot(directory)
    if not data.is_file() or time.time() - max(p[1] for p in before.values()) / 1e9 < args.min_age:
        print("RETAIN missing/recently modified data: " + directory.name, flush=True)
        return 0
    destination, name = identity(data, match.group(2), args.entity, args.project)
    size = sum(p[0] for p in before.values())
    print("CANDIDATE jobs={} GiB={:.3f} run={}".format(
        ",".join(sorted(jobs)), size / 1024**3, destination), flush=True)
    if not args.apply:
        return 0
    print("SYNC final upload: " + directory.name, flush=True)
    subprocess.run([sys.executable, "-m", "wandb", "sync", "--append",
                    "--include-offline", "--include-synced", "--no-mark-synced",
                    "--skip-console", "-e", args.entity, "-p", args.project, str(directory)],
                   check=True, timeout=args.sync_timeout)
    after_sync = snapshot(directory)
    if not unchanged(before, after_sync, allow_sync_metadata=True):
        raise RuntimeError("Run payload changed during final sync; retain")
    print("VERIFY full cloud history and saved files: " + destination, flush=True)
    remote = api_factory().run(destination)
    points, files = verify_cloud(remote, data, directory, name)
    if (jobs & live_jobs() or not all(job_finished(j, args.repo) for j in jobs) or
            directory.is_symlink() or directory.resolve().parent != args.wandb_root or
            not unchanged(after_sync, snapshot(directory))):
        raise RuntimeError("Job/files changed during verification; retain")
    # Log evidence BEFORE irreversible removal; do not delete if disk cannot save audit.
    audit = dict(time=time.time(), directory=str(directory), jobs=sorted(jobs),
                 cloud=destination, bytes=size, history_points=points, verified_files=files,
                 status="verified_before_remove")
    with (args.wandb_root / "completed_cleanup_audit.jsonl").open("a") as handle:
        handle.write(json.dumps(audit) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    shutil.rmtree(directory)
    print("REMOVED local W&B copy: {} ({:.3f} GiB); cloud={}, Sacred/logs untouched".format(
        directory, size / 1024**3, destination), flush=True)
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get("REPO_DIR", "/home/kyang/code/gomarl-dual-branch")))
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get("RUNTIME_ROOT", "/home/kyang/gomarl-runtime/gomarl-dual-branch")))
    parser.add_argument("--min-age", type=int, default=300)
    parser.add_argument("--sync-timeout", type=int, default=900)
    parser.add_argument("--entity", default="hjh331-sjtu")
    parser.add_argument("--project", default="gomarl")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    runtime = args.runtime_root
    if (not runtime.is_absolute() or runtime.is_symlink() or runtime.resolve() != runtime or
            runtime in {Path("/"), Path("/home/kyang"), args.repo} or
            Path("/home/kyang") not in runtime.parents):
        raise SystemExit("Require an explicit non-symlink experiment runtime beneath /home/kyang")
    args.wandb_root = runtime / "wandb"
    logs = runtime / "ozstar_logs"
    if any(p.is_symlink() or not p.is_dir() for p in (args.wandb_root, logs)):
        raise SystemExit("W&B/log roots must be existing ordinary directories")
    if args.min_age < 0 or args.sync_timeout <= 0:
        raise SystemExit("Invalid age/timeout")
    print("START {}: indexing job logs under {}".format(
        "APPLY" if args.apply else "PREVIEW", logs), flush=True)
    import wandb
    index = job_index(logs, args.wandb_root, args.repo, progress=True)
    print("INDEX done: {} run directories mapped".format(len(index)), flush=True)
    removed = failed = 0
    with (args.wandb_root / ".gomarl-sync-once.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("WAIT: another sync holds .gomarl-sync-once.lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        print("CHECK: sync lock acquired; checking scheduler and run directories", flush=True)
        # A scheduler failure aborts before selecting any destructive target.
        live_jobs()
        for directory in sorted(args.wandb_root.iterdir()):
            if not RUN_RE.fullmatch(directory.name):
                continue
            try:
                removed += process(directory, index.get(directory.name, set()), args,
                                   lambda: wandb.Api(timeout=90))
            except Exception as exc:
                failed += 1
                print("RETAIN {}: {}".format(directory.name, exc), file=sys.stderr, flush=True)
    print("{} freed={:.3f} GiB verification_failures={}; Sacred, job logs and shared caches kept".format(
        "APPLY" if args.apply else "PREVIEW", removed / 1024**3, failed), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
