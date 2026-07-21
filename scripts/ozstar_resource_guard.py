#!/usr/bin/env python3
"""Cancel persistently underutilized OzSTAR Slurm jobs.

The guard samples aggregate CPU time and resident memory for every running job.
CPU efficiency is computed from the delta between consecutive samples so the
decision reflects the recent interval instead of the job's lifetime average.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path("/home/kyang/code/gomarl")


def run(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout


def duration_seconds(value: str) -> float | None:
    value = value.strip().split(".", 1)[0]
    if not value or value in {"Unknown", "N/A"}:
        return None
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = value.split(":")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = (int(part) for part in parts)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def memory_gib(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"0", "Unknown", "N/A"}:
        return None
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2).upper()
    factors = {
        "": 1 / 1024**3,
        "K": 1 / 1024**2,
        "M": 1 / 1024,
        "G": 1,
        "T": 1024,
        "P": 1024**2,
    }
    return number * factors[unit]


def load_state(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def parse_requested_memory(job_id: str) -> float | None:
    output = run(["scontrol", "show", "job", "-o", job_id], check=False)
    match = re.search(r"ReqTRES=[^ ]*mem=([^, ]+)", output)
    return memory_gib(match.group(1)) if match else None


def live_usage(job_id: str) -> tuple[float | None, float | None]:
    output = run(
        [
            "sstat",
            "-j",
            f"{job_id}.batch",
            "-n",
            "-P",
            "--format=JobID,AveCPU,AveRSS",
        ],
        check=False,
    )
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and ".batch" in fields[0]:
            return duration_seconds(fields[1]), memory_gib(fields[2])
    return None, None


def running_jobs(user_name: str) -> list[dict[str, Any]]:
    output = run(
        ["squeue", "-h", "-u", user_name, "-t", "R", "-o", "%A|%j|%C"],
        check=False,
    )
    jobs = []
    for line in output.splitlines():
        fields = line.split("|", 2)
        if len(fields) != 3:
            continue
        job_id, name, cpus = fields
        try:
            alloc_cpus = int(cpus)
        except ValueError:
            continue
        jobs.append({"job_id": job_id, "name": name, "cpus": alloc_cpus})
    return jobs


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")


def low_duration(now: float, started_at: float | None) -> float:
    return max(0.0, now - started_at) if started_at is not None else 0.0


def sample_once(args: argparse.Namespace, state: dict[str, Any], logger: Logger) -> None:
    now = time.time()
    jobs = running_jobs(args.user)
    active_ids = {job["job_id"] for job in jobs}
    exclude = re.compile(args.exclude) if args.exclude else None

    logger.write(
        f"sample running_jobs={len(jobs)} threshold={args.threshold:.1f}% "
        f"window={args.window_minutes:.0f}m mode={args.mode} cancel={args.cancel}"
    )
    logger.write(
        "JOBID      NAME                                 CPU_NOW MEM_NOW "
        "CPU_LOW_MIN MEM_LOW_MIN ACTION"
    )

    for job in jobs:
        job_id = job["job_id"]
        name = job["name"]
        if exclude and exclude.search(name):
            logger.write(f"{job_id:<10} {name:<36} excluded")
            continue

        record = state.setdefault(job_id, {})
        cpu_total, current_rss_gib = live_usage(job_id)
        requested_gib = parse_requested_memory(job_id)

        cpu_efficiency = None
        previous_cpu = record.get("cpu_total_seconds")
        previous_time = record.get("sample_time")
        if cpu_total is not None and previous_cpu is not None and previous_time is not None:
            wall_delta = now - float(previous_time)
            cpu_delta = cpu_total - float(previous_cpu)
            if wall_delta > 0 and cpu_delta >= 0 and job["cpus"] > 0:
                cpu_efficiency = 100 * cpu_delta / (wall_delta * job["cpus"])

        memory_efficiency = None
        if current_rss_gib is not None and requested_gib and requested_gib > 0:
            memory_efficiency = 100 * current_rss_gib / requested_gib

        if cpu_efficiency is None:
            record["cpu_low_since"] = None
        elif cpu_efficiency < args.threshold:
            if record.get("cpu_low_since") is None:
                record["cpu_low_since"] = now
        else:
            record["cpu_low_since"] = None

        if memory_efficiency is None:
            record["memory_low_since"] = None
        elif memory_efficiency < args.threshold:
            if record.get("memory_low_since") is None:
                record["memory_low_since"] = now
        else:
            record["memory_low_since"] = None

        cpu_low = low_duration(now, record.get("cpu_low_since"))
        memory_low = low_duration(now, record.get("memory_low_since"))
        window_seconds = args.window_minutes * 60
        cpu_trigger = cpu_efficiency is not None and cpu_low >= window_seconds
        memory_trigger = memory_efficiency is not None and memory_low >= window_seconds
        triggered = (
            cpu_trigger or memory_trigger
            if args.mode == "any"
            else cpu_trigger and memory_trigger
        )

        action = "KEEP"
        if triggered:
            reasons = []
            if cpu_trigger:
                reasons.append(f"CPU<{args.threshold:g}% for {cpu_low / 60:.0f}m")
            if memory_trigger:
                reasons.append(f"MEM<{args.threshold:g}% for {memory_low / 60:.0f}m")
            reason = ", ".join(reasons)
            if args.cancel:
                result = subprocess.run(["scancel", job_id], capture_output=True, text=True)
                action = "CANCELLED" if result.returncode == 0 else "CANCEL_FAILED"
                logger.write(
                    f"decision job={job_id} name={name} action={action} reason={reason} "
                    f"stderr={result.stderr.strip()!r}"
                )
            else:
                action = "WOULD_CANCEL"
                logger.write(
                    f"decision job={job_id} name={name} action={action} reason={reason}"
                )

        cpu_text = "?" if cpu_efficiency is None else f"{cpu_efficiency:6.1f}%"
        memory_text = "?" if memory_efficiency is None else f"{memory_efficiency:6.1f}%"
        logger.write(
            f"{job_id:<10} {name:<36} {cpu_text:>7} {memory_text:>7} "
            f"{cpu_low / 60:11.1f} {memory_low / 60:11.1f} {action}"
        )

        record.update(
            {
                "name": name,
                "cpu_total_seconds": cpu_total,
                "sample_time": now,
                "current_rss_gib": current_rss_gib,
                "requested_gib": requested_gib,
            }
        )

    for job_id in list(state):
        if job_id not in active_ids:
            del state[job_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=os.environ.get("USER", getpass.getuser()))
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--window-minutes", type=float, default=60.0)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument(
        "--mode",
        choices=("any", "both"),
        default="any",
        help="any: CPU or memory can trigger; both: CPU and memory must trigger",
    )
    parser.add_argument("--exclude", default="", help="job-name regex to ignore")
    parser.add_argument("--cancel", action="store_true", help="run scancel when triggered")
    parser.add_argument("--once", action="store_true", help="take one sample and exit")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_REPO / "ozstar_logs/resource_guard_state.json",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_REPO / "ozstar_logs/resource_guard.log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for command in ("squeue", "sstat", "scontrol"):
        if subprocess.run(["bash", "-lc", f"command -v {command}"], capture_output=True).returncode:
            print(f"{command} is unavailable; run this on an OzSTAR login node.", file=sys.stderr)
            return 2

    logger = Logger(args.log_file)
    logger.write(
        f"resource guard started; mode={args.mode}. The any/OR policy may "
        f"cancel CPU-efficient jobs whose requested memory is under {args.threshold:g}% utilized"
    )
    state = load_state(args.state_file)
    # A monitor restart breaks the proof of continuous underutilization. Keep the
    # previous counters for the next CPU delta, but restart both low-use clocks.
    for record in state.values():
        record["cpu_low_since"] = None
        record["memory_low_since"] = None
    while True:
        try:
            sample_once(args, state, logger)
            save_state(args.state_file, state)
        except Exception as error:  # Keep a long-running guard observable and alive.
            logger.write(f"sample_error type={type(error).__name__} message={error}")
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
