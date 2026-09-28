#!/usr/bin/env python3
"""Render the four paper main-result plots from complete three-seed W&B runs.

Only the main-result model families are considered: VDN, QMIX, ID-based
hypernetwork, observation-based hypernetwork, and HyperSelect.  A curve is
included only when seeds 1, 2, and 3 all exist.  Rendering is delegated to
``plot_wandb_seed_aggregate.py``, whose defaults reproduce CASH's mean/std
smoothing and downsampling pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PLOTTER = ROOT / "scripts" / "plot_wandb_seed_aggregate.py"
SEEDS = (1, 2, 3)

SCENES = (
    ("grf_counter", "Counterattack Easy", "test_game_win_mean", "counter"),
    ("grf_pass", "Pass and Shoot", "test_game_win_mean", "pass"),
    ("smac_5m6m", "5m vs. 6m", "test_battle_won_mean", "5m6m"),
    ("smac_mmm2", "MMM2", "test_battle_won_mean", "mmm2"),
)

MODEL_LABELS = {
    "vdn": "VDN",
    "qmix": "QMIX",
    "id_hypernet": "ID-based Hypernet",
    "obs_hypernet": "Obs-based Hypernet",
    "hyperselect": "HyperSelect",
}
MODEL_ORDER = tuple(MODEL_LABELS)


def discover_complete_series(
    run_names: Iterable[str], scene: str
) -> List[Tuple[str, str]]:
    """Return ordered ``(label, template)`` pairs with all three seeds."""
    pattern = re.compile(
        r"^{}\_paper\_(vdn|qmix|id_hypernet|obs_hypernet|hyperselect)"
        r"\_(5m|10m)\_s([123])$".format(re.escape(scene))
    )
    groups: Dict[Tuple[str, str], set] = {}
    for name in run_names:
        match = pattern.match(name)
        if not match:
            continue
        model, budget, seed = match.groups()
        groups.setdefault((model, budget), set()).add(int(seed))

    complete = [key for key, seeds in groups.items() if seeds == set(SEEDS)]
    complete.sort(key=lambda key: (MODEL_ORDER.index(key[0]), int(key[1][:-1])))

    model_counts: Dict[str, int] = {}
    for model, _ in complete:
        model_counts[model] = model_counts.get(model, 0) + 1

    result = []
    for model, budget in complete:
        label = MODEL_LABELS[model]
        if model_counts[model] > 1:
            label += " ({} steps)".format(budget.upper())
        template = "{}_paper_{}_{}_s{{seed}}".format(scene, model, budget)
        result.append((label, template))
    return result


def fetch_run_names(project: str) -> Sequence[str]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required; use the marl_cpu environment") from exc
    api = wandb.Api(timeout=90)
    return [run.name for run in api.runs(project, per_page=500)]


def fetch_local_run_names(wandb_root: Path) -> Sequence[str]:
    pattern = re.compile(
        r"(?:grf_(?:counter|pass)|smac_(?:5m6m|mmm2))_paper_"
        r"(?:vdn|qmix|id_hypernet|obs_hypernet|hyperselect)_"
        r"(?:5m|10m)_s[123]"
    )
    names = set()
    for config in wandb_root.glob("offline-run-*/files/config.yaml"):
        try:
            names.update(pattern.findall(config.read_text(errors="ignore")))
        except OSError:
            continue
    return sorted(names)


def git_push_outputs(outputs: Sequence[Path], remote: str, branch: str) -> None:
    relative = [str(path.relative_to(ROOT)) for path in outputs]
    subprocess.run(["git", "add", "--"] + relative, cwd=ROOT, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT
    ).returncode
    if staged == 0:
        print("No figure changes to commit")
        return
    subprocess.run(
        ["git", "commit", "-m", "Add three-seed CASH-style main result plots"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "push", remote, "HEAD:" + branch], cwd=ROOT, check=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="hjh331-sjtu/gomarl")
    parser.add_argument(
        "--wandb-root",
        type=Path,
        default=Path(os.environ.get(
            "WANDB_ROOT",
            "/home/kyang/gomarl-runtime/gomarl-dual-branch/wandb",
        )),
        help="Retained local offline-run root (preferred over cloud discovery)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "main_results",
    )
    parser.add_argument("--push", action="store_true",
                        help="Commit the four PNG files and push them")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="codex/dual-branch-benefit-drop")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_names = fetch_local_run_names(args.wandb_root.resolve())
    if run_names:
        print("Discovered {} retained local main-result runs".format(len(run_names)))
    else:
        print("No retained main-result runs found; falling back to cloud discovery")
        run_names = fetch_run_names(args.project)
        print("Discovered {} cloud W&B runs".format(len(run_names)))

    outputs: List[Path] = []
    for scene, title, metric, filename in SCENES:
        series = discover_complete_series(run_names, scene)
        if not series:
            raise RuntimeError(
                "{} has no complete main-result group with seeds 1, 2, 3".format(scene)
            )
        print("{}: {}".format(scene, ", ".join(label for label, _ in series)))
        output = output_dir / (filename + "_three_seed_cash.png")
        command = [
            sys.executable,
            str(PLOTTER),
            "--project", args.project,
            "--wandb-root", str(args.wandb_root.resolve()),
            "--metric", metric,
            "--seeds", "1", "2", "3",
            "--percent",
            "--title", title,
            "--max-step", "10000000",
            "--output", str(output),
        ]
        for label, template in series:
            command.extend(["--series", "{}={}".format(label, template)])
        subprocess.run(command, cwd=ROOT, check=True)
        outputs.append(output)

    if len(outputs) != 4 or not all(path.is_file() for path in outputs):
        raise RuntimeError("Expected exactly four rendered PNG files")
    print("Rendered four figures:")
    for path in outputs:
        print(path)
    if args.push:
        git_push_outputs(outputs, args.remote, args.branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
