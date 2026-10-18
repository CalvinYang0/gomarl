#!/usr/bin/env python3
"""One pass over ended manifest jobs; no tmux, polling, cleanup or training.

Appends available local records to their original W&B run IDs. The existing
signature ledger skips unchanged uploads verified visible through their original
scalar snapshot, not full media completeness. Missing local data are reported.
"""
import argparse
import os
from pathlib import Path

from ozstar_calibration_suite_sync import sync_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-cloud', action='store_true',
                        help='Check each exact original run in the cloud; upload missing/behind local records')
    args = parser.parse_args(argv)
    repo = Path(os.environ.get('REPO_DIR', '/home/kyang/code/gomarl-dual-branch')).resolve()
    runtime = Path(os.environ.get('RUNTIME_ROOT', '/home/kyang/gomarl-runtime/gomarl-dual-branch')).resolve()
    python = os.environ.get('PYTHON_BIN', '/home/kyang/.conda/envs/marl_cpu/bin/python')
    extra = {'verify_existing': True} if args.verify_cloud else {}
    return sync_once(repo, runtime, python, ended_only=True, **extra)


if __name__ == '__main__':
    main()
