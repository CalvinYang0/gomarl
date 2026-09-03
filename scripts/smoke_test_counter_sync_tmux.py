#!/usr/bin/env python3
"""Exercise tmux launcher and one loop round using shell mocks, no HPC calls."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ozstar_wandb_sync_tmux.sh"


def run(command, repo, **extra):
    env = dict(os.environ, REPO_DIR=str(repo), PYTHON_BIN=sys.executable, **extra)
    return subprocess.run(["bash", "-c", command], env=env, text=True,
                          capture_output=True, timeout=10)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="gomarl-sync-test-") as temp:
        repo = Path(temp)
        (repo / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
        mocks = '''
flock() { return 0; }
timeout() { "$@"; }
squeue() { return 0; }
scontrol() { return 0; }
tmux() {
  printf 'TMUX_ARG=<%s>\n' "$@"
  if [[ "$1" == has-session ]]; then return "${MOCK_EXISTS:-1}"; fi
}
export -f flock timeout squeue scontrol tmux
'''
        launch = mocks + 'bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" start'
        result = run(launch, repo)
        assert result.returncode == 0, result.stderr
        assert 'TMUX_ARG=<new-session>' in result.stdout
        assert 'INTERVAL_SECONDS=600' in result.stdout
        assert '--loop' in result.stdout
        repeated = run(launch, repo, MOCK_EXISTS="0")
        assert repeated.returncode == 0 and 'duplicate' in repeated.stdout
        assert 'TMUX_ARG=<new-session>' not in repeated.stdout
        stop = run(mocks + 'bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" stop', repo)
        assert stop.returncode == 0 and 'TMUX_ARG=<kill-session>' in stop.stdout
        # Interrupt only the loop subprocess after its first (mocked) round.
        loop = mocks + '''
sleep() { kill -TERM "$$"; }
export -f sleep
bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" --loop
'''
        result = run(loop, repo)
        assert result.returncode == 0, result.stderr
        assert 'matched=0 uploaded=0 failed=0' in result.stdout
        assert 'next round in' in result.stdout and 'loop stopping' in result.stdout
        failure = run(mocks + '''
squeue() { return 1; }
export -f squeue
sleep() { kill -TERM "$$"; }
export -f sleep
bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" --loop
''', repo)
        assert failure.returncode == 0
        assert 'will retry next round' in failure.stdout
        locked = run(mocks + '''
flock() { return 1; }
export -f flock
bash "$REPO_DIR/scripts/ozstar_sync_running_counter_once.sh"
''', repo)
        assert locked.returncode == 0 and 'skipping overlapping' in locked.stdout
    print("PASS: launch/reuse/stop, immediate round, empty queue, retry and overlap protection (mocked tools)")
