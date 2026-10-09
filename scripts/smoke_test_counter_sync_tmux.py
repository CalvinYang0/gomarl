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
    env = dict(
        os.environ,
        REPO_DIR=str(repo),
        RUNTIME_ROOT=str(repo / "runtime"),
        PYTHON_BIN=sys.executable,
        **extra,
    )
    return subprocess.run(["bash", "-c", command], env=env, text=True,
                          capture_output=True, timeout=10)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="gomarl-sync-test-") as temp:
        repo = Path(temp)
        (repo / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
        mocks = '''
flock() { return 0; }
timeout() {
  shift
  "$@"
}
squeue() { printf 'QUEUE_QUERY=<%s>\n' "$*" >&2; return 0; }
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
        assert 'UPDATE_5M6M_FIGURES' not in result.stdout
        assert '--loop' in result.stdout
        repeated = run(launch, repo, MOCK_EXISTS="0")
        assert repeated.returncode == 0 and 'duplicate' in repeated.stdout
        assert 'TMUX_ARG=<new-session>' not in repeated.stdout
        stop = run(mocks + 'bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" stop', repo)
        assert stop.returncode == 0 and 'TMUX_ARG=<kill-session>' in stop.stdout
        restart = run(mocks + 'bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" restart', repo)
        assert restart.returncode == 0 and 'TMUX_ARG=<new-session>' in restart.stdout
        # Interrupt only the loop subprocess after its first (mocked) round.
        loop = mocks + '''
sleep() { kill -TERM "$$"; }
export -f sleep
bash "$REPO_DIR/scripts/ozstar_wandb_sync_tmux.sh" --loop
'''
        result = run(loop, repo)
        assert result.returncode == 0, result.stderr
        assert 'matched=0 uploaded=0 failed=0' in result.stdout
        assert '-h -t R -o' in result.stderr
        assert 'figure' not in result.stdout.lower()
        assert 'Final upload' not in result.stdout
        assert 'next round in' in result.stdout and 'loop stopping' in result.stdout
        # Even stale tmux environment variables cannot re-enable plotting.
        stale = run(loop, repo, UPDATE_5M6M_FIGURES="YES")
        assert stale.returncode == 0 and 'figure' not in stale.stdout.lower()
        recent_launch = run(mocks + 'bash "$REPO_DIR/scripts/ozstar_recent_wandb_sync_tmux.sh" restart', repo)
        assert recent_launch.returncode == 0
        assert 'UPDATE_5M6M_FIGURES' not in recent_launch.stdout
        recent_loop = run(mocks + '''
sleep() { exit 0; }
export -f sleep
bash "$REPO_DIR/scripts/ozstar_recent_wandb_sync_tmux.sh" --loop
''', repo)
        assert recent_loop.returncode == 0, recent_loop.stderr
        assert 'matched=0 uploaded=0 failed=0' in recent_loop.stdout
        assert '-h -t R -o' in recent_loop.stderr
        assert 'figure' not in recent_loop.stdout.lower()
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
        # An unrelated completed run remains on disk but must not be synced.
        live = repo / "runtime/wandb/offline-run-20261009_010000-live1"
        old = repo / "runtime/wandb/offline-run-20261008_010000-done1"
        for directory, run_id in ((live, "live1"), (old, "done1")):
            (directory / "files").mkdir(parents=True)
            (directory / ("run-" + run_id + ".wandb")).write_bytes(b"mock record")
            (directory / "files/config.yaml").write_text(
                "wandb_run_name: smac_3m_linear_obs_baseline_10m_s1_valuediag\n")
        active = run(mocks + '''
squeue() {
  [[ "$*" == *"-t R"* ]] || return 1
  echo '123|smac_3m_linear_obs_baseline_10m_s1_valuediag|2026-10-09T01:00:00'
}
scontrol() { echo "JobId=123 WorkDir=$REPO_DIR StdOut=/missing StdErr=/missing"; }
stat() { echo 11; }
timeout() {
  shift
  [[ "$2" == '-m' && "$3" == 'wandb' ]] || return 1
  echo "MOCK_WANDB_SYNC $*"
}
export -f squeue scontrol stat timeout
bash "$REPO_DIR/scripts/ozstar_sync_running_counter_once.sh"
''', repo)
        assert active.returncode == 0, active.stderr
        assert 'matched=1 uploaded=1 failed=0' in active.stdout
        assert 'live1' in active.stdout and 'done1' not in active.stdout
        assert old.is_dir()  # Nothing is deleted, including completed data.
        for filename in ("ozstar_wandb_sync_tmux.sh", "ozstar_recent_wandb_sync_tmux.sh"):
            content = (ROOT / "scripts" / filename).read_text()
            assert 'plot_5m6m' not in content
            assert 'ozstar_finalize_wandb' not in content
            assert 'ozstar_sync_recent_wandb.py' not in content
    print("PASS: both launchers sync running jobs only, no figures/cleanup, "
          "restart/retry, empty queue and overlap protection (mocked tools)")
