#!/usr/bin/env python3
"""Synthetic offline SDK records + mocked Slurm/cloud; no real uploads/deletions."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import ozstar_clean_uploaded_completed_runs as cleaner


def fixture(root, run_id="abcdefgh", closed=True):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore
    directory = root / ("offline-run-20261008_120000-" + run_id)
    directory.mkdir()
    data = directory / ("run-" + run_id + ".wandb")
    store = DataStore()
    store.open_for_write(str(data))
    record = pb.Record()
    record.run.run_id = run_id
    record.run.entity = "hjh331-sjtu"
    record.run.project = "gomarl"
    record.run.display_name = "synthetic_run"
    store.write(record)
    rows = [{"_step": 0, "t_env": 10000, "test_battle_won_mean": 0.4},
            {"_step": 1, "t_env": 10000000, "test_battle_won_mean": 0.8}]
    for row in rows:
        record = pb.Record()
        for key, value in row.items():
            item = record.history.item.add()
            item.key, item.value_json = key, json.dumps(value)
        store.write(record)
    if closed:
        # Core backend writes exit without the legacy final record.
        for field in ("exit",):
            record = pb.Record()
            getattr(record, field).SetInParent()
            store.write(record)
    store.close()
    media = directory / "files/media/videos/battle.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"synthetic video fixture")
    file = SimpleNamespace(name="media/videos/battle.mp4", size=media.stat().st_size,
                           md5=base64.b64encode(hashlib.md5(media.read_bytes()).digest()).decode())
    remote = SimpleNamespace(name="synthetic_run", state="finished",
                             scan_history=lambda **kwargs: iter(deepcopy(rows)),
                             files=lambda: iter([file]))
    return directory, data, remote, file, rows


def expect_retained(directory, args, api, error=None, sync=None, jobs={"123"}):
    with patch.object(cleaner, "live_jobs", return_value=set()), \
         patch.object(cleaner, "job_finished", return_value=True), \
         patch.object(cleaner.subprocess, "run", side_effect=sync) as upload:
        try:
            cleaner.process(directory, jobs, args, lambda: api)
        except Exception as exc:
            assert error is not None and isinstance(exc, error), exc
        else:
            assert error is None
        assert directory.is_dir()
        return upload.call_count


def real_sdk_check(runtime):
    """Reproduce the production core layout, with no network upload."""
    import wandb
    sdk_root = runtime / "sdk"
    sdk_root.mkdir()
    run = wandb.init(mode="offline", dir=str(sdk_root), entity="hjh331-sjtu",
                     project="gomarl", name="cleanup_sdk_fixture",
                     settings=wandb.Settings(console="off", _disable_stats=True))
    directory = Path(run.dir).parent.resolve()
    run_id = run.id
    run.log({"t_env": 10000, "test_battle_won_mean": 0.4})
    run.log({"t_env": 10000000, "test_battle_won_mean": 0.8})
    run.finish()
    link = directory / "logs/debug-core.log"
    assert link.is_symlink(), "Expected real wandb-core 0.18.7 debug log link"
    target = link.resolve()
    assert target.is_file()
    before = cleaner.snapshot(directory)
    assert before["logs/debug-core.log"][3] == str(link.readlink())
    data = directory / ("run-" + run_id + ".wandb")
    rows = list(cleaner.histories(data))
    saved = []
    for path in (directory / "files").rglob("*"):
        if path.is_file():
            name = str(path.relative_to(directory / "files"))
            if name not in cleaner.SDK_FILES:
                saved.append(SimpleNamespace(name=name, size=path.stat().st_size,
                    md5=hashlib.md5(path.read_bytes()).hexdigest()))
    remote = SimpleNamespace(name="cleanup_sdk_fixture", state="finished",
                             scan_history=lambda **kwargs: iter(rows),
                             files=lambda: iter(saved))
    args = SimpleNamespace(repo=runtime / "repo", wandb_root=directory.parent,
                           apply=True, min_age=0, sync_timeout=10,
                           entity="hjh331-sjtu", project="gomarl")
    with patch.object(cleaner, "live_jobs", return_value=set()), \
         patch.object(cleaner, "job_finished", return_value=True), \
         patch.object(cleaner.subprocess, "run"):
        assert cleaner.process(directory, {"123"}, args,
                               lambda: SimpleNamespace(run=lambda path: remote)) > 0
    assert not directory.exists() and target.is_file()


def main():
    with tempfile.TemporaryDirectory(prefix="gomarl-safe-cleanup-") as tmp:
        runtime = Path(tmp).resolve()
        root = runtime / "wandb"
        root.mkdir()
        logs = runtime / "ozstar_logs"
        logs.mkdir()
        sacred = runtime / "results/sacred/keep.json"
        sacred.parent.mkdir(parents=True)
        sacred.write_text("{}")
        directory, data, remote, media, rows = fixture(root)
        log = logs / "synthetic_run_123.out"
        log.write_text("W&B run data saved in " + str(directory) + "\n")
        assert cleaner.job_index(logs, root, runtime / "repo") == {directory.name: {"123"}}
        unrelated_log = logs / "unrelated_456.out"
        unrelated_log.write_text("/another/root/wandb/" + directory.name + "\n")
        assert cleaner.job_index(logs, root, runtime / "repo") == {directory.name: {"123"}}
        args = SimpleNamespace(repo=runtime / "repo", wandb_root=root, apply=False,
                               min_age=0, sync_timeout=10, entity="hjh331-sjtu", project="gomarl")
        api = SimpleNamespace(run=lambda path: remote)
        assert cleaner.identity(data, "abcdefgh", args.entity, args.project) == (
            "hjh331-sjtu/gomarl/abcdefgh", "synthetic_run")
        assert list(cleaner.histories(data)) == rows
        assert cleaner.verify_cloud(remote, data, directory, "synthetic_run") == (2, 1)
        assert expect_retained(directory, args, api) == 0  # Preview never uploads/removes.
        args.apply = True
        assert expect_retained(directory, args, api, jobs=set()) == 0
        with patch.object(cleaner, "live_jobs", return_value={"123"}), \
             patch.object(cleaner.subprocess, "run") as upload:
            assert cleaner.process(directory, {"123"}, args, lambda: api) == 0
            upload.assert_not_called()
        args.min_age = 300
        assert expect_retained(directory, args, api) == 0
        args.min_age = 0
        assert expect_retained(directory, args, api, subprocess.CalledProcessError,
                               subprocess.CalledProcessError(1, "sync")) == 1
        assert expect_retained(directory, args, api, subprocess.TimeoutExpired,
                               subprocess.TimeoutExpired("sync", 1)) == 1
        def unavailable(path):
            raise ConnectionError("synthetic cloud unavailable")
        expect_retained(directory, args, SimpleNamespace(run=unavailable), ConnectionError)
        remote.scan_history = lambda **kwargs: iter(rows[:1])
        expect_retained(directory, args, api, RuntimeError)
        remote.scan_history = lambda **kwargs: iter([rows[0], dict(rows[1], test_battle_won_mean=0.7)])
        expect_retained(directory, args, api, RuntimeError)
        remote.scan_history = lambda **kwargs: iter(rows)
        media.md5 = "wrong hash despite matching file size"
        expect_retained(directory, args, api, RuntimeError)
        media.md5 = base64.b64encode(hashlib.md5(
            (directory / "files/media/videos/battle.mp4").read_bytes()).digest()).decode()
        remote.state = "running"
        expect_retained(directory, args, api, RuntimeError)
        remote.state = "finished"
        def added_file(*unused, **kwargs):
            (directory / "new_writer.txt").write_text("new data after snapshot")
        expect_retained(directory, args, api, RuntimeError, added_file)
        (directory / "new_writer.txt").unlink()
        with patch.object(cleaner, "live_jobs", side_effect=[set(), {"123"}]), \
             patch.object(cleaner, "job_finished", return_value=True), \
             patch.object(cleaner.subprocess, "run"):
            try:
                cleaner.process(directory, {"123"}, args, lambda: api)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Requeued/active job must block removal")
        assert directory.is_dir()
        open_dir, _, _, _, _ = fixture(root, "unfinished", closed=False)
        expect_retained(open_dir, args, api, RuntimeError)
        outside = runtime / "protected"
        outside.mkdir()
        (outside / "keep").write_text("protected")
        link = root / "offline-run-20261008_130000-link"
        link.symlink_to(outside, target_is_directory=True)
        expect_retained(link, args, api, RuntimeError)
        (directory / "unsafe_link").symlink_to(outside, target_is_directory=True)
        expect_retained(directory, args, api, RuntimeError)
        (directory / "unsafe_link").unlink()
        assert (outside / "keep").read_text() == "protected"
        # Only the exact SDK log link is accepted. Never follow its shared target.
        run_logs = directory / "logs"
        run_logs.mkdir()
        debug_link = run_logs / "debug-core.log"
        shared_log = outside / "shared-debug.log"
        shared_log.write_text("shared service log")
        debug_link.symlink_to(shared_log)
        before_link = cleaner.snapshot(directory)
        shared_log.write_text("shared service log changed by another job")
        assert cleaner.unchanged(before_link, cleaner.snapshot(directory))
        debug_link.unlink()
        debug_link.symlink_to(outside / "missing-debug.log")
        dangling = cleaner.snapshot(directory)
        assert "logs/debug-core.log" in dangling
        assert not cleaner.unchanged(before_link, dangling)
        debug_link.unlink()
        debug_link.symlink_to(shared_log)
        def changed_link(*unused, **kwargs):
            debug_link.unlink()
            debug_link.symlink_to(outside / "missing-debug.log")
        expect_retained(directory, args, api, RuntimeError, changed_link)
        debug_link.unlink()
        run_logs.rmdir()
        run_logs.symlink_to(outside, target_is_directory=True)
        expect_retained(directory, args, api, RuntimeError)
        run_logs.unlink()
        run_logs.mkdir()
        debug_link.symlink_to(shared_log)
        unsafe_media = directory / "files/media/videos/unsafe.mp4"
        unsafe_media.symlink_to(shared_log)
        expect_retained(directory, args, api, RuntimeError)
        unsafe_media.unlink()
        # Sync rewrites known SDK metadata, but payload changes remain forbidden.
        def sdk_metadata(*unused, **kwargs):
            (directory / "files/config.yaml").write_text("wandb_version: 1")
            (directory / "files/wandb-summary.json").write_text("{}")
        original_open = Path.open
        def audit_failure(path, *positional, **kwargs):
            if path.name == "completed_cleanup_audit.jsonl":
                raise OSError("synthetic full disk")
            return original_open(path, *positional, **kwargs)
        with patch.object(Path, "open", audit_failure):
            expect_retained(directory, args, api, OSError, sdk_metadata)
        with patch.object(cleaner, "live_jobs", return_value=set()), \
             patch.object(cleaner, "job_finished", return_value=True), \
             patch.object(cleaner.subprocess, "run", side_effect=sdk_metadata) as upload:
            freed = cleaner.process(directory, {"123"}, args, lambda: api)
            assert freed > 0 and not directory.exists()
            assert upload.call_args.kwargs["check"] is True
        assert sacred.is_file() and log.is_file() and open_dir.is_dir()
        assert shared_log.read_text() == "shared service log changed by another job"
        audit = json.loads((root / "completed_cleanup_audit.jsonl").read_text())
        assert audit["cloud"] == "hjh331-sjtu/gomarl/abcdefgh"
        assert audit["history_points"] == 2 and audit["verified_files"] == 1
        with patch.object(cleaner, "command", return_value="123|COMPLETED|" + str(args.repo)):
            assert cleaner.job_finished("123", args.repo)
        for record in ("123|RUNNING|" + str(args.repo), "123|COMPLETED|/another/repo", ""):
            with patch.object(cleaner, "command", return_value=record):
                assert not cleaner.job_finished("123", args.repo)
        real_sdk_check(runtime)
    print("PASS: synthetic and real core SDK runs; exact terminal job proof, active/unknown/requeued protection, preview, failed sync/API/history/media verification retains; SDK debug link/dangling link accepted without following target, shared log preserved, replaced/non-SDK links blocked; audit failure retains, only verified W&B copy removed; Sacred/job logs kept")


if __name__ == "__main__":
    main()
