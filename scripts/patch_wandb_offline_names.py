#!/usr/bin/env python3
"""Copy a W&B offline run and shorten upload-only run/group names.

This is intentionally conservative: it never edits the original offline run.
It patches a copied run directory before `wandb sync`, mainly to remove
experiment-origin suffixes such as `_rerun` and `_ozstar` from W&B group names.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


DROP_PARTS = (
    "_rerun",
    "_ozstar",
    "rerun_",
    "ozstar_",
)

COMPACT_PARTS = (
    ("3s5z_vs_3s6z", "3s5zv3s6z"),
    ("5m_vs_6m", "5m6m"),
    ("public_private", "pubpriv"),
    ("transformer", "tr"),
    ("relation", "rel"),
    ("interaction", "int"),
    ("decision", "dec"),
    ("private", "priv"),
    ("public", "pub"),
    ("hypercond", "hyp"),
)


def _clean_underscores(text):
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def shorten_name(text, max_len):
    if not text:
        return text

    shortened = str(text)
    for part in DROP_PARTS:
        shortened = shortened.replace(part, "")
    shortened = _clean_underscores(shortened)
    if len(shortened) <= max_len:
        return shortened

    for old, new in COMPACT_PARTS:
        shortened = shortened.replace(old, new)
    shortened = _clean_underscores(shortened)
    if len(shortened) <= max_len:
        return shortened

    digest = hashlib.sha1(shortened.encode("utf8")).hexdigest()[:8]
    keep = max_len - len(digest) - 1
    return "{}_{}".format(shortened[:keep].rstrip("_"), digest)


def _record_is_run(record):
    try:
        return record.HasField("run")
    except Exception:
        try:
            return record.WhichOneof("record_type") == "run"
        except Exception:
            return False


def patch_wandb_file(path, max_group_len, max_name_len):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    tmp_path = path.with_suffix(path.suffix + ".patched")
    if tmp_path.exists():
        tmp_path.unlink()

    scanner = DataStore()
    writer = DataStore()
    changed = []

    scanner.open_for_scan(str(path))
    writer.open_for_write(str(tmp_path))
    try:
        while True:
            data = scanner.scan_data()
            if data is None:
                break

            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)

            if _record_is_run(record):
                run = record.run
                for field_name, max_len in (
                    ("run_group", max_group_len),
                    ("display_name", max_name_len),
                    ("job_type", max_group_len),
                ):
                    if not hasattr(run, field_name):
                        continue
                    old = getattr(run, field_name)
                    if not old:
                        continue
                    new = shorten_name(old, max_len)
                    if new != old:
                        setattr(run, field_name, new)
                        changed.append((field_name, old, new))

            writer.write(record)
    finally:
        if hasattr(scanner, "close"):
            scanner.close()
        if hasattr(writer, "close"):
            writer.close()

    tmp_path.replace(path)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Original wandb/offline-run-* directory.")
    parser.add_argument("output_root", help="Directory for patched sync copies.")
    parser.add_argument("--max-group-len", type=int, default=120)
    parser.add_argument("--max-name-len", type=int, default=180)
    args = parser.parse_args()

    src = Path(args.run_dir).resolve()
    out_root = Path(args.output_root).resolve()
    if not src.is_dir():
        raise SystemExit("not a directory: {}".format(src))

    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)

    run_files = list(dst.glob("run-*.wandb"))
    if len(run_files) != 1:
        raise SystemExit("expected one run-*.wandb in {}, found {}".format(dst, len(run_files)))

    changed = patch_wandb_file(run_files[0], args.max_group_len, args.max_name_len)
    for field_name, old, new in changed:
        print("patched {}: {} -> {}".format(field_name, old, new), file=sys.stderr)

    print(dst)


if __name__ == "__main__":
    main()
