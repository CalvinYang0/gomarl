#!/usr/bin/env python3
"""Copy a W&B offline run and patch upload-only W&B metadata.

It never edits the original offline run. In addition to shortening names, it
can assign a new run id to a copied, completed offline run. This is useful when
an earlier live sync created a partial remote run that W&B already marked as
finished: the complete local record can then be uploaded as a new run.
"""

import argparse
import hashlib
import secrets
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


def patch_wandb_file(path, max_group_len, max_name_len, new_run_id=None):
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
                if new_run_id is not None:
                    if not hasattr(run, "run_id"):
                        raise RuntimeError("W&B RunRecord has no run_id field")
                    old_run_id = run.run_id
                    if old_run_id != new_run_id:
                        run.run_id = new_run_id
                        changed.append(("run_id", old_run_id, new_run_id))
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


def generate_run_id():
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Original wandb/offline-run-* directory.")
    parser.add_argument("output_root", help="Directory for patched sync copies.")
    parser.add_argument("--max-group-len", type=int, default=120)
    parser.add_argument("--max-name-len", type=int, default=180)
    parser.add_argument(
        "--new-run-id",
        nargs="?",
        const="auto",
        help="Assign a fresh 8-character W&B run id to the copied run. "
        "Use without a value to generate one automatically.",
    )
    args = parser.parse_args()

    src = Path(args.run_dir).resolve()
    out_root = Path(args.output_root).resolve()
    if not src.is_dir():
        raise SystemExit("not a directory: {}".format(src))

    out_root.mkdir(parents=True, exist_ok=True)
    new_run_id = args.new_run_id
    if new_run_id == "auto":
        new_run_id = generate_run_id()
    if new_run_id is not None:
        if len(new_run_id) != 8 or not new_run_id.isalnum():
            raise SystemExit("--new-run-id must be an 8-character alphanumeric id")
        dst_name = "{}-{}".format(src.name.rsplit("-", 1)[0], new_run_id)
    else:
        dst_name = src.name

    dst = out_root / dst_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)

    run_files = list(dst.glob("run-*.wandb"))
    if len(run_files) != 1:
        raise SystemExit("expected one run-*.wandb in {}, found {}".format(dst, len(run_files)))

    if new_run_id is not None:
        renamed_run_file = run_files[0].with_name("run-{}.wandb".format(new_run_id))
        run_files[0].rename(renamed_run_file)
        run_files = [renamed_run_file]

    changed = patch_wandb_file(
        run_files[0], args.max_group_len, args.max_name_len, new_run_id=new_run_id
    )
    for field_name, old, new in changed:
        print("patched {}: {} -> {}".format(field_name, old, new), file=sys.stderr)

    print(dst)


if __name__ == "__main__":
    main()
