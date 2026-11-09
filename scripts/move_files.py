"""
Move all files from a source folder to a destination folder (e.g. a network
drive), keeping the folder structure.

- Network errors (e.g. "An unexpected network error occurred") are retried
  for the file that failed. Files that already moved are not redone.
- A file that already exists at the destination is skipped and stays in
  the source folder. With --delete-duplicates, the source copy is deleted
  when the destination file has the same size.
- Each file is copied to a temporary "<name>.partial" file first, then renamed
  once its size checks out. The source is deleted only after that. A dropped
  connection therefore never leaves a half-written file under the real name,
  which would otherwise be skipped as "already exists" on the next run.

Usage:
    python move_files.py SOURCE DESTINATION [--retries 15] [--delay 5]
                         [--delete-duplicates] [--keep-empty-dirs] [--dry-run]
"""

import argparse
import errno
import os
import shutil
import sys
import time
from pathlib import Path

PARTIAL_SUFFIX = ".partial"

# Windows error codes that mean the network share or connection went away
NETWORK_WINERRORS = {
    53,    # The network path was not found
    59,    # An unexpected network error occurred
    64,    # The specified network name is no longer available
    67,    # The network name cannot be found
    121,   # The semaphore timeout period has expired
    1231,  # The network location cannot be reached
    1232,  # The network location cannot be reached
}
NETWORK_ERRNOS = {
    errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.ENETDOWN,
    errno.ECONNRESET, errno.ECONNABORTED, getattr(errno, "ESTALE", -1),
}


def is_network_error(exc: BaseException) -> bool:
    """Return True if the error looks like a dropped network connection."""
    if isinstance(exc, shutil.Error):
        # shutil.Error carries a list of (src, dst, message) tuples
        return "network" in str(exc).lower()
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in NETWORK_WINERRORS:
            return True
        if exc.errno in NETWORK_ERRNOS:
            return True
    return "network" in str(exc).lower()


def with_retry(action, description, retries, delay):
    """Run action(); retry on network errors, re-raise anything else."""
    attempt = 0
    while True:
        try:
            return action()
        except (OSError, shutil.Error) as e:
            if not is_network_error(e):
                raise
            attempt += 1
            if attempt > retries:
                raise
            print(f"  Network error on {description}: {e}\n"
                  f"  Retry {attempt}/{retries} in {delay} s...")
            time.sleep(delay)


def move_one_file(src: Path, dst: Path, delete_duplicates: bool, dry_run: bool) -> str:
    """Move a single file. Returns 'moved', 'skipped' or 'duplicate_deleted'."""
    if dst.exists():
        if delete_duplicates and src.stat().st_size == dst.stat().st_size:
            if not dry_run:
                src.unlink()
            return "duplicate_deleted"
        return "skipped"

    if dry_run:
        return "moved"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + PARTIAL_SUFFIX)
    if tmp.exists():          # left over from an interrupted attempt
        tmp.unlink()

    shutil.copy2(src, tmp)    # copy data + timestamps

    src_size, tmp_size = src.stat().st_size, tmp.stat().st_size
    if src_size != tmp_size:
        tmp.unlink()
        raise OSError(errno.EIO, f"network copy incomplete ({tmp_size} of {src_size} bytes)")

    os.replace(tmp, dst)      # atomic rename on the same drive
    src.unlink()              # only now remove the original
    return "moved"


def move_folder(source, destination, retries=15, delay=5,
                delete_duplicates=False, keep_empty_dirs=False, dry_run=False) -> bool:
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        print(f"Source folder '{source}' does not exist.")
        return False

    counts = {"moved": 0, "skipped": 0, "duplicate_deleted": 0, "failed": 0}

    for src_dir, _subdirs, files in os.walk(source):
        rel = Path(src_dir).relative_to(source)
        dest_dir = destination / rel

        if not dry_run:
            try:
                with_retry(lambda: dest_dir.mkdir(parents=True, exist_ok=True),
                           f"folder '{dest_dir}'", retries, delay)
            except (OSError, shutil.Error) as e:
                print(f"Could not create '{dest_dir}': {e}. Skipping its files.")
                counts["failed"] += len(files)
                continue

        for name in files:
            if name.endswith(PARTIAL_SUFFIX):
                continue
            src_file = Path(src_dir) / name
            dst_file = dest_dir / name
            label = str(rel / name)
            try:
                result = with_retry(
                    lambda: move_one_file(src_file, dst_file, delete_duplicates, dry_run),
                    f"'{label}'", retries, delay)
            except (OSError, shutil.Error) as e:
                print(f"FAILED  {label}: {e}")
                counts["failed"] += 1
                continue

            counts[result] += 1
            msg = {"moved": "Moved  ",
                   "skipped": "Skipped (exists at destination)",
                   "duplicate_deleted": "Exists at destination, same size: source deleted"}[result]
            print(f"{msg}  {label}" if result == "moved" else f"{msg}: {label}")

    # Remove source folders that are now empty (deepest first)
    if not keep_empty_dirs and not dry_run:
        for src_dir, _subdirs, _files in os.walk(source, topdown=False):
            p = Path(src_dir)
            if p != source and not any(p.iterdir()):
                p.rmdir()

    print(f"\nDone. Moved: {counts['moved']}, skipped: {counts['skipped']}, "
          f"duplicates deleted: {counts['duplicate_deleted']}, failed: {counts['failed']}")
    return counts["failed"] == 0


def main():
    ap = argparse.ArgumentParser(description="Move files to another folder, keeping structure, "
                                             "retrying on network errors.")
    ap.add_argument("source", help="folder to move files from")
    ap.add_argument("destination", help="folder to move files into (created if missing)")
    ap.add_argument("--retries", type=int, default=15, help="retries per file on network error (default 15)")
    ap.add_argument("--delay", type=float, default=5, help="seconds between retries (default 5)")
    ap.add_argument("--delete-duplicates", action="store_true",
                    help="delete source file if an identical-size file already exists at destination")
    ap.add_argument("--keep-empty-dirs", action="store_true",
                    help="don't remove source folders left empty after the move")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen without changing anything")
    args = ap.parse_args()

    ok = move_folder(args.source, args.destination, args.retries, args.delay,
                     args.delete_duplicates, args.keep_empty_dirs, args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
