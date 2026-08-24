"""Small atomic-file primitives shared by label_images metadata stores."""

import os
import tempfile
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for directory entry changes.

    Some platforms/filesystems do not allow opening or syncing directories.  Those
    cases are deliberately ignored; failures from a directory that was opened are
    propagated because they mean durability could not be established.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path: Path) -> None:
    """Create missing levels and sync every newly linked directory entry."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        fsync_directory(directory)
        fsync_directory(directory.parent)


def atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes through a same-directory temporary file and atomic replace."""
    durable_mkdir(path.parent)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        tmp_path.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def restore_file(path: Path, snapshot: bytes | None) -> None:
    """Restore exact previous bytes, or remove a file that did not exist."""
    if snapshot is None:
        existed = path.exists()
        path.unlink(missing_ok=True)
        if existed:
            fsync_directory(path.parent)
        return
    atomic_write(path, snapshot)
