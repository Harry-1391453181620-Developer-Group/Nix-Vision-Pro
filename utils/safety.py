"""
Safety utilities: enforce Dataset read-only at runtime for Python code.
- install_dataset_write_guard(): installs a Python audit hook to block writes/deletes/renames under Dataset/.
- tree_signature(path): returns a compact signature to detect changes to the dataset tree.

This does NOT block manual changes done outside Python (e.g., Explorer). It protects our code paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Project root and dataset path
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = (PROJ_ROOT / "Dataset").resolve()

_WRITE_FLAGS = ("w", "a", "x", "+")


def _is_in_dataset(p: Path) -> bool:
    try:
        rp = Path(p).resolve()
    except Exception:
        return False
    try:
        rp.relative_to(DATASET_DIR)
        return True
    except Exception:
        return False


def install_dataset_write_guard(env_var: str = "DATASET_WRITE_GUARD") -> None:
    """Install an audit hook that blocks writes/deletes/renames inside Dataset/.
    Set env DATASET_WRITE_GUARD=0 to disable (not recommended).
    """
    if os.environ.get(env_var, "1") == "0":
        return

    def _audit(event: str, args: tuple) -> None:
        # Block opening Dataset files in write/append/create modes
        if event == "open":
            if not args:
                return
            filename = args[0]
            mode = args[1] if len(args) > 1 else "r"
            if isinstance(filename, (str, bytes, os.PathLike)) and any(f in str(mode) for f in _WRITE_FLAGS):
                if _is_in_dataset(Path(filename)):
                    raise PermissionError("Dataset is read-only: refused to open for writing: %r" % (filename,))
        # Block filesystem mutations under Dataset
        elif event in {"os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace"}:
            for a in args:
                if isinstance(a, (str, bytes, os.PathLike)) and _is_in_dataset(Path(a)):
                    raise PermissionError(f"Dataset is read-only: refused {event} on {a!r}")

    # Avoid multiple registrations
    if not getattr(install_dataset_write_guard, "_installed", False):
        sys.addaudithook(_audit)
        setattr(install_dataset_write_guard, "_installed", True)


def tree_signature(root: Path) -> Tuple[int, int, int, str]:
    """Compute a cheap signature of a directory tree.
    Returns (num_files, total_bytes, num_dirs, digest_hex).
    """
    import hashlib

    root = Path(root).resolve()
    hasher = hashlib.md5()
    num_files = 0
    num_bytes = 0
    num_dirs = 0
    for dirpath, dirnames, filenames in os.walk(root):
        drel = os.path.relpath(dirpath, root)
        num_dirs += 1
        hasher.update(drel.encode("utf-8", errors="ignore"))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except FileNotFoundError:
                # changed concurrently; treat as difference
                st = None
            rel = os.path.relpath(p, root)
            hasher.update(rel.encode("utf-8", errors="ignore"))
            if st:
                num_files += 1
                num_bytes += int(getattr(st, "st_size", 0))
                hasher.update(str(int(getattr(st, "st_mtime", 0))).encode())
                hasher.update(str(int(getattr(st, "st_size", 0))).encode())
    return num_files, num_bytes, num_dirs, hasher.hexdigest()
