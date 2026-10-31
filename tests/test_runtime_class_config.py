"""Regression tests for nested runtime class discovery."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from PIL import Image

import config
from tests.conftest import workspace_tmp_path


_TEST_ROOT = workspace_tmp_path("runtime_classes")


@contextmanager
def _dataset_dir() -> Iterator[Path]:
    root = _TEST_ROOT / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)


def test_nested_class_detection() -> None:
    with _dataset_dir() as root:
        _image(root / "living" / "animals" / "cat" / "a.png")
        _image(root / "living" / "plants" / "tree" / "b.png")
        _image(root / "non-living" / "rock" / "c.png")
        assert config.get_class_names(root) == ["cat", "tree", "rock"]
        assert config.get_num_classes(root) == 3


def test_missing_dataset_falls_back() -> None:
    assert config.get_class_names(_TEST_ROOT / "missing", fallback=["a", "b"]) == ["a", "b"]
