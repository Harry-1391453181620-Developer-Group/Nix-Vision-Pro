"""Regression tests for runtime class discovery from the active dataset layout."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import config


_TEST_TMP_ROOT = Path('.pytest-tmp') / 'runtime_case_dirs'
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def _dataset_dir() -> Iterator[Path]:
    """Create a temporary dataset directory inside the workspace and clean it up afterward."""
    dataset_root = _TEST_TMP_ROOT / uuid4().hex
    dataset_root.mkdir(parents=True, exist_ok=False)
    try:
        yield dataset_root
    finally:
        shutil.rmtree(dataset_root, ignore_errors=True)


def _touch_image(path: Path) -> None:
    """Create a tiny placeholder image file for suffix-based dataset detection tests."""
    path.write_bytes(b'test')


def test_get_class_names_detects_only_non_empty_class_dirs() -> None:
    with _dataset_dir() as dataset_root:
        for class_name in ('ship', 'cat'):
            class_dir = dataset_root / class_name
            class_dir.mkdir()
            _touch_image(class_dir / 'sample.jpg')
        (dataset_root / 'empty_class').mkdir()

        class_names = config.get_class_names(dataset_root, require_images=True)

        assert class_names == ['cat', 'ship']
        assert config.get_num_classes(dataset_root, require_images=True) == 2


def test_get_class_names_pads_when_class_count_is_larger() -> None:
    with _dataset_dir() as dataset_root:
        for class_name in ('bird', 'dog'):
            class_dir = dataset_root / class_name
            class_dir.mkdir()
            _touch_image(class_dir / 'sample.jpg')

        class_names = config.get_class_names(dataset_root, class_count=4, require_images=True)

        assert class_names == ['bird', 'dog', '2', '3']
        assert config.get_num_classes(dataset_root, class_count=4, require_images=True) == 4


def test_get_class_names_falls_back_when_dataset_root_is_missing() -> None:
    class_names = config.get_class_names(Path('.pytest-tmp') / 'missing_dataset_root')
    assert class_names == config.DEFAULT_CLASS_NAMES
