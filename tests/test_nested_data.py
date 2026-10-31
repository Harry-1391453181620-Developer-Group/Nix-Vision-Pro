"""Tests for nested dataset discovery and stratified splitting."""

from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest

from data.nested import discover_nested_dataset, stratified_split
from tests.conftest import workspace_tmp_path


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)


def _dataset_root() -> Path:
    root = workspace_tmp_path("nested_dataset") / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_discover_nested_dataset_builds_tree() -> None:
    root = _dataset_root()
    try:
        _make_image(root / "living" / "animals" / "cat" / "a.png")
        _make_image(root / "living" / "animals" / "dog" / "b.png")
        _make_image(root / "non-living" / "rock" / "c.png")
        index = discover_nested_dataset(root)
        assert index.class_names == ("cat", "dog", "rock")
        assert len(index) == 3
        assert index.tree.num_leaves == 3
        assert index.class_counts.tolist() == [1, 1, 1]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_ambiguous_leaf_and_internal_directory_is_rejected() -> None:
    root = _dataset_root()
    try:
        _make_image(root / "animal" / "cat" / "a.png")
        _make_image(root / "animal" / "b.png")
        with pytest.raises(ValueError, match="both images and nested"):
            discover_nested_dataset(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stratified_split_keeps_each_class_in_validation() -> None:
    labels = np.array([0] * 10 + [1] * 5 + [2] * 2, dtype=np.int64)
    train, val = stratified_split(labels, 0.2, seed=7)
    assert set(labels[val].tolist()) == {0, 1, 2}
    assert set(labels[train].tolist()) == {0, 1, 2}
    assert len(train) + len(val) == labels.size
