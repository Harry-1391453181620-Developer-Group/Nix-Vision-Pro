"""Nested image-folder discovery for MAOL.

A leaf class is a directory that directly contains images and no nested class
directories.  Every ancestor directory is an internal MAOL well.  This keeps the
tree encoded by the filesystem itself, as required by the theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from maol.tree import ClassTree

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SKIP_DIRS = {"__pycache__", ".git", ".stfolder", ".tmp", "logs"}


@dataclass(frozen=True)
class DatasetIndex:
    root: Path
    paths: tuple[Path, ...]
    labels: np.ndarray
    class_names: tuple[str, ...]
    tree: ClassTree
    class_counts: np.ndarray
    node_weights: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _images_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_nested_dataset(root: str | Path) -> DatasetIndex:
    """Walk ``root`` and return leaf classes, labels, and the encoded tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    leaf_paths: list[str] = []
    leaf_files: dict[str, list[Path]] = {}

    def walk(directory: Path, relative_parts: tuple[str, ...]) -> None:
        subdirs = sorted(
            child
            for child in directory.iterdir()
            if child.is_dir() and child.name not in _SKIP_DIRS
        )
        images = _images_in(directory)
        if images and subdirs:
            raise ValueError(
                f"Directory {directory} contains both images and nested class directories. "
                "Keep each class in a directory with no child class directories."
            )
        if images:
            if not relative_parts:
                raise ValueError(
                    "Dataset root cannot contain images directly; place each class in a folder."
                )
            relative_path = "/".join(relative_parts)
            leaf_paths.append(relative_path)
            leaf_files[relative_path] = images
            return
        for child in subdirs:
            walk(child, relative_parts + (child.name,))

    walk(root, ())
    if not leaf_paths:
        raise ValueError(f"No leaf class directories containing images were found under {root}")

    tree = ClassTree.from_leaf_paths(leaf_paths)
    leaf_path_set = set(leaf_paths)
    paths: list[Path] = []
    labels: list[int] = []
    for node in tree.nodes:
        if node.path not in leaf_path_set:
            continue
        class_index = node.leaf_index
        if class_index is None:
            raise RuntimeError(f"Tree node {node.path!r} is not marked as a leaf")
        files = leaf_files[node.path]
        paths.extend(files)
        labels.extend([int(class_index)] * len(files))

    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=tree.num_leaves).astype(np.int64)
    node_weights = np.asarray(tree.node_weights_from_labels(labels_array.tolist()), dtype=np.float64)
    return DatasetIndex(
        root=root,
        paths=tuple(paths),
        labels=labels_array,
        class_names=tree.class_names,
        tree=tree,
        class_counts=counts,
        node_weights=node_weights,
    )


def stratified_split(
    labels: np.ndarray,
    val_split: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by class so every class with at least two samples appears in validation."""
    labels = np.asarray(labels, dtype=np.int64)
    if not (0.0 <= float(val_split) < 1.0):
        raise ValueError("val_split must satisfy 0 <= value < 1")
    if float(val_split) <= 0.0 or labels.size < 2:
        order = np.random.default_rng(seed).permutation(labels.size)
        return order.astype(np.int64), np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for class_index in range(int(labels.max()) + 1):
        indices = np.flatnonzero(labels == class_index)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        if indices.size == 1:
            train_parts.append(indices)
            continue
        requested = int(round(indices.size * float(val_split)))
        val_count = min(max(1, requested), indices.size - 1)
        val_parts.append(indices[:val_count])
        train_parts.append(indices[val_count:])

    train = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.empty((0,), dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    return train.astype(np.int64), val.astype(np.int64)


def split_dataset(
    index: DatasetIndex,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
) -> tuple[list[Path], np.ndarray, list[Path], np.ndarray]:
    """Materialize path/label lists for the trainer."""
    train_idx = np.asarray(list(train_indices), dtype=np.int64)
    val_idx = np.asarray(list(val_indices), dtype=np.int64)
    train_paths = [index.paths[int(i)] for i in train_idx]
    val_paths = [index.paths[int(i)] for i in val_idx]
    return (
        train_paths,
        index.labels[train_idx],
        val_paths,
        index.labels[val_idx],
    )


__all__ = [
    "DatasetIndex",
    "IMAGE_SUFFIXES",
    "discover_nested_dataset",
    "split_dataset",
    "stratified_split",
]
