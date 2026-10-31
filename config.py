"""Runtime configuration and nested-dataset class discovery for MAOL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

# Input image size (H, W) after preprocessing.
INPUT_SIZE: Tuple[int, int] = (224, 224)

# Fallback labels only used when no dataset can be discovered yet.
DEFAULT_CLASS_NAMES: List[str] = [
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "man",
    "mouse",
    "daisy",
    "maple-tree",
    "mushroom",
    "airplane",
    "basketball",
    "coffee-mug",
    "drum",
    "motorcycle",
    "mountain",
    "seashore",
]

IMAGE_FILE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DATA_DIR: Path = Path("Dataset")
CHECKPOINT_DIR: Path = Path("checkpoints")


def _normalize_class_names(class_names: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for class_name in class_names:
        text = str(class_name).strip()
        if text:
            normalized.append(text)
    return normalized


def detect_class_names(
    data_dir: str | Path | None = None,
    *,
    require_images: bool = False,
    fallback: Sequence[str] | None = None,
) -> List[str]:
    """Return leaf class names from the nested dataset layout.

    ``require_images`` is retained for callers that want a no-image fallback; the
    nested discovery itself only reports directories that actually contain images.
    """
    del require_images
    dataset_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    fallback_names = _normalize_class_names(DEFAULT_CLASS_NAMES if fallback is None else fallback)
    try:
        from data.nested import discover_nested_dataset

        index = discover_nested_dataset(dataset_dir)
        return list(index.class_names)
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError):
        return list(fallback_names)


def get_class_names(
    data_dir: str | Path | None = None,
    *,
    class_count: int | None = None,
    require_images: bool = False,
    fallback: Sequence[str] | None = None,
) -> List[str]:
    class_names = detect_class_names(data_dir, require_images=require_images, fallback=fallback)
    if class_count is None:
        return class_names
    if class_count <= 0:
        raise ValueError("class_count must be > 0")
    if class_count < len(class_names):
        raise ValueError(f"class_count={class_count} is smaller than detected classes={len(class_names)}")
    resolved = list(class_names)
    while len(resolved) < class_count:
        resolved.append(str(len(resolved)))
    return resolved


def resolve_runtime_class_names(
    data_dir: str | Path | None,
    *,
    num_classes: int,
    checkpoint_class_names: Sequence[str] | None = None,
    require_images: bool = False,
) -> List[str]:
    if num_classes <= 0:
        raise ValueError("num_classes must be > 0")
    preferred = _normalize_class_names(checkpoint_class_names or ())
    if len(preferred) == num_classes:
        return preferred
    detected = detect_class_names(data_dir, require_images=require_images, fallback=())
    if len(detected) == num_classes:
        return detected
    return [str(index) for index in range(num_classes)]


def get_num_classes(
    data_dir: str | Path | None = None,
    *,
    class_count: int | None = None,
    require_images: bool = False,
    fallback: Sequence[str] | None = None,
) -> int:
    return len(
        get_class_names(
            data_dir,
            class_count=class_count,
            require_images=require_images,
            fallback=fallback,
        )
    )


CLASS_NAMES: List[str] = get_class_names()
NUM_CLASSES: int = len(CLASS_NAMES)

BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.0002
NUM_EPOCHS: int = 120

_OUTPUT_DTYPE = os.environ.get("CNN_OUTPUT_DTYPE", "").lower()
if _OUTPUT_DTYPE == "int8":
    INPUT_VALUE_RANGE: Tuple[float, float] = (-128.0, 127.0)
else:
    INPUT_VALUE_RANGE: Tuple[float, float] = (0.0, 255.0)
NORMALIZE_TO: Tuple[float, float] = (0.0, 1.0)
