"""Default configuration plus runtime helpers for dataset-driven class metadata."""

import os
from pathlib import Path
from typing import List, Sequence, Tuple

# Input image size (H, W) after preprocessing.
INPUT_SIZE: Tuple[int, int] = (32, 32)

# Default class names used when no dataset folders can be detected yet.
DEFAULT_CLASS_NAMES: List[str] = [
    "airplane",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
]

# Supported image suffixes for quick dataset inspection.
IMAGE_FILE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Data paths (relative to project root or absolute).
DATA_DIR: Path = Path("Dataset")
CHECKPOINT_DIR: Path = Path("checkpoints")


def _normalize_class_names(class_names: Sequence[str]) -> List[str]:
    """
    Convert any incoming class-name sequence into a clean list of strings.

    Empty values are ignored so malformed fallback data cannot leak blank labels
    into runtime model configuration.
    """
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
    """
    Detect active class names from immediate subdirectories under a dataset root.

    When `require_images` is True, empty directories are skipped so training only
    counts classes that actually have image files. When no usable directories can
    be found, the function falls back to the provided class list or the built-in
    project defaults so first-time dataset generation still works.
    """
    dataset_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    fallback_names = _normalize_class_names(DEFAULT_CLASS_NAMES if fallback is None else fallback)
    try:
        class_dirs = sorted(entry for entry in dataset_dir.iterdir() if entry.is_dir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return list(fallback_names)

    detected: List[str] = []
    for class_dir in class_dirs:
        if require_images:
            try:
                has_images = any(
                    entry.is_file() and entry.suffix.lower() in IMAGE_FILE_SUFFIXES
                    for entry in class_dir.iterdir()
                )
            except PermissionError:
                has_images = False
            if not has_images:
                continue
        detected.append(class_dir.name)

    return detected or list(fallback_names)


def get_class_names(
    data_dir: str | Path | None = None,
    *,
    class_count: int | None = None,
    require_images: bool = False,
    fallback: Sequence[str] | None = None,
) -> List[str]:
    """
    Resolve runtime class names for a command or UI session.

    If `class_count` is larger than the detected dataset classes, the returned
    list is padded with numeric placeholder labels so callers can keep model
    dimensions aligned with an explicit `--class-count` override.
    """
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
    """
    Resolve labels for inference when checkpoint metadata may override the dataset.

    Preference order:
    1. checkpoint-provided class names when they match `num_classes`
    2. dataset-derived class names when they match `num_classes`
    3. numeric placeholders
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be > 0")

    preferred = _normalize_class_names(checkpoint_class_names or ())
    if len(preferred) == num_classes:
        return preferred

    detected = detect_class_names(
        data_dir,
        require_images=require_images,
        fallback=(),
    )
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
    """
    Resolve the runtime number of classes from the dataset or an explicit override.
    """
    return len(
        get_class_names(
            data_dir,
            class_count=class_count,
            require_images=require_images,
            fallback=fallback,
        )
    )


# Runtime class metadata for the default Dataset directory.
CLASS_NAMES: List[str] = get_class_names()
NUM_CLASSES: int = len(CLASS_NAMES)

# Training defaults.
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
NUM_EPOCHS: int = 100

# Normalization: input value range (e.g. [0, 255]) and output range (e.g. [0, 1]).
_OUTPUT_DTYPE = os.environ.get("CNN_OUTPUT_DTYPE", "").lower()
if _OUTPUT_DTYPE == "int8":
    INPUT_VALUE_RANGE: Tuple[float, float] = (-128.0, 127.0)
else:
    INPUT_VALUE_RANGE: Tuple[float, float] = (0.0, 255.0)
NORMALIZE_TO: Tuple[float, float] = (0.0, 1.0)
