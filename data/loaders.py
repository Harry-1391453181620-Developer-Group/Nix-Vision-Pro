"""
Load images from disk using Pillow and return as np.ndarray (H, W, 3).
Memory-aware and transparency-safe.
"""

from pathlib import Path
from typing import List, Tuple, Union

import os
import math
import numpy as np
from PIL import Image

# Defaults can be overridden via env vars to tune memory behavior
_MAX_SIDE = int(os.environ.get("CNN_MAX_SIDE", "640"))            # hard cap on longer side
_MAX_IMAGE_MB = float(os.environ.get("CNN_MAX_IMAGE_MB", "8.0"))  # max bytes for float64 array in MiB
_MIN_SIDE = int(os.environ.get("CNN_MIN_SIDE", "256"))            # do not shrink below this


def _to_rgb_safe(img: Image.Image) -> Image.Image:
    """Convert to RGB while handling palette/alpha without PIL warnings.
    - P images with byte transparency -> convert to RGBA then composite
    - LA/RGBA -> composite on solid background
    - else -> direct RGB
    """
    if img.mode == "P" and "transparency" in getattr(img, "info", {}):
        img = img.convert("RGBA")
    if img.mode in ("LA", "RGBA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.getchannel("A")
        bg.paste(img.convert("RGB"), mask=alpha)
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _downscale_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return img


def _downscale_for_memory(img: Image.Image, max_image_mb: float, min_side: int) -> Image.Image:
    """Ensure float64 array size (H*W*3*8) MiB <= max_image_mb by shrinking as needed."""
    w, h = img.size
    mb = (w * h * 3 * 8) / (1024 * 1024)
    if mb <= max_image_mb:
        return img
    # Target area so that float64 bytes ~= max_image_mb
    target_area = (max_image_mb * 1024 * 1024) / (3 * 8)
    cur_area = w * h
    scale = math.sqrt(target_area / cur_area)
    # Respect min_side guard
    new_w = max(min_side, int(w * scale))
    new_h = max(min_side, int(h * scale))
    if new_w < w or new_h < h:
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return img


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load a single image from disk.

    Args:
        path: Path to the image file (e.g. JPEG, PNG).

    Returns:
        ndarray of shape (H, W, 3), dtype float64, values in [0, 255].
        Channel order R, G, B. Grayscale images are converted to RGB by repeating.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    img = _to_rgb_safe(img)
    # First, bound by side length; then reduce further if float64 conversion would exceed cap
    img = _downscale_if_needed(img, _MAX_SIDE)
    img = _downscale_for_memory(img, _MAX_IMAGE_MB, _MIN_SIDE)
    # Keep test-facing API: float64 in [0,255]
    arr = np.array(img, dtype=np.float64)
    return arr


def load_images_from_dir(
    dir_path: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> Tuple[List[np.ndarray], List[Path]]:
    """
    Load all images from a directory.

    Args:
        dir_path: Directory containing image files.
        extensions: File extensions to include (case-insensitive).

    Returns:
        Tuple of (list of ndarrays (H, W, 3), list of file paths).
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    exts = {e.lower() for e in extensions}
    images: List[np.ndarray] = []
    paths: List[Path] = []
    for p in sorted(dir_path.iterdir()):
        if p.suffix.lower() in exts:
            images.append(load_image(p))
            paths.append(p)
    return images, paths
