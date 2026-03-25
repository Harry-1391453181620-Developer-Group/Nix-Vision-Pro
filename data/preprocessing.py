"""Preprocessing: resize, normalize, optional augmentation. NumPy only (resize may use Pillow)."""

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    use_pillow: bool = True,
) -> np.ndarray:
    """
    Resize image to a fixed (height, width).

    Args:
        image: ndarray (H, W, 3), any dtype.
        target_size: (height, width) to resize to.
        use_pillow: If True, use Pillow for resize (recommended). If False, use simple NumPy indexing (nearest).

    Returns:
        ndarray of shape target_size + (3,), same dtype as input.
    """
    h, w = target_size
    if use_pillow:
        pil = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
        pil = pil.resize((w, h), Image.BILINEAR)
        return np.array(pil, dtype=image.dtype)
    # Simple NumPy: scale indices and take nearest neighbor
    H, W = image.shape[:2]
    y_idx = np.linspace(0, H - 1, h).astype(np.intp)
    x_idx = np.linspace(0, W - 1, w).astype(np.intp)
    return image[np.ix_(y_idx, x_idx)]


def normalize(
    image: np.ndarray,
    scale: Tuple[float, float] = (0.0, 1.0),
    input_range: Tuple[float, float] = (0.0, 255.0),
) -> np.ndarray:
    """
    Normalize image values to a range (default [0, 1]).

    Args:
        image: ndarray (H, W, 3) or (N, H, W, 3).
        scale: (min, max) output range.
        input_range: (min, max) assumed input range.

    Returns:
        ndarray same shape, float64, values in scale.
    """
    lo, hi = input_range
    out_lo, out_hi = scale
    out = np.asarray(image, dtype=np.float64)
    out = (out - lo) / (hi - lo)
    out = out * (out_hi - out_lo) + out_lo
    return out


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    normalize_to: Tuple[float, float] = (0.0, 1.0),
    input_value_range: Tuple[float, float] = (0.0, 255.0),
) -> np.ndarray:
    """
    Full preprocessing: resize then normalize.

    Args:
        image: ndarray (H, W, 3).
        target_size: (height, width).
        normalize_to: output value range.
        input_value_range: assumed input value range.

    Returns:
        ndarray (target_H, target_W, 3), float64, in normalize_to range.
    """
    out = resize(image, target_size)
    out = normalize(out, scale=normalize_to, input_range=input_value_range)
    return out


def batch_preprocess(
    images: list[np.ndarray],
    target_size: Tuple[int, int],
    normalize_to: Tuple[float, float] = (0.0, 1.0),
    input_value_range: Tuple[float, float] = (0.0, 255.0),
) -> np.ndarray:
    """
    Preprocess a list of images and stack into a batch.

    Returns:
        ndarray (N, H, W, 3), float64.
    """
    out = [preprocess_image(im, target_size, normalize_to, input_value_range) for im in images]
    return np.stack(out, axis=0)
