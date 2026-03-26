"""Shared training policies used by both NumPy and PyTorch backends.

This module centralizes the augmentation and phase-based learning-rate logic so
the two backends stay behaviorally aligned. The helpers are pure NumPy/Pillow
operations and therefore run before any backend-specific tensor conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from data.preprocessing import resize


def parse_bool_flag(value: bool | str) -> bool:
    """Parse flexible boolean CLI values such as `true`, `false`, `1`, or `0`."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


@dataclass(frozen=True)
class PhaseConfig:
    """Resolved phase metadata for one epoch."""

    phase_index: int
    epoch_index_in_phase: int
    epochs_in_phase: int


def validate_phase_learning_rates(lr_values: Sequence[float], phase_count: int) -> list[float]:
    """Validate the phase LR list before training starts."""
    values = [float(value) for value in lr_values]
    if phase_count <= 0:
        raise ValueError("phase_count must be > 0")
    if len(values) != phase_count:
        raise ValueError("The number of --lr values must equal --phase-count")
    if any(value <= 0.0 for value in values):
        raise ValueError("All learning rates must be > 0")
    for index in range(1, len(values)):
        if values[index] > values[index - 1]:
            raise ValueError("Phase learning rates must be monotonically non-increasing")
    return values


def validate_freeze_cycle_args(
    freeze_patience: int,
    freeze_epoch_num: int,
    after_unfreeze_lr_change: float,
) -> tuple[int, int, float]:
    """Validate the timed-freeze configuration before training starts.

    The LR decrement is additive, so only non-negative values are allowed. The
    actual deduction is guarded separately to ensure the resulting phase base LR
    never crosses the next phase start LR and never becomes non-positive.
    """
    freeze_patience = int(freeze_patience)
    freeze_epoch_num = int(freeze_epoch_num)
    after_unfreeze_lr_change = float(after_unfreeze_lr_change)
    if freeze_patience < 1:
        raise ValueError("freeze_patience must be >= 1")
    if freeze_epoch_num < 1:
        raise ValueError("freeze_epoch_num must be >= 1")
    if after_unfreeze_lr_change < 0.0:
        raise ValueError("after_unfreeze_lr_change must be >= 0")
    return freeze_patience, freeze_epoch_num, after_unfreeze_lr_change


def adjust_phase_base_lr_after_unfreeze(
    current_phase_base_lr: float,
    after_unfreeze_lr_change: float,
    next_phase_start_lr: float | None,
) -> tuple[float, bool]:
    """Apply the additive post-unfreeze LR deduction when it is allowed.

    Rules:
    - if the candidate LR is not strictly positive, keep the current phase LR
    - if a next phase exists and the candidate LR would go below that phase's
      configured start LR, keep the current phase LR
    - otherwise apply the deduction
    """
    current_phase_base_lr = float(current_phase_base_lr)
    after_unfreeze_lr_change = float(after_unfreeze_lr_change)
    if after_unfreeze_lr_change <= 0.0:
        return current_phase_base_lr, False

    candidate_lr = current_phase_base_lr - after_unfreeze_lr_change
    if candidate_lr <= 0.0:
        return current_phase_base_lr, False
    if next_phase_start_lr is not None and candidate_lr < float(next_phase_start_lr):
        return current_phase_base_lr, False
    return float(candidate_lr), True


def build_epoch_phase_map(epochs: int, phase_count: int) -> list[PhaseConfig]:
    """Split epochs with `np.array_split` and map each epoch to its phase metadata."""
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    phase_count = max(1, int(phase_count))
    phase_splits = np.array_split(np.arange(int(epochs)), phase_count)
    mapping: list[PhaseConfig] = [None] * int(epochs)  # type: ignore[list-item]
    for phase_index, phase_epochs in enumerate(phase_splits):
        if phase_epochs.size == 0:
            continue
        for epoch_index_in_phase, epoch_value in enumerate(phase_epochs.tolist()):
            mapping[epoch_value] = PhaseConfig(
                phase_index=phase_index,
                epoch_index_in_phase=epoch_index_in_phase,
                epochs_in_phase=int(phase_epochs.size),
            )
    if any(entry is None for entry in mapping):
        raise RuntimeError("Phase mapping failed to assign every epoch")
    return mapping


def compute_phase_learning_rate(
    *,
    base_lr: float,
    schedule: str,
    min_lr_ratio: float,
    gamma: float,
    step_size: int,
    warmup_epochs: int,
    epoch_index_in_phase: int,
    epochs_in_phase: int,
    batch_index: int,
    num_batches: int,
) -> float:
    """Compute the active LR for one batch inside a single phase.

    Warmup runs first inside each phase, then the requested schedule takes over.
    Cosine scheduling intentionally restarts from the phase base LR.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")
    base_lr = float(base_lr)
    phase_total_steps = max(1, int(epochs_in_phase) * int(num_batches))
    step_in_phase = int(epoch_index_in_phase) * int(num_batches) + int(batch_index)
    warmup_steps = max(0, min(int(warmup_epochs), int(epochs_in_phase)) * int(num_batches))

    # Warmup ramps linearly from 10% of the phase base LR up to the base LR.
    if warmup_steps > 0 and step_in_phase < warmup_steps:
        progress = (step_in_phase + 1) / warmup_steps
        return float(base_lr * (0.1 + 0.9 * progress))

    if schedule == "constant":
        return base_lr

    if schedule == "step":
        decay_index = max(0, int(epoch_index_in_phase) // max(1, int(step_size)))
        return float(base_lr * (float(gamma) ** decay_index))

    if schedule != "cosine":
        raise ValueError(f"Unsupported lr schedule: {schedule}")

    remaining_total = max(1, phase_total_steps - warmup_steps)
    remaining_step = max(0, step_in_phase - warmup_steps)
    if remaining_total <= 1:
        return base_lr
    progress = min(max(remaining_step / max(1, remaining_total - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(base_lr * (float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine))


def random_resized_crop_batch(
    x: np.ndarray,
    rng: np.random.Generator,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> np.ndarray:
    """Apply torchvision-style random resized crop and resize back to input size."""
    batch_size, height, width, _ = x.shape
    out = np.empty_like(x)
    for index in range(batch_size):
        area = height * width
        applied = False
        for _ in range(10):
            target_area = float(area) * float(rng.uniform(scale[0], scale[1]))
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect = float(np.exp(rng.uniform(*log_ratio)))
            crop_h = int(round(np.sqrt(target_area * aspect)))
            crop_w = int(round(np.sqrt(target_area / aspect)))
            if 1 <= crop_h <= height and 1 <= crop_w <= width:
                y0 = int(rng.integers(0, max(1, height - crop_h + 1)))
                x0 = int(rng.integers(0, max(1, width - crop_w + 1)))
                crop = x[index, y0 : y0 + crop_h, x0 : x0 + crop_w, :]
                crop_u8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
                out[index] = np.asarray(resize(crop_u8, (height, width)), dtype=np.float32) / 255.0
                applied = True
                break
        if not applied:
            out[index] = x[index]
    return out


def _rotate_image_reflect(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate one image with reflection padding, centered rotation, and bilinear resampling."""
    height, width, _ = image.shape
    pad = int(np.ceil(max(height, width) * (np.sqrt(2.0) - 1.0) / 2.0)) + 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    padded_u8 = np.clip(padded * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(padded_u8)
    center = (padded.shape[1] / 2.0, padded.shape[0] / 2.0)
    rotated = pil.rotate(
        float(angle_degrees),
        resample=Image.BILINEAR,
        expand=False,
        center=center,
    )
    rotated_arr = np.asarray(rotated, dtype=np.float32) / 255.0
    y0 = pad
    x0 = pad
    return rotated_arr[y0 : y0 + height, x0 : x0 + width, :]


def _apply_color_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply brightness, contrast, and saturation jitter within safe ranges."""
    brightness = 1.0 + float(rng.uniform(-0.2, 0.2))
    contrast = 1.0 + float(rng.uniform(-0.2, 0.2))
    saturation = 1.0 + float(rng.uniform(-0.2, 0.2))

    # Brightness scales all channels uniformly.
    out = image * brightness

    # Contrast re-centers around the per-image mean to avoid channel drift.
    mean = np.mean(out, axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean

    # Saturation blends between grayscale and the current color image.
    grayscale = np.mean(out, axis=2, keepdims=True)
    out = grayscale + (out - grayscale) * saturation
    return out


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the approved training augmentation policy in a deterministic RNG order."""
    out = x.copy()
    batch_size, height, width, _ = out.shape

    # Geometry first: crop then flip, before any rotation or photometric changes.
    if rng.random() < 0.9:
        out = random_resized_crop_batch(out, rng)
    flip_mask = rng.random(batch_size) < 0.5
    out[flip_mask] = out[flip_mask, :, ::-1, :]

    # Rotation is optional to avoid over-stacking geometric transforms.
    for index in range(batch_size):
        if rng.random() < 0.5:
            angle = float(rng.uniform(-12.0, 12.0))
            out[index] = _rotate_image_reflect(out[index], angle)

    # Photometric jitter is also optional to keep the augmented distribution stable.
    for index in range(batch_size):
        if rng.random() < 0.5:
            out[index] = _apply_color_jitter(out[index], rng)

    # Cutout remains the final destructive transform in the stack.
    for index in range(batch_size):
        if rng.random() < 0.3:
            cut = int(rng.integers(max(2, height // 8), max(4, height // 4)))
            cy = int(rng.integers(0, height))
            cx = int(rng.integers(0, width))
            y1 = max(0, cy - cut // 2)
            y2 = min(height, cy + cut // 2)
            x1 = max(0, cx - cut // 2)
            x2 = min(width, cx + cut // 2)
            out[index, y1:y2, x1:x2, :] = 0.0

    return np.clip(out, 0.0, 1.0).astype(x.dtype, copy=False)
