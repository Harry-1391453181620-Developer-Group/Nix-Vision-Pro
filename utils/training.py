"""Shared training policies used by both NumPy and PyTorch backends.

This module centralizes augmentation, argument validation, MixUp, and
phase-based learning-rate logic so the two maintained backends stay aligned.
The helpers intentionally stay in NumPy/Pillow space because the two trainers
still load and preprocess images before backend-specific tensor conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from data.preprocessing import resize


EPS = 1e-12


@dataclass(frozen=True)
class PhaseConfig:
    """Resolved phase metadata for one epoch.

    Each epoch stores both its global phase membership and its local position
    inside that phase so the scheduler can restart cleanly per phase.
    """

    phase_index: int
    epoch_index_in_phase: int
    epochs_in_phase: int


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved augmentation strengths shared by both training backends.

    The numeric values are strengths, not probabilities. When a strength is
    zero, that transform is disabled. Otherwise the transform is applied with a
    random factor sampled from the documented range for that strength.
    """

    rotation: float = 12.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2


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


def _validate_unit_interval(name: str, value: float) -> float:
    """Validate a floating-point CLI strength that must stay within [0, 1]."""
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= value <= 1")
    return value


def validate_augmentation_args(
    rotation: float,
    brightness: float,
    contrast: float,
    saturation: float,
) -> AugmentationConfig:
    """Validate user-facing augmentation strengths before training starts.

    Rotation is an angle bound in degrees. The photometric values are strength
    values inside [0, 1], where the runtime factor is sampled from [1-s, 1+s].
    """
    rotation = float(rotation)
    if not (0.0 <= rotation < 180.0):
        raise ValueError("rotation must satisfy 0 <= value < 180")
    return AugmentationConfig(
        rotation=rotation,
        brightness=_validate_unit_interval("brightness", brightness),
        contrast=_validate_unit_interval("contrast", contrast),
        saturation=_validate_unit_interval("saturation", saturation),
    )


def validate_mixup_probability(mixup_prob: float) -> float:
    """Validate the per-batch MixUp activation probability."""
    return _validate_unit_interval("mixup_prob", mixup_prob)


def validate_focal_gamma(focal_gamma: float) -> float:
    """Validate focal-loss gamma before either backend builds its loss path."""
    focal_gamma = float(focal_gamma)
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be >= 0")
    return focal_gamma


def validate_model_width_scale(width_scale: float) -> float:
    """Validate the width multiplier used to shrink or grow stage 2 channels."""
    width_scale = float(width_scale)
    if width_scale <= 0.0:
        raise ValueError("model_width_scale must be > 0")
    return width_scale


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
    actual deduction is guarded separately to ensure the resulting effective LR
    stays above the scheduler floor and the next phase LR when that bound exists.
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


def compute_effective_learning_rate(
    scheduled_lr: float,
    phase_lr_offset: float,
    phase_base_lr: float,
    min_lr_ratio: float,
) -> float:
    """Convert the scheduler LR into the effective optimizer LR.

    Cosine or step scheduling still operates on the configured phase base LR.
    Temporary post-unfreeze deductions are represented as a cumulative offset and
    are applied after scheduling, with a floor at `phase_base_lr * min_lr_ratio`.
    """
    scheduled_lr = float(scheduled_lr)
    phase_lr_offset = float(phase_lr_offset)
    phase_base_lr = float(phase_base_lr)
    min_lr = phase_base_lr * float(min_lr_ratio)
    return float(max(scheduled_lr - phase_lr_offset, min_lr))


def adjust_phase_lr_offset_after_unfreeze(
    *,
    current_effective_lr: float,
    phase_lr_offset: float,
    after_unfreeze_lr_change: float,
    phase_base_lr: float,
    min_lr_ratio: float,
    next_phase_start_lr: float | None,
    eps: float = EPS,
) -> tuple[float, bool]:
    """Apply the additive post-unfreeze LR deduction when it is still useful.

    Rules:
    - only deduct when the current effective LR is still meaningfully above the
      scheduler floor (`> 2 * min_lr`)
    - the candidate LR after deduction must stay strictly positive
    - if a next phase exists, the candidate LR must stay at or above the next
      phase start LR, with a small epsilon for float robustness
    - the cumulative offset is capped so the effective LR can never undercut the
      next phase start LR, or the scheduler floor in the final phase
    """
    current_effective_lr = float(current_effective_lr)
    phase_lr_offset = float(phase_lr_offset)
    after_unfreeze_lr_change = float(after_unfreeze_lr_change)
    phase_base_lr = float(phase_base_lr)
    min_lr = phase_base_lr * float(min_lr_ratio)

    if after_unfreeze_lr_change <= 0.0:
        return phase_lr_offset, False
    if current_effective_lr <= (2.0 * min_lr + eps):
        return phase_lr_offset, False

    candidate_lr = current_effective_lr - after_unfreeze_lr_change
    if candidate_lr <= eps:
        return phase_lr_offset, False
    if next_phase_start_lr is not None and candidate_lr < (float(next_phase_start_lr) - eps):
        return phase_lr_offset, False

    if next_phase_start_lr is not None:
        max_offset = max(0.0, phase_base_lr - float(next_phase_start_lr))
    else:
        max_offset = max(0.0, phase_base_lr - min_lr)

    new_offset = min(phase_lr_offset + after_unfreeze_lr_change, max_offset)
    if new_offset <= (phase_lr_offset + eps):
        return phase_lr_offset, False
    return float(new_offset), True


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
    return rotated_arr[pad : pad + height, pad : pad + width, :]


def _apply_color_jitter(image: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    """Apply brightness, contrast, and saturation jitter from user-supplied strengths."""
    out = image.copy()

    if config.brightness > 0.0:
        brightness = float(rng.uniform(1.0 - config.brightness, 1.0 + config.brightness))
        out = out * brightness

    if config.contrast > 0.0:
        contrast = float(rng.uniform(1.0 - config.contrast, 1.0 + config.contrast))
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        out = (out - mean) * contrast + mean

    if config.saturation > 0.0:
        saturation = float(rng.uniform(1.0 - config.saturation, 1.0 + config.saturation))
        grayscale = np.mean(out, axis=2, keepdims=True)
        out = grayscale + (out - grayscale) * saturation

    return out


def _apply_random_erasing(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erase one random rectangle and fill it with the per-image channel mean.

    Filling with the current image mean keeps the erased region neutral even if a
    future preprocessing change makes zero a non-natural value.
    """
    height, width, _ = image.shape
    total_area = float(height * width)
    fill_value = np.mean(image, axis=(0, 1), keepdims=True)

    for _ in range(10):
        target_area = total_area * float(rng.uniform(0.10, 0.20))
        aspect_ratio = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
        erase_h = min(max(1, erase_h), height)
        erase_w = min(max(1, erase_w), width)
        if erase_h <= height and erase_w <= width:
            y0 = int(rng.integers(0, max(1, height - erase_h + 1)))
            x0 = int(rng.integers(0, max(1, width - erase_w + 1)))
            out = image.copy()
            out[y0 : y0 + erase_h, x0 : x0 + erase_w, :] = fill_value
            return out
    return image


def augment_batch(
    x: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig | None = None,
) -> np.ndarray:
    """Apply the shared training augmentation policy in a deterministic RNG order.

    The transform order is fixed so both backends see the same data policy:
    crop -> flip -> rotation -> color jitter -> random erasing.
    """
    config = config or AugmentationConfig()
    out = x.copy()
    batch_size = out.shape[0]

    # Geometry first so later photometric changes operate on final spatial data.
    if rng.random() < 0.9:
        out = random_resized_crop_batch(out, rng)
    flip_mask = rng.random(batch_size) < 0.5
    out[flip_mask] = out[flip_mask, :, ::-1, :]

    if config.rotation > 0.0:
        for index in range(batch_size):
            angle = float(rng.uniform(-config.rotation, config.rotation))
            if abs(angle) > EPS:
                out[index] = _rotate_image_reflect(out[index], angle)

    if config.brightness > 0.0 or config.contrast > 0.0 or config.saturation > 0.0:
        for index in range(batch_size):
            out[index] = _apply_color_jitter(out[index], rng, config)

    # Random erasing stays last because it is the only intentionally destructive transform.
    for index in range(batch_size):
        if rng.random() < 0.3:
            out[index] = _apply_random_erasing(out[index], rng)

    return np.clip(out, 0.0, 1.0).astype(x.dtype, copy=False)


def apply_mixup(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    prob: float,
    beta_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Apply MixUp to a batch of inputs and labels using one shared lambda.

    Returning the activation flag and lambda lets callers log or branch on the
    exact behavior without trying to re-sample any RNG state.
    """
    prob = float(prob)
    beta_alpha = float(beta_alpha)
    if x.shape[0] != y.shape[0]:
        raise ValueError("MixUp expects the same batch dimension for inputs and labels")
    if prob <= 0.0 or x.shape[0] < 2:
        return x, y, False, 1.0
    if beta_alpha <= 0.0:
        raise ValueError("beta_alpha must be > 0")
    if rng.random() >= prob:
        return x, y, False, 1.0

    lam = float(rng.beta(beta_alpha, beta_alpha))
    permutation = rng.permutation(x.shape[0])
    mixed_x = lam * x + (1.0 - lam) * x[permutation]
    mixed_y = lam * y + (1.0 - lam) * y[permutation]
    return (
        mixed_x.astype(x.dtype, copy=False),
        mixed_y.astype(y.dtype, copy=False),
        True,
        lam,
    )
