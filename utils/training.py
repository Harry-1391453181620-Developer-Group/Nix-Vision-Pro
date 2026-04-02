"""Shared training policies used by both NumPy and PyTorch backends.

This module centralizes RandAugment, batch-mixing, EMA, and phase-based
learning-rate logic so the two maintained backends stay aligned. The helpers
intentionally stay in NumPy/Pillow space because the two trainers still load and
preprocess images before backend-specific tensor conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


EPS = 1e-12
RANDAUGMENT_LAYERS = 2
RANDAUGMENT_LEVEL = 0.9


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

    RandAugment samples from a fixed op pool, but the operation magnitudes still
    come from the user-facing CLI strengths so both backends share the same
    limits for geometry and photometric changes.
    """

    rotation: float = 12.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    randaugment_layers: int = RANDAUGMENT_LAYERS


class ModelEMA:
    """Maintain an EMA copy of a model, including floating buffers.

    The EMA model is a real model instance so validation and checkpoint saves can
    operate on it directly. EMA updates blend floating-point entries from the
    full `state_dict`, while non-floating values such as integer counters are
    copied exactly to avoid invalid interpolation.
    """

    def __init__(self, ema_model: Any, decay: float, phase_warmup_steps: int = 0):
        self.ema_model = ema_model
        self.decay = validate_ema_decay(decay)
        self.phase_warmup_steps = max(0, int(phase_warmup_steps))
        self.num_updates = 0

    def sync_from(self, model: Any) -> None:
        """Reset the EMA model to match the current live model exactly."""
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        if list(ema_state.keys()) != list(model_state.keys()):
            raise ValueError("EMA model state does not match source model state")
        for key in ema_state:
            _copy_state_value_inplace(ema_state[key], model_state[key])
        if hasattr(self.ema_model, "eval"):
            self.ema_model.eval()

    def _compute_effective_decay(self, step_in_phase: int, mix_active: bool) -> float:
        """Lower EMA decay briefly at each phase start so EMA can catch up.

        The first EMA window of each phase ramps from 0.9 up to the configured
        base decay. Mixed batches reduce the decay slightly again because their
        parameter updates are intentionally noisier.
        """
        decay = float(self.decay)
        if self.phase_warmup_steps > 0 and 0 <= int(step_in_phase) < self.phase_warmup_steps and decay > 0.9:
            progress = (int(step_in_phase) + 1) / self.phase_warmup_steps
            warmup_decay = 0.9 + (decay - 0.9) * progress
            decay = min(decay, warmup_decay)
        if mix_active:
            decay *= 0.999
        return float(min(max(decay, 0.0), 0.999999))

    def update(self, model: Any, *, step_in_phase: int, mix_active: bool) -> float:
        """Blend the current live model state into the EMA model."""
        decay = self._compute_effective_decay(step_in_phase=step_in_phase, mix_active=mix_active)
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        if list(ema_state.keys()) != list(model_state.keys()):
            raise ValueError("EMA model state does not match source model state")
        for key in ema_state:
            if _is_floating_state_value(model_state[key]):
                _blend_state_value_inplace(ema_state[key], model_state[key], decay)
            else:
                _copy_state_value_inplace(ema_state[key], model_state[key])
        self.num_updates += 1
        if hasattr(self.ema_model, "eval"):
            self.ema_model.eval()
        return decay


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
    """Validate user-facing augmentation strengths before training starts."""
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
    """Validate the per-batch mix activation probability."""
    return _validate_unit_interval("mixup_prob", mixup_prob)


def validate_cutmix_ratio(cutmix_ratio: float) -> float:
    """Validate the conditional probability of choosing CutMix over MixUp."""
    return _validate_unit_interval("cutmix_ratio", cutmix_ratio)


def validate_mixup_alpha(mixup_alpha: float) -> float:
    """Validate the shared Beta(alpha, alpha) parameter used by MixUp and CutMix."""
    mixup_alpha = float(mixup_alpha)
    if mixup_alpha <= 0.0:
        raise ValueError("mixup_alpha must be > 0")
    return mixup_alpha


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
    """Validate the timed-freeze configuration before training starts."""
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


def validate_ema_decay(ema_decay: float) -> float:
    """Validate EMA decay before either backend enables EMA."""
    ema_decay = float(ema_decay)
    if not (0.0 < ema_decay < 1.0):
        raise ValueError("ema_decay must satisfy 0 < value < 1")
    return ema_decay


def _is_floating_state_value(value: Any) -> bool:
    """Return whether a checkpoint entry is safe to EMA-blend."""
    if hasattr(value, "is_floating_point"):
        try:
            return bool(value.is_floating_point())
        except TypeError:
            pass
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    try:
        return bool(np.issubdtype(dtype, np.floating))
    except TypeError:
        return False


def _copy_state_value_inplace(target: Any, source: Any) -> None:
    """Copy one state entry into another without breaking storage sharing."""
    if hasattr(target, "copy_"):
        source_value = source.detach() if hasattr(source, "detach") else source
        target.copy_(source_value)
        return
    target[...] = np.asarray(source, dtype=target.dtype)


def _blend_state_value_inplace(target: Any, source: Any, decay: float) -> None:
    """Blend one floating state entry into another in place."""
    if hasattr(target, "mul_") and hasattr(target, "add_"):
        source_value = source.detach() if hasattr(source, "detach") else source
        target.mul_(decay)
        target.add_(source_value.to(device=target.device, dtype=target.dtype), alpha=1.0 - decay)
        return
    source_value = np.asarray(source, dtype=target.dtype)
    target[...] = decay * target + (1.0 - decay) * source_value


def compute_effective_learning_rate(
    scheduled_lr: float,
    phase_lr_offset: float,
    phase_base_lr: float,
    min_lr_ratio: float,
) -> float:
    """Convert the scheduler LR into the effective optimizer LR."""
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
    """Apply the additive post-unfreeze LR deduction when it is still useful."""
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
    """Compute the active LR for one batch inside a single phase."""
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")
    base_lr = float(base_lr)
    phase_total_steps = max(1, int(epochs_in_phase) * int(num_batches))
    step_in_phase = int(epoch_index_in_phase) * int(num_batches) + int(batch_index)
    warmup_steps = max(0, min(int(warmup_epochs), int(epochs_in_phase)) * int(num_batches))

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


def _to_pil_image(image: np.ndarray) -> Image.Image:
    """Convert one normalized float image into a PIL image for RandAugment ops."""
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_u8)


def _from_pil_image(image: Image.Image, dtype: np.dtype) -> np.ndarray:
    """Convert one PIL image back into the normalized float domain."""
    return (np.asarray(image, dtype=np.float32) / 255.0).astype(dtype, copy=False)


def _apply_random_erasing(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erase one random rectangle and fill it with the per-image channel mean."""
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


def _randaugment_signed_magnitude(rng: np.random.Generator, magnitude: float) -> float:
    """Sample a RandAugment magnitude with random sign for signed ops."""
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * float(magnitude)


def _apply_randaugment_operation(
    image: np.ndarray,
    op_name: str,
    rng: np.random.Generator,
    config: AugmentationConfig,
    level: float,
) -> np.ndarray:
    """Apply one RandAugment operation to a single normalized image."""
    if op_name == "cutout":
        return _apply_random_erasing(image, rng)

    pil_image = _to_pil_image(image)

    if op_name == "rotate":
        angle = _randaugment_signed_magnitude(rng, config.rotation * level)
        pil_image = pil_image.rotate(angle, resample=Image.BILINEAR)
    elif op_name == "brightness":
        factor = max(0.05, 1.0 + _randaugment_signed_magnitude(rng, config.brightness * level))
        pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)
    elif op_name == "contrast":
        factor = max(0.05, 1.0 + _randaugment_signed_magnitude(rng, config.contrast * level))
        pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)
    elif op_name == "saturation":
        factor = max(0.05, 1.0 + _randaugment_signed_magnitude(rng, config.saturation * level))
        pil_image = ImageEnhance.Color(pil_image).enhance(factor)
    elif op_name == "sharpness":
        factor = max(0.05, 1.0 + _randaugment_signed_magnitude(rng, 0.9 * level))
        pil_image = ImageEnhance.Sharpness(pil_image).enhance(factor)
    elif op_name == "posterize":
        bits = max(1, 8 - int(round(4.0 * level)))
        pil_image = ImageOps.posterize(pil_image, bits)
    elif op_name == "solarize":
        threshold = int(round(255.0 * (1.0 - 0.8 * level)))
        pil_image = ImageOps.solarize(pil_image, threshold=threshold)
    elif op_name == "autocontrast":
        pil_image = ImageOps.autocontrast(pil_image)
    elif op_name == "equalize":
        pil_image = ImageOps.equalize(pil_image)
    elif op_name == "invert":
        pil_image = ImageOps.invert(pil_image)
    else:
        raise ValueError(f"Unsupported RandAugment op: {op_name}")

    return _from_pil_image(pil_image, dtype=image.dtype)


def augment_batch(
    x: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig | None = None,
) -> np.ndarray:
    """Apply RandAugment to a batch in a deterministic RNG order.

    Validation does not call this helper, so RandAugment is training-only by
    construction. Each image samples the configured number of ops with
    replacement from the op pool, matching the intended RandAugment behavior.
    """
    config = config or AugmentationConfig()
    out = x.copy()
    op_pool = (
        "rotate",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "posterize",
        "solarize",
        "autocontrast",
        "equalize",
        "invert",
        "cutout",
    )
    for index in range(out.shape[0]):
        image = out[index]
        for _ in range(config.randaugment_layers):
            op_name = op_pool[int(rng.integers(0, len(op_pool)))]
            image = _apply_randaugment_operation(image, op_name, rng, config, level=RANDAUGMENT_LEVEL)
        out[index] = image
    return np.clip(out, 0.0, 1.0).astype(x.dtype, copy=False)


def apply_mixup(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    prob: float,
    beta_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Apply MixUp to a batch of inputs and labels using one shared lambda."""
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


def apply_cutmix(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    beta_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Apply CutMix and recompute lambda from the actual pasted patch area."""
    beta_alpha = float(beta_alpha)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CutMix expects the same batch dimension for inputs and labels")
    if x.shape[0] < 2:
        return x, y, False, 1.0
    if beta_alpha <= 0.0:
        raise ValueError("beta_alpha must be > 0")

    batch_size, height, width, _ = x.shape
    lam_sample = float(rng.beta(beta_alpha, beta_alpha))
    cut_ratio = np.sqrt(max(0.0, 1.0 - lam_sample))
    cut_w = max(1, int(round(width * cut_ratio)))
    cut_h = max(1, int(round(height * cut_ratio)))

    center_x = int(rng.integers(0, width))
    center_y = int(rng.integers(0, height))
    x1 = max(0, center_x - cut_w // 2)
    y1 = max(0, center_y - cut_h // 2)
    x2 = min(width, center_x + (cut_w + 1) // 2)
    y2 = min(height, center_y + (cut_h + 1) // 2)

    permutation = rng.permutation(batch_size)
    mixed_x = x.copy()
    mixed_x[:, y1:y2, x1:x2, :] = x[permutation, y1:y2, x1:x2, :]

    patch_area = float(max(0, y2 - y1) * max(0, x2 - x1))
    lam = 1.0 - (patch_area / float(height * width))
    mixed_y = lam * y + (1.0 - lam) * y[permutation]
    return (
        mixed_x.astype(x.dtype, copy=False),
        mixed_y.astype(y.dtype, copy=False),
        True,
        float(lam),
    )


def apply_batch_mix(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    prob: float,
    beta_alpha: float,
    cutmix_ratio: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Route a mixed batch to CutMix or MixUp through one probability gate."""
    prob = float(prob)
    cutmix_ratio = float(cutmix_ratio)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Batch mixing expects the same batch dimension for inputs and labels")
    if prob <= 0.0 or x.shape[0] < 2:
        return x, y, "none", 1.0
    if rng.random() >= prob:
        return x, y, "none", 1.0
    if rng.random() < cutmix_ratio:
        mixed_x, mixed_y, _, lam = apply_cutmix(x, y, rng, beta_alpha=beta_alpha)
        return mixed_x, mixed_y, "cutmix", lam
    mixed_x, mixed_y, _, lam = apply_mixup(x, y, rng, prob=1.0, beta_alpha=beta_alpha)
    return mixed_x, mixed_y, "mixup", lam
