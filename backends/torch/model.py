"""PyTorch CNN + SE backend model.

The stage-2 width is now parameterized with a width scale so the project can use
smaller intermediate feature maps without hard-coding a single architecture.
Every entry point that constructs this model must therefore pass the same width
scale that was used during training if it wants checkpoint shapes to match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch import nn


DEFAULT_INPUT_SIZE: Tuple[int, int] = (32, 32)


@dataclass(frozen=True)
class CheckpointRuntimeConfig:
    input_size: Tuple[int, int]
    num_classes: int
    width_scale: float
    stage2_channels: int
    class_names: tuple[str, ...]
    metadata: dict[str, Any]


def load_checkpoint_state(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    """Load either a legacy raw state_dict or a structured checkpoint."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.suffix not in {".pt", ".pth"}:
        raise ValueError("Checkpoint path must use .pt or .pth extension")
    try:
        payload = torch.load(checkpoint, map_location=map_location or "cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=map_location or "cpu")
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint format: expected a state_dict mapping")
    if "model" in payload:
        state = payload.get("model")
        metadata = payload.get("meta", {})
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint format: `model` must be a state_dict mapping")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Invalid checkpoint format: `meta` must be a dictionary when present")
        return state, dict(metadata)
    return payload, {}


def _resolve_stage2_channels(width_scale: float) -> int:
    """Convert a width multiplier into a safe integer channel count.

    The base architecture uses 64 channels in stage 2. A scale of 0.75 therefore
    maps to 48 channels, which is the requested default reduction.
    """
    width_scale = float(width_scale)
    if width_scale <= 0.0:
        raise ValueError("width_scale must be > 0")
    return max(8, int(round(64 * width_scale)))


def _normalize_input_size(value: Any) -> Tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        height = int(value[0])
        width = int(value[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return (height, width)


def _infer_input_size_from_state(
    state: Mapping[str, torch.Tensor],
    default_input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> Tuple[int, int]:
    fc1_weight = state.get("fc1.weight")
    if fc1_weight is None or fc1_weight.ndim != 2:
        return tuple(default_input_size)
    flatten_dim = int(fc1_weight.shape[1])
    if flatten_dim <= 0 or flatten_dim % 128 != 0:
        return tuple(default_input_size)
    spatial_area = flatten_dim // 128
    spatial_edge = int(round(math.sqrt(spatial_area)))
    if spatial_edge * spatial_edge != spatial_area:
        return tuple(default_input_size)
    return (spatial_edge * 8, spatial_edge * 8)


def _normalize_checkpoint_class_names(
    value: Any,
    *,
    expected_count: int,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    class_names = tuple(str(item).strip() for item in value if str(item).strip())
    return class_names if len(class_names) == expected_count else ()


def resolve_checkpoint_runtime_config(
    path: str | Path,
    map_location: str | torch.device | None = None,
    *,
    default_input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> CheckpointRuntimeConfig:
    state, metadata = load_checkpoint_state(path, map_location=map_location)
    return resolve_runtime_config_from_state(
        state,
        metadata,
        default_input_size=default_input_size,
    )


def resolve_runtime_config_from_state(
    state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any] | None = None,
    *,
    default_input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> CheckpointRuntimeConfig:
    metadata_dict = {} if metadata is None else dict(metadata)

    fc2_weight = state.get("fc2.weight")
    if fc2_weight is None or fc2_weight.ndim != 2:
        raise ValueError("Checkpoint is missing `fc2.weight`, so num_classes cannot be resolved")
    conv3_weight = state.get("conv3.weight")
    if conv3_weight is None or conv3_weight.ndim != 4:
        raise ValueError("Checkpoint is missing `conv3.weight`, so width_scale cannot be resolved")

    num_classes = int(metadata_dict.get("num_classes", fc2_weight.shape[0]))
    stage2_channels = int(metadata_dict.get("stage2_channels", conv3_weight.shape[0]))
    width_scale = float(metadata_dict.get("width_scale", stage2_channels / 64.0))
    input_size = (
        _normalize_input_size(metadata_dict.get("input_size"))
        or _infer_input_size_from_state(state, default_input_size=default_input_size)
    )
    class_names = _normalize_checkpoint_class_names(
        metadata_dict.get("class_names"),
        expected_count=num_classes,
    )

    return CheckpointRuntimeConfig(
        input_size=input_size,
        num_classes=num_classes,
        width_scale=width_scale,
        stage2_channels=stage2_channels,
        class_names=class_names,
        metadata=metadata_dict,
    )


class SqueezeExcitation(nn.Module):
    """Channel attention block used after each two-convolution stage."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class TorchCNN(nn.Module):
    """Three-stage CNN + SE classifier matching the active project architecture."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_classes: int,
        seed: int | None = None,
        dropout_p: float = 0.5,
        width_scale: float = 0.75,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        height, width = input_size
        if height < 8 or width < 8:
            raise ValueError("input_size must be at least (8, 8)")

        stage2_channels = _resolve_stage2_channels(width_scale)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.se1 = SqueezeExcitation(32, reduction=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2 is the width-scaled stage. This is the main architecture knob.
        self.conv3 = nn.Conv2d(32, stage2_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(stage2_channels)
        self.conv4 = nn.Conv2d(stage2_channels, stage2_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(stage2_channels)
        self.se2 = SqueezeExcitation(stage2_channels, reduction=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3 keeps its output width so the classifier head dimension stays stable.
        self.conv5 = nn.Conv2d(stage2_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.se3 = SqueezeExcitation(128, reduction=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        feat_h, feat_w = height // 8, width // 8
        self.fc1 = nn.Linear(feat_h * feat_w * 128, 256)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(256, num_classes)

        self._input_size = tuple(input_size)
        self._num_classes = int(num_classes)
        self._width_scale = float(width_scale)
        self._stage2_channels = int(stage2_channels)

    @property
    def width_scale(self) -> float:
        """Expose the configured width scale for tests and debugging."""
        return self._width_scale

    @property
    def stage2_channels(self) -> int:
        """Expose the resolved stage-2 channel count for tests and diagnostics."""
        return self._stage2_channels

    @property
    def input_size(self) -> Tuple[int, int]:
        """Expose the configured input size for checkpoint metadata and tests."""
        return self._input_size

    @property
    def num_classes(self) -> int:
        """Expose the classifier output size for checkpoint metadata and tests."""
        return self._num_classes

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        """Return the feature extractor modules affected by temporary freezing."""
        return (
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            self.se1,
            self.conv3,
            self.bn3,
            self.conv4,
            self.bn4,
            self.se2,
            self.conv5,
            self.bn5,
            self.conv6,
            self.bn6,
            self.se3,
        )

    def backbone_batchnorm_layers(self) -> tuple[nn.BatchNorm2d, ...]:
        """Return backbone BN layers so the trainer can control stats during freeze."""
        return (self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6)

    def head_modules(self) -> tuple[nn.Module, ...]:
        """Return the classifier head modules that stay trainable during freeze."""
        return (self.fc1, self.fc2)

    def iter_head_parameters(self) -> Iterable[nn.Parameter]:
        """Yield classifier head parameters in a stable order."""
        for module in self.head_modules():
            yield from module.parameters()

    def iter_backbone_parameters(self) -> Iterable[nn.Parameter]:
        """Yield all backbone parameters in a stable order."""
        for module in self.backbone_modules():
            yield from module.parameters()

    def iter_backbone_bn_affine_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only BN affine parameters for the optional adaptive-freeze mode."""
        for bn_layer in self.backbone_batchnorm_layers():
            if bn_layer.weight is not None:
                yield bn_layer.weight
            if bn_layer.bias is not None:
                yield bn_layer.bias

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.se1(x)
        x = self.pool1(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.se2(x)
        x = self.pool2(x)

        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.se3(x)
        x = self.pool3(x)
        return x

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(x)!r}")
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32)

        # Accept both NHWC and NCHW to keep compatibility with the current pipeline.
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {tuple(x.shape)}")
        if x.shape[1] == 3:
            pass
        elif x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError("Input must be NHWC or NCHW with 3 channels")

        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def save_weights(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        checkpoint = Path(path)
        if checkpoint.suffix not in {".pt", ".pth"}:
            raise ValueError("Checkpoint path must use .pt or .pth extension")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload_metadata = {
            "checkpoint_version": 2,
            "backend": "torch",
            "num_classes": int(self.num_classes),
            "width_scale": float(self.width_scale),
            "stage2_channels": int(self.stage2_channels),
            "input_size": list(self.input_size),
            **({} if metadata is None else dict(metadata)),
        }
        payload = {
            "model": self.state_dict(),
            "meta": payload_metadata,
        }
        torch.save(payload, checkpoint)

    def load_weights(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        state, metadata = load_checkpoint_state(path, map_location=map_location)
        self.load_state_dict(state)
        return metadata


CNN = TorchCNN

__all__ = [
    "CheckpointRuntimeConfig",
    "TorchCNN",
    "CNN",
    "load_checkpoint_state",
    "resolve_checkpoint_runtime_config",
    "resolve_runtime_config_from_state",
]
