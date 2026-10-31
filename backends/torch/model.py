"""CNN backbone + MAOL prototype head for Nix Vision Pro.

The transformer branch, Omega projector, and IDSI instrumentation were removed
when the project moved from MAOIDL to MAOL.  The model is a five-stage CNN with
SE blocks; ``h`` is the L2-normalized ``fc1 + ReLU`` representation and the
prototype head replaces the old linear classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from maol.head import PrototypeHead
from maol.tree import ClassTree


DEFAULT_INPUT_SIZE: Tuple[int, int] = (32, 32)
DEFAULT_EMBEDDING_DIM = 256


@dataclass(frozen=True)
class CheckpointRuntimeConfig:
    input_size: Tuple[int, int]
    num_classes: int
    width_scale: float
    stage2_channels: int
    stage_channels: tuple[int, ...]
    class_names: tuple[str, ...]
    tree: ClassTree | None
    head_type: str
    embedding_dim: int
    embedding_norm: str | None
    tau: float | None
    metadata: dict[str, Any]


def load_checkpoint_state(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    """Load a structured MAOL checkpoint with ``weights_only=True``."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.suffix not in {".pt", ".pth"}:
        raise ValueError("Checkpoint path must use .pt or .pth extension")
    try:
        payload = torch.load(
            checkpoint,
            map_location=map_location or "cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError("This PyTorch build does not support weights_only loading") from exc
    except Exception as exc:
        raise ValueError(
            f"Checkpoint {checkpoint} could not be loaded safely with weights_only=True. "
            "Only tensor/metadata checkpoints are supported."
        ) from exc
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


def _resolve_stage_channels(width_scale: float) -> tuple[int, int, int, int, int]:
    width_scale = float(width_scale)
    if width_scale <= 0.0:
        raise ValueError("width_scale must be > 0")
    base = (32, 64, 128, 256, 256)
    return tuple(max(8, int(round(channels * width_scale))) for channels in base)  # type: ignore[return-value]


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
    default_input_size: Tuple[int, int],
    stage_channels: tuple[int, ...],
) -> Tuple[int, int]:
    fc1_weight = state.get("fc1.weight")
    if fc1_weight is None or fc1_weight.ndim != 2:
        return tuple(default_input_size)
    flatten_dim = int(fc1_weight.shape[1])
    channels = int(stage_channels[-1]) if stage_channels else 256
    if channels <= 0 or flatten_dim % channels != 0:
        return tuple(default_input_size)
    spatial_area = flatten_dim // channels
    spatial_edge = int(round(math.sqrt(spatial_area)))
    if spatial_edge * spatial_edge != spatial_area:
        return tuple(default_input_size)
    return (spatial_edge * 32, spatial_edge * 32)


def _normalize_class_names(value: Any, expected_count: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    names = tuple(str(item).strip() for item in value if str(item).strip())
    return names if len(names) == expected_count else ()


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

    prototype_centers = state.get("head.centers")
    if prototype_centers is None:
        prototype_centers = state.get("prototype.centers")
    fc2_weight = state.get("fc2.weight")
    if prototype_centers is not None:
        num_classes = int(metadata_dict.get("num_classes", prototype_centers.shape[0]))
    elif fc2_weight is not None and fc2_weight.ndim == 2:
        num_classes = int(metadata_dict.get("num_classes", fc2_weight.shape[0]))
    else:
        raise ValueError("Checkpoint is missing both prototype and linear classifier weights")

    raw_stage_channels = metadata_dict.get("stage_channels")
    stage_channels: tuple[int, ...] | None = None
    if isinstance(raw_stage_channels, (list, tuple)) and raw_stage_channels:
        try:
            parsed = tuple(int(channel) for channel in raw_stage_channels)
            if all(channel > 0 for channel in parsed):
                stage_channels = parsed
        except (TypeError, ValueError):
            stage_channels = None

    if stage_channels is None:
        conv3_weight = state.get("conv3.weight")
        if conv3_weight is None or conv3_weight.ndim != 4:
            raise ValueError("Checkpoint is missing stage channel metadata and `conv3.weight`")
        stage_channels = (32, int(conv3_weight.shape[0]), 128, 256, 256)
        width_scale = float(metadata_dict.get("width_scale", stage_channels[1] / 64.0))
    else:
        width_scale = float(metadata_dict.get("width_scale", 1.0))

    stage2_channels = int(stage_channels[1]) if len(stage_channels) > 1 else int(stage_channels[0])
    input_size = (
        _normalize_input_size(metadata_dict.get("input_size"))
        or _infer_input_size_from_state(state, default_input_size, stage_channels)
    )
    class_names = _normalize_class_names(metadata_dict.get("class_names"), num_classes)

    tree_payload = metadata_dict.get("tree")
    tree: ClassTree | None
    if isinstance(tree_payload, Mapping):
        tree = ClassTree.from_dict(tree_payload)
    else:
        tree = None

    head_type = str(metadata_dict.get("head_type", "prototype" if prototype_centers is not None else "linear"))
    embedding_norm = str(metadata_dict.get("embedding_norm", "raw")).strip().lower()
    embedding_dim = int(
        metadata_dict.get(
            "embedding_dim",
            int(prototype_centers.shape[1]) if prototype_centers is not None else DEFAULT_EMBEDDING_DIM,
        )
    )
    tau_value = metadata_dict.get("tau")
    tau = float(tau_value) if tau_value is not None else None

    return CheckpointRuntimeConfig(
        input_size=input_size,
        num_classes=num_classes,
        width_scale=width_scale,
        stage2_channels=stage2_channels,
        stage_channels=tuple(stage_channels),
        class_names=class_names,
        tree=tree,
        head_type=head_type,
        embedding_dim=embedding_dim,
        embedding_norm=embedding_norm,
        tau=tau,
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
    """Five-stage CNN + SE backbone with a MAOL prototype classifier."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_classes: int,
        tree: ClassTree | None,
        *,
        seed: int | None = None,
        width_scale: float = 0.75,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        node_weights: Iterable[float] | None = None,
    ):
        super().__init__()
        if tree is None:
            raise ValueError("MAOL requires a class tree")
        if int(num_classes) != tree.num_leaves:
            raise ValueError(f"num_classes={num_classes} does not match tree leaves={tree.num_leaves}")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        height, width = input_size
        if height < 8 or width < 8:
            raise ValueError("input_size must be at least (8, 8)")
        c1, c2, c3, c4, c5 = _resolve_stage_channels(width_scale)
        self._stage_channels = (c1, c2, c3, c4, c5)

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c1)
        self.se1 = SqueezeExcitation(c1, reduction=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(c2)
        self.se2 = SqueezeExcitation(c2, reduction=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(c3)
        self.conv6 = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(c3)
        self.se3 = SqueezeExcitation(c3, reduction=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(c4)
        self.conv8 = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(c4)
        self.se4 = SqueezeExcitation(c4, reduction=4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(c5)
        self.conv10 = nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(c5)
        self.se5 = SqueezeExcitation(c5, reduction=4)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        feat_h, feat_w = height // 32, width // 32
        self.fc1 = nn.Linear(feat_h * feat_w * c5, int(embedding_dim))
        self.head = PrototypeHead(
            tree,
            int(embedding_dim),
            node_weights=list(node_weights) if node_weights is not None else None,
        )

        self._input_size = tuple(input_size)
        self._num_classes = int(num_classes)
        self._width_scale = float(width_scale)
        self._stage2_channels = int(c2)
        self._embedding_dim = int(embedding_dim)
        self.tree = tree
        self.class_names = tuple(tree.class_names)

    # ------------------------------------------------------------------
    # Properties and module groups
    # ------------------------------------------------------------------
    @property
    def stage2_channels(self) -> int:
        return self._stage2_channels

    @property
    def stage_channels(self) -> tuple[int, ...]:
        return self._stage_channels

    @property
    def width_scale(self) -> float:
        return self._width_scale

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._input_size

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.conv1, self.bn1, self.conv2, self.bn2, self.se1,
            self.conv3, self.bn3, self.conv4, self.bn4, self.se2,
            self.conv5, self.bn5, self.conv6, self.bn6, self.se3,
            self.conv7, self.bn7, self.conv8, self.bn8, self.se4,
            self.conv9, self.bn9, self.conv10, self.bn10, self.se5,
        )

    def backbone_batchnorm_layers(self) -> tuple[nn.BatchNorm2d, ...]:
        return (self.bn1, self.bn2, self.bn3, self.bn4, self.bn5,
                self.bn6, self.bn7, self.bn8, self.bn9, self.bn10)

    def head_modules(self) -> tuple[nn.Module, ...]:
        return (self.fc1, self.head)

    def iter_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.head_modules():
            yield from module.parameters()

    def iter_backbone_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.backbone_modules():
            yield from module.parameters()

    def iter_backbone_bn_affine_parameters(self) -> Iterable[nn.Parameter]:
        for bn_layer in self.backbone_batchnorm_layers():
            if bn_layer.weight is not None:
                yield bn_layer.weight
            if bn_layer.bias is not None:
                yield bn_layer.bias

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _normalize_runtime_input(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(x)!r}")
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {tuple(x.shape)}")
        if x.shape[1] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2)
        raise ValueError("Input must be NHWC or NCHW with 3 channels")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(self.se1(x))

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(self.se2(x))

        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool3(self.se3(x))

        x = torch.relu(self.bn7(self.conv7(x)))
        x = torch.relu(self.bn8(self.conv8(x)))
        x = self.pool4(self.se4(x))

        x = torch.relu(self.bn9(self.conv9(x)))
        x = torch.relu(self.bn10(self.conv10(x)))
        x = self.pool5(self.se5(x))
        return x

    def forward_raw_embedding(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        x_tensor = self._normalize_runtime_input(x)
        features = self.forward_features(x_tensor)
        return torch.relu(self.fc1(torch.flatten(features, start_dim=1)))

    def forward_embedding(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return the L2-normalized embedding used by the head and all MAOL losses.

        The backbone's raw ReLU scale is unbounded; normalizing here keeps the
        classifier, wells, radii, quantiles, margins and tau in one unit-sphere
        geometry.  This is parameter-free and does not add a layer.
        """
        raw = self.forward_raw_embedding(x)
        return F.normalize(raw, p=2, dim=-1, eps=1e-6)

    def forward_with_representation(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_embedding(x)
        return self.head.logits(h), h

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.head.logits(self.forward_embedding(x))

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_weights(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        checkpoint = Path(path)
        if checkpoint.suffix not in {".pt", ".pth"}:
            raise ValueError("Checkpoint path must use .pt or .pth extension")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload_metadata = {
            "checkpoint_version": 3,
            "theory": "MAOL",
            "head_type": "prototype",
            "backend": "torch",
            "num_classes": int(self.num_classes),
            "class_names": list(self.class_names),
            "tree": self.tree.to_dict(),
            "input_size": list(self.input_size),
            "width_scale": float(self.width_scale),
            "stage2_channels": int(self.stage2_channels),
            "stage_channels": list(self.stage_channels),
            "embedding_dim": int(self.embedding_dim),
            "embedding_norm": "l2",
            "tau": float(self.head.tau.detach().cpu().item()),
            **({} if metadata is None else dict(metadata)),
        }
        payload = {"model": self.state_dict(), "meta": payload_metadata}
        torch.save(payload, checkpoint)

    def load_weights(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        state, metadata = load_checkpoint_state(path, map_location=map_location)
        self.load_state_dict(state, strict=strict)
        return metadata


CNN = TorchCNN

__all__ = [
    "CheckpointRuntimeConfig",
    "DEFAULT_EMBEDDING_DIM",
    "TorchCNN",
    "CNN",
    "load_checkpoint_state",
    "resolve_checkpoint_runtime_config",
    "resolve_runtime_config_from_state",
]
