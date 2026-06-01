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
DEFAULT_OMEGA_FEATURE_DIM = 256
DEFAULT_MAX_TOKENS = 64


def _normalize_omega_metadata_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return False


def _normalize_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _normalize_optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0.0 else None


def _normalize_choice(value: Any, choices: set[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in choices else None


@dataclass(frozen=True)
class CheckpointRuntimeConfig:
    input_size: Tuple[int, int]
    num_classes: int
    width_scale: float
    stage2_channels: int
    class_names: tuple[str, ...]
    omega_enabled: bool
    omega_projector_depth: int | None
    omega_hidden_dim: int | None
    tokenize: bool
    token_dim: int | None
    transformer_depth: int | None
    attention_heads: int | None
    transformer_mlp_ratio: float | None
    token_pool: str | None
    token_positional_encoding: str | None
    token_dropout: float | None
    transformer_layernorm: str | None
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
    omega_enabled = _normalize_omega_metadata_flag(metadata_dict.get("omega_enabled", False))
    omega_projector_depth = _normalize_optional_positive_int(metadata_dict.get("omega_projector_depth"))
    omega_hidden_dim = _normalize_optional_positive_int(metadata_dict.get("omega_hidden_dim"))
    token_projection_weight = state.get("token_projection.weight")
    tokenize = _normalize_omega_metadata_flag(
        metadata_dict.get("tokenize", token_projection_weight is not None)
    )
    token_dim = _normalize_optional_positive_int(metadata_dict.get("token_dim"))
    if token_dim is None and token_projection_weight is not None and token_projection_weight.ndim == 2:
        token_dim = int(token_projection_weight.shape[0])
    transformer_depth = _normalize_optional_positive_int(metadata_dict.get("transformer_depth"))
    attention_heads = _normalize_optional_positive_int(metadata_dict.get("attention_heads"))
    transformer_mlp_ratio = _normalize_optional_positive_float(metadata_dict.get("transformer_mlp_ratio"))
    token_pool = _normalize_choice(metadata_dict.get("token_pool"), {"mean", "cls"})
    token_positional_encoding = _normalize_choice(
        metadata_dict.get("token_positional_encoding"),
        {"none", "learned", "sinusoidal"},
    )
    try:
        token_dropout = float(metadata_dict["token_dropout"]) if "token_dropout" in metadata_dict else None
    except (TypeError, ValueError):
        token_dropout = None
    transformer_layernorm = _normalize_choice(metadata_dict.get("transformer_layernorm"), {"pre", "post"})

    return CheckpointRuntimeConfig(
        input_size=input_size,
        num_classes=num_classes,
        width_scale=width_scale,
        stage2_channels=stage2_channels,
        class_names=class_names,
        omega_enabled=omega_enabled,
        omega_projector_depth=omega_projector_depth,
        omega_hidden_dim=omega_hidden_dim,
        tokenize=bool(tokenize),
        token_dim=token_dim,
        transformer_depth=transformer_depth,
        attention_heads=attention_heads,
        transformer_mlp_ratio=transformer_mlp_ratio,
        token_pool=token_pool,
        token_positional_encoding=token_positional_encoding,
        token_dropout=token_dropout,
        transformer_layernorm=transformer_layernorm,
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


class OmegaProjector(nn.Module):
    """Shallow projector used by the Phase 1 Omega-loss path."""

    def __init__(self, *, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        if depth not in {1, 2}:
            raise ValueError("omega projector depth must be 1 or 2")
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("omega projector dimensions must be > 0")

        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(nn.Linear(input_dim, input_dim))
        else:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                ]
            )
        layers.append(nn.LayerNorm(input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def _resolve_token_grid(feat_h: int, feat_w: int, max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[int, int]:
    """Keep spatial tokens lightweight while preserving the feature-map aspect ratio."""
    feat_h = int(feat_h)
    feat_w = int(feat_w)
    max_tokens = int(max_tokens)
    if feat_h <= 0 or feat_w <= 0:
        raise ValueError("token feature grid must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if feat_h * feat_w <= max_tokens:
        return feat_h, feat_w

    scale = math.sqrt(max_tokens / float(feat_h * feat_w))
    target_h = max(1, int(math.floor(feat_h * scale)))
    target_w = max(1, int(math.floor(feat_w * scale)))
    while target_h * target_w > max_tokens:
        if target_h >= target_w and target_h > 1:
            target_h -= 1
        elif target_w > 1:
            target_w -= 1
        else:
            break
    return target_h, target_w


def _build_sinusoidal_positions(num_tokens: int, token_dim: int) -> torch.Tensor:
    positions = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, token_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / max(1, token_dim))
    )
    encoding = torch.zeros(num_tokens, token_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    if token_dim > 1:
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class TokenTransformerBlock(nn.Module):
    """Small residual transformer block used as a Phase 2 dynamics refinement."""

    def __init__(
        self,
        *,
        token_dim: int,
        attention_heads: int,
        mlp_ratio: float,
        dropout: float,
        layernorm: str,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if attention_heads <= 0 or token_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide token_dim")
        if mlp_ratio <= 0.0:
            raise ValueError("transformer_mlp_ratio must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("token_dropout must satisfy 0 <= value < 1")
        if layernorm not in {"pre", "post"}:
            raise ValueError("transformer_layernorm must be 'pre' or 'post'")

        hidden_dim = max(token_dim, int(round(token_dim * float(mlp_ratio))))
        self.layernorm = str(layernorm)
        self.residual_scale = 0.1
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(p=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(p=dropout),
        )
        self._init_small_updates()

    def _init_small_updates(self) -> None:
        nn.init.normal_(self.attn.in_proj_weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.normal_(self.attn.out_proj.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.attn.out_proj.bias)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
        final_linear = self.mlp[-2]
        if isinstance(final_linear, nn.Linear):
            nn.init.normal_(final_linear.weight, mean=0.0, std=0.001)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.layernorm == "pre":
            attn_input = self.norm1(tokens)
            attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
            tokens = tokens + (self.residual_scale * self.drop(attn_output))
            tokens = tokens + (self.residual_scale * self.mlp(self.norm2(tokens)))
            return tokens

        attn_output, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + (self.residual_scale * self.drop(attn_output)))
        return self.norm2(tokens + (self.residual_scale * self.mlp(tokens)))


class TorchCNN(nn.Module):
    """Three-stage CNN + SE classifier matching the active project architecture."""

    CNN_IDSI_LAYER_NAMES: tuple[str, ...] = (
        "stage1",
        "stage2",
        "stage3",
        "classifier_pre_head",
    )
    TOKEN_IDSI_LAYER_NAMES: tuple[str, ...] = (
        "stage1",
        "stage2",
        "stage3",
        "token_projection",
        "transformer_token_block",
    )

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_classes: int,
        seed: int | None = None,
        dropout_p: float = 0.5,
        width_scale: float = 0.75,
        omega_enabled: bool = False,
        omega_projector_depth: int = 1,
        omega_hidden_dim: int = DEFAULT_OMEGA_FEATURE_DIM,
        tokenize: bool = False,
        token_dim: int = 128,
        transformer_depth: int = 1,
        attention_heads: int = 4,
        transformer_mlp_ratio: float = 2.0,
        token_pool: str = "mean",
        token_positional_encoding: str = "learned",
        token_dropout: float = 0.1,
        transformer_layernorm: str = "pre",
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        height, width = input_size
        if height < 8 or width < 8:
            raise ValueError("input_size must be at least (8, 8)")
        if token_pool not in {"mean", "cls"}:
            raise ValueError("token_pool must be 'mean' or 'cls'")
        if token_positional_encoding not in {"none", "learned", "sinusoidal"}:
            raise ValueError("token_positional_encoding must be 'none', 'learned', or 'sinusoidal'")
        if transformer_layernorm not in {"pre", "post"}:
            raise ValueError("transformer_layernorm must be 'pre' or 'post'")
        token_dim = int(token_dim)
        transformer_depth = int(transformer_depth)
        attention_heads = int(attention_heads)
        transformer_mlp_ratio = float(transformer_mlp_ratio)
        token_dropout = float(token_dropout)
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if transformer_depth <= 0:
            raise ValueError("transformer_depth must be > 0")
        if attention_heads <= 0 or token_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide token_dim")
        if transformer_mlp_ratio <= 0.0:
            raise ValueError("transformer_mlp_ratio must be > 0")
        if not (0.0 <= token_dropout < 1.0):
            raise ValueError("token_dropout must satisfy 0 <= value < 1")

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
        self.fc1 = nn.Linear(feat_h * feat_w * 128, DEFAULT_OMEGA_FEATURE_DIM)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(DEFAULT_OMEGA_FEATURE_DIM, num_classes)
        token_grid_h, token_grid_w = _resolve_token_grid(feat_h, feat_w, DEFAULT_MAX_TOKENS)
        if tokenize:
            self.token_feature_pool = (
                nn.Identity()
                if (token_grid_h, token_grid_w) == (feat_h, feat_w)
                else nn.AdaptiveAvgPool2d((token_grid_h, token_grid_w))
            )
            self.token_projection = nn.Linear(128, token_dim)
            self.token_dropout = nn.Dropout(p=token_dropout)
            self.token_transformer = nn.Sequential(
                *[
                    TokenTransformerBlock(
                        token_dim=token_dim,
                        attention_heads=attention_heads,
                        mlp_ratio=transformer_mlp_ratio,
                        dropout=token_dropout,
                        layernorm=transformer_layernorm,
                    )
                    for _ in range(transformer_depth)
                ]
            )
            self.token_classifier = nn.Linear(token_dim, num_classes)
        else:
            self.token_feature_pool = None
            self.token_projection = None
            self.token_dropout = None
            self.token_transformer = None
            self.token_classifier = None
        if tokenize and token_pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.01)
        else:
            self.register_parameter("cls_token", None)
        num_spatial_tokens = token_grid_h * token_grid_w
        if tokenize and token_positional_encoding == "learned":
            self.token_positional_embedding = nn.Parameter(torch.zeros(1, num_spatial_tokens, token_dim))
            nn.init.normal_(self.token_positional_embedding, mean=0.0, std=0.01)
            self.register_buffer("token_sinusoidal_embedding", torch.zeros(1, 0, token_dim), persistent=False)
        elif tokenize and token_positional_encoding == "sinusoidal":
            self.register_parameter("token_positional_embedding", None)
            self.register_buffer(
                "token_sinusoidal_embedding",
                _build_sinusoidal_positions(num_spatial_tokens, token_dim),
                persistent=False,
            )
        else:
            self.register_parameter("token_positional_embedding", None)
            self.register_buffer("token_sinusoidal_embedding", torch.zeros(1, 0, token_dim), persistent=False)
        self.omega_projector = (
            OmegaProjector(
                input_dim=DEFAULT_OMEGA_FEATURE_DIM,
                hidden_dim=int(omega_hidden_dim),
                depth=int(omega_projector_depth),
            )
            if omega_enabled and not tokenize
            else None
        )

        self._input_size = tuple(input_size)
        self._num_classes = int(num_classes)
        self._width_scale = float(width_scale)
        self._stage2_channels = int(stage2_channels)
        self._omega_enabled = bool(self.omega_projector is not None)
        self._omega_projector_depth = int(omega_projector_depth) if self.omega_projector is not None else None
        self._omega_hidden_dim = int(omega_hidden_dim) if self.omega_projector is not None else None
        self._tokenize = bool(tokenize)
        self._token_dim = int(token_dim)
        self._transformer_depth = int(transformer_depth)
        self._attention_heads = int(attention_heads)
        self._transformer_mlp_ratio = float(transformer_mlp_ratio)
        self._token_pool = str(token_pool)
        self._token_positional_encoding = str(token_positional_encoding)
        self._token_dropout = float(token_dropout)
        self._transformer_layernorm = str(transformer_layernorm)
        self._token_grid_size = (int(token_grid_h), int(token_grid_w))

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

    @property
    def omega_enabled(self) -> bool:
        """Expose whether the Phase 1 Omega projector exists."""
        return self._omega_enabled

    @property
    def omega_projector_depth(self) -> int | None:
        """Expose the configured Omega projector depth for checkpoint metadata."""
        return self._omega_projector_depth

    @property
    def omega_hidden_dim(self) -> int | None:
        """Expose the configured Omega hidden dimension for checkpoint metadata."""
        return self._omega_hidden_dim

    @property
    def tokenize(self) -> bool:
        """Expose whether the Phase 2 tokenized dynamics path is active."""
        return self._tokenize

    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def transformer_depth(self) -> int:
        return self._transformer_depth

    @property
    def attention_heads(self) -> int:
        return self._attention_heads

    @property
    def transformer_mlp_ratio(self) -> float:
        return self._transformer_mlp_ratio

    @property
    def token_pool(self) -> str:
        return self._token_pool

    @property
    def token_positional_encoding(self) -> str:
        return self._token_positional_encoding

    @property
    def token_dropout_p(self) -> float:
        return self._token_dropout

    @property
    def transformer_layernorm(self) -> str:
        return self._transformer_layernorm

    @property
    def token_grid_size(self) -> tuple[int, int]:
        return self._token_grid_size

    @property
    def idsi_layer_names(self) -> tuple[str, ...]:
        """Stable monitored-layer names used by Layer-IDSI logging."""
        return self.TOKEN_IDSI_LAYER_NAMES if self.tokenize else self.CNN_IDSI_LAYER_NAMES

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
        modules: list[nn.Module] = [self.fc1, self.fc2]
        if self.omega_projector is not None:
            modules.append(self.omega_projector)
        if self.tokenize:
            token_modules = (
                self.token_feature_pool,
                self.token_projection,
                self.token_dropout,
                self.token_transformer,
                self.token_classifier,
            )
            modules.extend(module for module in token_modules if isinstance(module, nn.Module))
        return tuple(modules)

    def iter_head_parameters(self) -> Iterable[nn.Parameter]:
        """Yield classifier head parameters in a stable order."""
        for module in self.head_modules():
            yield from module.parameters()
        if self.tokenize:
            if self.cls_token is not None:
                yield self.cls_token
            if self.token_positional_embedding is not None:
                yield self.token_positional_embedding

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

    def _forward_features_with_layer_idsi(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return features plus matched-space stage transitions for Layer-IDSI.

        Whole CNN stages change channel count or spatial resolution. Phase 1.2
        therefore monitors the residual-aligned same-shape transition inside
        each stage, after the stage projection convolution and before pooling.
        """
        stage1_in = torch.relu(self.bn1(self.conv1(x)))
        stage1_out = torch.relu(self.bn2(self.conv2(stage1_in)))
        stage1_out = self.se1(stage1_out)
        x = self.pool1(stage1_out)

        stage2_in = torch.relu(self.bn3(self.conv3(x)))
        stage2_out = torch.relu(self.bn4(self.conv4(stage2_in)))
        stage2_out = self.se2(stage2_out)
        x = self.pool2(stage2_out)

        stage3_in = torch.relu(self.bn5(self.conv5(x)))
        stage3_out = torch.relu(self.bn6(self.conv6(stage3_in)))
        stage3_out = self.se3(stage3_out)
        x = self.pool3(stage3_out)

        return x, (stage1_in, stage2_in, stage3_in), (stage1_out, stage2_out, stage3_out)

    def _add_token_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_positional_encoding == "none":
            return tokens
        if self.token_positional_encoding == "learned":
            if self.token_positional_embedding is None:
                return tokens
            return tokens + self.token_positional_embedding.to(dtype=tokens.dtype, device=tokens.device)
        if self.token_sinusoidal_embedding.numel() == 0:
            return tokens
        return tokens + (0.05 * self.token_sinusoidal_embedding.to(dtype=tokens.dtype, device=tokens.device))

    def _tokens_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.tokenize:
            raise RuntimeError("Tokenization is disabled for this model instance")
        assert isinstance(self.token_feature_pool, nn.Module)
        assert isinstance(self.token_projection, nn.Linear)
        pooled_features = self.token_feature_pool(features)
        raw_tokens = pooled_features.flatten(2).transpose(1, 2).contiguous()
        projected_tokens = self.token_projection(raw_tokens)
        positioned_tokens = self._add_token_positional_encoding(projected_tokens)
        return projected_tokens, positioned_tokens

    def _forward_token_dynamics_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.tokenize:
            raise RuntimeError("Tokenization is disabled for this model instance")
        assert isinstance(self.token_dropout, nn.Dropout)
        assert isinstance(self.token_transformer, nn.Sequential)
        assert isinstance(self.token_classifier, nn.Linear)

        projected_tokens, positioned_tokens = self._tokens_from_features(features)
        transformer_input = positioned_tokens
        if self.token_pool == "cls":
            if self.cls_token is None:
                raise RuntimeError("CLS token pooling requires a cls_token parameter")
            cls_tokens = self.cls_token.expand(transformer_input.shape[0], -1, -1)
            transformer_input = torch.cat((cls_tokens, transformer_input), dim=1)
        transformer_input = self.token_dropout(transformer_input)
        refined_tokens = self.token_transformer(transformer_input)
        if self.token_pool == "cls":
            pooled = refined_tokens[:, 0]
            spatial_refined_tokens = refined_tokens[:, 1:]
        else:
            spatial_refined_tokens = refined_tokens
            pooled = spatial_refined_tokens.mean(dim=1)
        logits = self.token_classifier(self.dropout(pooled))
        return logits, projected_tokens, positioned_tokens, transformer_input, refined_tokens, spatial_refined_tokens

    def _normalize_runtime_input(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
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
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2)
        raise ValueError("Input must be NHWC or NCHW with 3 channels")

    def _forward_representation(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        x_tensor = self._normalize_runtime_input(x)
        features = self._forward_features(x_tensor)
        if self.tokenize:
            logits, _, _, _, _, spatial_refined_tokens = self._forward_token_dynamics_from_features(features)
            del logits
            return spatial_refined_tokens.mean(dim=1)
        flattened = torch.flatten(features, start_dim=1)
        return torch.relu(self.fc1(flattened))

    def _forward_representation_with_layer_idsi(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x_tensor = self._normalize_runtime_input(x)
        features, layer_inputs, layer_outputs = self._forward_features_with_layer_idsi(x_tensor)
        if self.tokenize:
            logits, _, _, _, _, spatial_refined_tokens = self._forward_token_dynamics_from_features(features)
            del logits
            return spatial_refined_tokens.mean(dim=1), layer_inputs, layer_outputs
        flattened = torch.flatten(features, start_dim=1)
        h = torch.relu(self.fc1(flattened))
        return h, layer_inputs, layer_outputs

    def _forward_logits_from_representation(self, h: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(h)
        return self.fc2(dropped)

    def forward_with_representation(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits plus the 256-d representation used by Phase 1 Omega-loss."""
        if self.tokenize:
            x_tensor = self._normalize_runtime_input(x)
            features = self._forward_features(x_tensor)
            logits, _, _, _, _, spatial_refined_tokens = self._forward_token_dynamics_from_features(features)
            return logits, spatial_refined_tokens.mean(dim=1)
        h = self._forward_representation(x)
        return self._forward_logits_from_representation(h), h

    def forward_with_token_dynamics(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits and token-space input/output states for Phase 2 metrics."""
        if not self.tokenize:
            raise RuntimeError("Tokenization is disabled for this model instance")
        x_tensor = self._normalize_runtime_input(x)
        features = self._forward_features(x_tensor)
        logits, _, _, transformer_input, refined_tokens, _ = self._forward_token_dynamics_from_features(features)
        return logits, transformer_input, refined_tokens

    def forward_with_token_dynamics_and_layer_idsi(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
    ]:
        """Return token dynamics plus CNN/token Layer-IDSI transition tensors."""
        if not self.tokenize:
            raise RuntimeError("Tokenization is disabled for this model instance")
        x_tensor = self._normalize_runtime_input(x)
        features, cnn_inputs, cnn_outputs = self._forward_features_with_layer_idsi(x_tensor)
        (
            logits,
            projected_tokens,
            positioned_tokens,
            transformer_input,
            refined_tokens,
            spatial_refined_tokens,
        ) = self._forward_token_dynamics_from_features(features)
        return (
            logits,
            transformer_input,
            refined_tokens,
            cnn_inputs + (projected_tokens, transformer_input),
            cnn_outputs + (positioned_tokens, refined_tokens),
            spatial_refined_tokens,
        )

    def forward_with_omega(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, representation, and Omega projector output for training."""
        if self.tokenize:
            return self.forward_with_token_dynamics(x)
        if self.omega_projector is None:
            raise RuntimeError("Omega projector is disabled for this model instance")
        logits, h = self.forward_with_representation(x)
        return logits, h, self.omega_projector(h)

    def forward_with_omega_and_layer_idsi(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        """Return Omega outputs plus matched-space Layer-IDSI transition tensors."""
        if self.tokenize:
            return self.forward_with_token_dynamics_and_layer_idsi(x)
        if self.omega_projector is None:
            raise RuntimeError("Omega projector is disabled for this model instance")
        h, layer_inputs, layer_outputs = self._forward_representation_with_layer_idsi(x)
        t_h = self.omega_projector(h)
        logits = self._forward_logits_from_representation(h)
        return logits, h, t_h, layer_inputs + (h,), layer_outputs + (t_h,)

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.tokenize:
            x_tensor = self._normalize_runtime_input(x)
            features = self._forward_features(x_tensor)
            logits, _, _, _, _, _ = self._forward_token_dynamics_from_features(features)
            return logits
        logits, _ = self.forward_with_representation(x)
        return logits

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
            "omega_enabled": bool(self.omega_enabled),
            "omega_projector_depth": self.omega_projector_depth,
            "omega_hidden_dim": self.omega_hidden_dim,
            "tokenize": bool(self.tokenize),
            "token_dim": int(self.token_dim),
            "transformer_depth": int(self.transformer_depth),
            "attention_heads": int(self.attention_heads),
            "transformer_mlp_ratio": float(self.transformer_mlp_ratio),
            "token_pool": self.token_pool,
            "token_positional_encoding": self.token_positional_encoding,
            "token_dropout": float(self.token_dropout_p),
            "transformer_layernorm": self.transformer_layernorm,
            "token_grid_size": list(self.token_grid_size),
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
    "DEFAULT_OMEGA_FEATURE_DIM",
    "TorchCNN",
    "CNN",
    "load_checkpoint_state",
    "resolve_checkpoint_runtime_config",
    "resolve_runtime_config_from_state",
]
