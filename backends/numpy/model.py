"""CNN + SE classifier used by the active NumPy backend.

The stage-2 width is now configurable through a width scale so both training and
inference can intentionally choose a smaller intermediate representation without
hard-coding a single channel count into every entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from nn.activations import relu, relu_backward
from nn.layers import BatchNorm2D, Conv2D, Dense, Dropout, MaxPool2D, SqueezeExcitation


def load_checkpoint_state(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load either a legacy raw `.npz` state_dict or a structured checkpoint."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.suffix != ".npz":
        raise ValueError("Checkpoint must be .npz")
    try:
        data = np.load(checkpoint, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Invalid checkpoint file: {exc}") from exc
    try:
        is_structured = "__meta__" in data.files or any(key.startswith("model.") for key in data.files)
        if not is_structured:
            return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}, {}

        metadata: dict[str, Any] = {}
        if "__meta__" in data.files:
            meta_raw = data["__meta__"]
            meta_text = str(meta_raw.item())
            if meta_text:
                metadata = dict(json.loads(meta_text))
        state = {
            key[len("model."):]: np.asarray(data[key], dtype=np.float64)
            for key in data.files
            if key.startswith("model.")
        }
        if not state:
            raise ValueError("Structured checkpoint missing `model.` weights")
        return state, metadata
    finally:
        data.close()


def _resolve_stage2_channels(width_scale: float) -> int:
    """Convert a width multiplier into a safe integer stage-2 channel count."""
    width_scale = float(width_scale)
    if width_scale <= 0.0:
        raise ValueError("width_scale must be > 0")
    return max(8, int(round(64 * width_scale)))


class CNN:
    """
    Three-stage CNN + SE classifier for image classification.

    Input: (N, H, W, 3). Output: (N, num_classes).
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_classes: int,
        seed: int | None = None,
        dropout_p: float = 0.5,
        width_scale: float = 0.75,
    ):
        if seed is not None:
            np.random.seed(seed)
        height, width = input_size
        if height < 8 or width < 8:
            raise ValueError("input_size must be at least (8, 8)")

        stage2_channels = _resolve_stage2_channels(width_scale)

        self.conv1 = Conv2D(3, 32, (3, 3), stride=1, padding=1)
        self.bn1 = BatchNorm2D(32)
        self.conv2 = Conv2D(32, 32, (3, 3), stride=1, padding=1)
        self.bn2 = BatchNorm2D(32)
        self.se1 = SqueezeExcitation(32, reduction=4)
        self.pool1 = MaxPool2D((2, 2), stride=2)

        # Stage 2 is the width-scaled stage that now defaults to 48 channels.
        self.conv3 = Conv2D(32, stage2_channels, (3, 3), stride=1, padding=1)
        self.bn3 = BatchNorm2D(stage2_channels)
        self.conv4 = Conv2D(stage2_channels, stage2_channels, (3, 3), stride=1, padding=1)
        self.bn4 = BatchNorm2D(stage2_channels)
        self.se2 = SqueezeExcitation(stage2_channels, reduction=4)
        self.pool2 = MaxPool2D((2, 2), stride=2)

        # Stage 3 keeps its output width stable so the dense head shape is unchanged.
        self.conv5 = Conv2D(stage2_channels, 128, (3, 3), stride=1, padding=1)
        self.bn5 = BatchNorm2D(128)
        self.conv6 = Conv2D(128, 128, (3, 3), stride=1, padding=1)
        self.bn6 = BatchNorm2D(128)
        self.se3 = SqueezeExcitation(128, reduction=4)
        self.pool3 = MaxPool2D((2, 2), stride=2)

        feat_h, feat_w = height // 8, width // 8
        self.fc1 = Dense(feat_h * feat_w * 128, 256)
        self.dropout = Dropout(p=dropout_p)
        self.fc2 = Dense(256, num_classes)

        self._input_size = input_size
        self._num_classes = num_classes
        self._training = True
        self._feature_shape = None
        self._backbone_frozen = False
        self._freeze_bn_affine = True
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

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def set_backbone_frozen(self, frozen: bool, freeze_bn_affine: bool = True) -> None:
        """Store backbone-freeze policy so forward/backward honor BN freeze semantics."""
        self._backbone_frozen = bool(frozen)
        self._freeze_bn_affine = bool(freeze_bn_affine)

    def _backbone_bn_training(self) -> bool:
        """Backbone BN layers stop updating running stats while the backbone is frozen."""
        return bool(self._training and not self._backbone_frozen)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, H, W, 3). Returns logits (N, num_classes)."""
        backbone_bn_training = self._backbone_bn_training()

        x = self.conv1.forward(x)
        x = self.bn1.forward(x, training=backbone_bn_training)
        self._pre_relu1 = x
        x = relu(x)
        x = self.conv2.forward(x)
        x = self.bn2.forward(x, training=backbone_bn_training)
        self._pre_relu2 = x
        x = relu(x)
        x = self.se1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv3.forward(x)
        x = self.bn3.forward(x, training=backbone_bn_training)
        self._pre_relu3 = x
        x = relu(x)
        x = self.conv4.forward(x)
        x = self.bn4.forward(x, training=backbone_bn_training)
        self._pre_relu4 = x
        x = relu(x)
        x = self.se2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv5.forward(x)
        x = self.bn5.forward(x, training=backbone_bn_training)
        self._pre_relu5 = x
        x = relu(x)
        x = self.conv6.forward(x)
        x = self.bn6.forward(x, training=backbone_bn_training)
        self._pre_relu6 = x
        x = relu(x)
        x = self.se3.forward(x)
        x = self.pool3.forward(x)

        self._feature_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        self._fc1_pre = x
        x = relu(x)
        x = self.dropout.forward(x, training=self._training)
        logits = self.fc2.forward(x)
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        dx = self.fc2.backward(dlogits)
        dx = self.dropout.backward(dx)
        dx = relu_backward(dx, self._fc1_pre)
        dx = self.fc1.backward(dx)
        dx = dx.reshape(self._feature_shape)

        dx = self.pool3.backward(dx)
        dx = self.se3.backward(dx)
        dx = relu_backward(dx, self._pre_relu6)
        dx = self.bn6.backward(dx)
        dx = self.conv6.backward(dx)
        dx = relu_backward(dx, self._pre_relu5)
        dx = self.bn5.backward(dx)
        dx = self.conv5.backward(dx)

        dx = self.pool2.backward(dx)
        dx = self.se2.backward(dx)
        dx = relu_backward(dx, self._pre_relu4)
        dx = self.bn4.backward(dx)
        dx = self.conv4.backward(dx)
        dx = relu_backward(dx, self._pre_relu3)
        dx = self.bn3.backward(dx)
        dx = self.conv3.backward(dx)

        dx = self.pool1.backward(dx)
        dx = self.se1.backward(dx)
        dx = relu_backward(dx, self._pre_relu2)
        dx = self.bn2.backward(dx)
        dx = self.conv2.backward(dx)
        dx = relu_backward(dx, self._pre_relu1)
        dx = self.bn1.backward(dx)
        self.conv1.backward(dx)

    def get_backbone_parameters(self, include_bn_affine: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return feature-extractor parameters, optionally excluding BN affine terms."""
        params: List[Tuple[np.ndarray, np.ndarray]] = []
        params.extend(self.conv1.get_params())
        if include_bn_affine:
            params.extend(self.bn1.get_params())
        params.extend(self.conv2.get_params())
        if include_bn_affine:
            params.extend(self.bn2.get_params())
        params.extend(self.se1.get_params())

        params.extend(self.conv3.get_params())
        if include_bn_affine:
            params.extend(self.bn3.get_params())
        params.extend(self.conv4.get_params())
        if include_bn_affine:
            params.extend(self.bn4.get_params())
        params.extend(self.se2.get_params())

        params.extend(self.conv5.get_params())
        if include_bn_affine:
            params.extend(self.bn5.get_params())
        params.extend(self.conv6.get_params())
        if include_bn_affine:
            params.extend(self.bn6.get_params())
        params.extend(self.se3.get_params())
        return params

    def get_backbone_bn_affine_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return only backbone BN affine parameters for adaptive-freeze mode."""
        params: List[Tuple[np.ndarray, np.ndarray]] = []
        for bn_layer in [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6]:
            params.extend(bn_layer.get_params())
        return params

    def get_head_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return classifier-head parameters that stay trainable during backbone freeze."""
        params: List[Tuple[np.ndarray, np.ndarray]] = []
        params.extend(self.fc1.get_params())
        params.extend(self.fc2.get_params())
        return params

    def get_trainable_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the active optimizer parameter list for the current freeze state."""
        if not self._backbone_frozen:
            return self.get_backbone_parameters(include_bn_affine=True) + self.get_head_parameters()
        params = self.get_head_parameters()
        if not self._freeze_bn_affine:
            params = self.get_backbone_bn_affine_parameters() + params
        return params

    def get_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return every parameter so checkpoints and legacy callers still see the full model."""
        return self.get_backbone_parameters(include_bn_affine=True) + self.get_head_parameters()

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "conv1.W": self.conv1.W,
            "conv1.b": self.conv1.b,
            "bn1.gamma": self.bn1.gamma,
            "bn1.beta": self.bn1.beta,
            "bn1.running_mean": self.bn1.running_mean,
            "bn1.running_var": self.bn1.running_var,
            "conv2.W": self.conv2.W,
            "conv2.b": self.conv2.b,
            "bn2.gamma": self.bn2.gamma,
            "bn2.beta": self.bn2.beta,
            "bn2.running_mean": self.bn2.running_mean,
            "bn2.running_var": self.bn2.running_var,
            "se1.fc1.W": self.se1.fc1.W,
            "se1.fc1.b": self.se1.fc1.b,
            "se1.fc2.W": self.se1.fc2.W,
            "se1.fc2.b": self.se1.fc2.b,
            "conv3.W": self.conv3.W,
            "conv3.b": self.conv3.b,
            "bn3.gamma": self.bn3.gamma,
            "bn3.beta": self.bn3.beta,
            "bn3.running_mean": self.bn3.running_mean,
            "bn3.running_var": self.bn3.running_var,
            "conv4.W": self.conv4.W,
            "conv4.b": self.conv4.b,
            "bn4.gamma": self.bn4.gamma,
            "bn4.beta": self.bn4.beta,
            "bn4.running_mean": self.bn4.running_mean,
            "bn4.running_var": self.bn4.running_var,
            "se2.fc1.W": self.se2.fc1.W,
            "se2.fc1.b": self.se2.fc1.b,
            "se2.fc2.W": self.se2.fc2.W,
            "se2.fc2.b": self.se2.fc2.b,
            "conv5.W": self.conv5.W,
            "conv5.b": self.conv5.b,
            "bn5.gamma": self.bn5.gamma,
            "bn5.beta": self.bn5.beta,
            "bn5.running_mean": self.bn5.running_mean,
            "bn5.running_var": self.bn5.running_var,
            "conv6.W": self.conv6.W,
            "conv6.b": self.conv6.b,
            "bn6.gamma": self.bn6.gamma,
            "bn6.beta": self.bn6.beta,
            "bn6.running_mean": self.bn6.running_mean,
            "bn6.running_var": self.bn6.running_var,
            "se3.fc1.W": self.se3.fc1.W,
            "se3.fc1.b": self.se3.fc1.b,
            "se3.fc2.W": self.se3.fc2.W,
            "se3.fc2.b": self.se3.fc2.b,
            "fc1.W": self.fc1.W,
            "fc1.b": self.fc1.b,
            "fc2.W": self.fc2.W,
            "fc2.b": self.fc2.b,
        }

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        expected = self.state_dict()
        missing = sorted(set(expected.keys()) - set(state.keys()))
        if missing:
            raise KeyError(f"Missing weights: {missing}")
        for key, target in expected.items():
            source = np.asarray(state[key], dtype=np.float64)
            if source.shape != target.shape:
                raise ValueError(f"Shape mismatch for {key}: expected {target.shape}, got {source.shape}")
            target[...] = source

    def save_weights(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        checkpoint = Path(path)
        if checkpoint.suffix != ".npz":
            raise ValueError("Checkpoint path must use .npz extension")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "__meta__": np.array(
                json.dumps({
                    "checkpoint_version": 2,
                    "backend": "numpy",
                    **({} if metadata is None else dict(metadata)),
                }),
                dtype=np.str_,
            )
        }
        for key, value in self.state_dict().items():
            payload[f"model.{key}"] = value
        np.savez(checkpoint, **payload)

    def load_weights(self, path: str | Path) -> dict[str, Any]:
        state, metadata = load_checkpoint_state(path)
        self.load_state_dict(state)
        return metadata
