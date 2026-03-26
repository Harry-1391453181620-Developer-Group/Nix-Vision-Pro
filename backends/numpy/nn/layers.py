"""CNN layers: Conv2D, DepthwiseConv2D, LayerNorm, MaxPool2D, GlobalAveragePool2D, Dropout, Dense."""

from typing import Optional, Tuple

import numpy as np

from nn.activations import relu, relu_backward

def _im2col(
    x: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: int,
    pad: int,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Reshape input (N, H, W, C) into column matrix for convolution.
    Returns (col, out_shape) where out_shape is (N, out_H, out_W).
    """
    N, H, W, C = x.shape
    kH, kW = kernel_size
    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (W + 2 * pad - kW) // stride + 1
    col = np.zeros((N, out_H, out_W, kH, kW, C), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            col[:, :, :, i, j, :] = x_pad[:, i : i + out_H * stride : stride, j : j + out_W * stride : stride, :]
    return col, (N, out_H, out_W)


def _col2im(
    col: np.ndarray,
    x_shape: Tuple[int, ...],
    kernel_size: Tuple[int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    """Accumulate column gradient back into image layout (N, H, W, C)."""
    N, H, W, C = x_shape
    kH, kW = kernel_size
    out_H, out_W = col.shape[1], col.shape[2]
    dx_pad = np.zeros((N, H + 2 * pad, W + 2 * pad, C), dtype=col.dtype)
    for i in range(kH):
        for j in range(kW):
            dx_pad[:, i : i + out_H * stride : stride, j : j + out_W * stride : stride, :] += col[:, :, :, i, j, :]
    if pad == 0:
        return dx_pad
    return dx_pad[:, pad:-pad, pad:-pad, :]


class Conv2D:
    """
    2D convolution. Channel-last: input (N, H, W, in_C), output (N, out_H, out_W, out_C).
    Filters shape (kH, kW, in_C, out_C).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        weight_scale: float | None = None,
    ):
        if stride <= 0:
            raise ValueError("stride must be >= 1")
        if padding < 0:
            raise ValueError("padding must be >= 0")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kH, kW = kernel_size
        scale = np.sqrt(2.0 / (in_channels * kH * kW)) if weight_scale is None else weight_scale
        self.W = np.random.randn(kH, kW, in_channels, out_channels).astype(np.float64) * scale
        self.b = np.zeros((out_channels,), dtype=np.float64)
        self._x = None
        self._col = None
        self._out_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, H, W, in_C). Returns (N, out_H, out_W, out_C)."""
        self._x = x
        col, self._out_shape = _im2col(x, self.kernel_size, self.stride, self.padding)
        self._col = col
        N, out_H, out_W = self._out_shape
        # col (N, out_H, out_W, kH, kW, in_C), W (kH, kW, in_C, out_C)
        # output (N, out_H, out_W, out_C)
        out = np.tensordot(col, self.W, axes=([3, 4, 5], [0, 1, 2]))
        out += self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, out_H, out_W, out_C). Returns gradient w.r.t. x, same shape as x."""
        N, out_H, out_W, out_C = dout.shape
        kH, kW, in_C, _ = self.W.shape
        # dW: (kH, kW, in_C, out_C)
        dW = np.tensordot(self._col, dout, axes=([0, 1, 2], [0, 1, 2]))
        self._dW = dW
        self._db = np.sum(dout, axis=(0, 1, 2))
        # dx: col gradient then col2im
        # dout (N, out_H, out_W, out_C), W (kH, kW, in_C, out_C) -> dcol (N, out_H, out_W, kH, kW, in_C)
        dcol = np.tensordot(dout, self.W, axes=([3], [3]))  # (N, out_H, out_W, kH, kW, in_C)
        dx = _col2im(dcol, self._x.shape, self.kernel_size, self.stride, self.padding)
        return dx

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return [(W, dW), (b, db)] for optimizer."""
        return [(self.W, getattr(self, "_dW", None)), (self.b, getattr(self, "_db", None))]


class MaxPool2D:
    """Max pooling. Channel-last. No overlap by default (stride = pool_size)."""

    def __init__(self, pool_size: Tuple[int, int] = (2, 2), stride: Optional[int] = None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        if self.stride <= 0:
            raise ValueError("stride must be >= 1")
        self._x = None
        self._argmax = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, H, W, C). Returns (N, out_H, out_W, C)."""
        self._x = x
        N, H, W, C = x.shape
        pH, pW = self.pool_size
        s = self.stride
        if H < pH or W < pW:
            raise ValueError("pool_size must not exceed spatial dimensions")
        out_H = (H - pH) // s + 1
        out_W = (W - pW) // s + 1
        out = np.zeros((N, out_H, out_W, C), dtype=x.dtype)
        argmax = np.zeros((N, out_H, out_W, C), dtype=np.intp)
        for i in range(out_H):
            for j in range(out_W):
                patch = x[:, i * s : i * s + pH, j * s : j * s + pW, :]  # (N, pH, pW, C)
                out[:, i, j, :] = np.max(patch, axis=(1, 2))
                argmax[:, i, j, :] = np.argmax(patch.reshape(N, -1, C), axis=1)
        self._argmax = argmax
        self._out_H, self._out_W = out_H, out_W
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, out_H, out_W, C). Returns gradient w.r.t. x."""
        N, H, W, C = self._x.shape
        pH, pW = self.pool_size
        s = self.stride
        out_H, out_W = self._out_H, self._out_W
        dx = np.zeros_like(self._x)
        for i in range(out_H):
            for j in range(out_W):
                for c in range(C):
                    for n in range(N):
                        idx = self._argmax[n, i, j, c]
                        hi, wi = np.unravel_index(idx, (pH, pW))
                        dx[n, i * s + hi, j * s + wi, c] += dout[n, i, j, c]
        return dx


class GlobalAveragePool2D:
    """Global average pooling over spatial dimensions H and W."""

    def __init__(self):
        self._x_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, H, W, C). Returns (N, C)."""
        self._x_shape = x.shape
        return np.mean(x, axis=(1, 2))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, C). Returns (N, H, W, C)."""
        if self._x_shape is None:
            raise RuntimeError("GlobalAveragePool2D.backward called before forward")
        N, H, W, C = self._x_shape
        scale = 1.0 / (H * W)
        return np.broadcast_to(dout[:, np.newaxis, np.newaxis, :] * scale, self._x_shape).copy()

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class Dropout:
    """Inverted dropout for regularization during training."""

    def __init__(self, p: float = 0.5):
        if not (0.0 <= p < 1.0):
            raise ValueError("p must satisfy 0 <= p < 1")
        self.p = p
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout only when training=True."""
        if not training or self.p == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.p
        self._mask = (np.random.rand(*x.shape) < keep_prob).astype(x.dtype) / keep_prob
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return dout
        return dout * self._mask

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class SqueezeExcitation:
    """
    Squeeze-and-Excitation block for channel-last feature maps.

    The block uses global average pooling followed by a two-layer bottleneck MLP
    to generate per-channel gates in [0, 1].
    """

    def __init__(self, channels: int, reduction: int = 4):
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        hidden = max(1, channels // reduction)
        self.channels = int(channels)
        self.gap = GlobalAveragePool2D()
        self.fc1 = Dense(self.channels, hidden)
        self.fc2 = Dense(hidden, self.channels)
        self._x = None
        self._z1 = None
        self._scale = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        pooled = self.gap.forward(x)
        self._z1 = self.fc1.forward(pooled)
        hidden = relu(self._z1)
        z2 = self.fc2.forward(hidden)
        self._scale = 1.0 / (1.0 + np.exp(-np.clip(z2, -20.0, 20.0)))
        return x * self._scale[:, np.newaxis, np.newaxis, :]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        scale = self._scale[:, np.newaxis, np.newaxis, :]
        dx = dout * scale
        dscale = np.sum(dout * self._x, axis=(1, 2))
        dz2 = dscale * self._scale * (1.0 - self._scale)
        dhidden = self.fc2.backward(dz2)
        dz1 = relu_backward(dhidden, self._z1)
        dpooled = self.fc1.backward(dz1)
        dx += self.gap.backward(dpooled)
        return dx

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params: list[tuple[np.ndarray, np.ndarray]] = []
        params.extend(self.fc1.get_params())
        params.extend(self.fc2.get_params())
        return params


class BatchNorm2D:
    """
    Batch Normalization for 2D feature maps with channel-last layout (N, H, W, C).
    Uses running mean/var during eval and batch stats during training.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if not (0.0 < momentum < 1.0):
            raise ValueError("momentum must be in (0,1)")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = np.ones((self.num_features,), dtype=np.float64)
        self.beta = np.zeros((self.num_features,), dtype=np.float64)
        self.running_mean = np.zeros((self.num_features,), dtype=np.float64)
        self.running_var = np.ones((self.num_features,), dtype=np.float64)
        self._x_hat = None
        self._inv_std = None
        self._batch_mean = None
        self._training = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: (N, H, W, C). Returns same shape."""
        C = x.shape[-1]
        if C != self.num_features:
            raise ValueError(f"BatchNorm2D expected C={self.num_features}, got {C}")
        self._training = bool(training)
        if training:
            # Batch statistics across N,H,W during normal training.
            mean = np.mean(x, axis=(0, 1, 2))
            var = np.var(x, axis=(0, 1, 2))
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x - mean) * inv_std
            self._x_hat = x_hat
            self._inv_std = inv_std
            self._batch_mean = mean
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            # Eval-mode normalization is also used during backbone freeze so BN
            # running statistics stay fixed while affine parameters may still adapt.
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (x - self.running_mean) * inv_std
            self._x_hat = x_hat
            self._inv_std = inv_std
            self._batch_mean = self.running_mean
        return x_hat * self.gamma + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, H, W, C). Returns dx of same shape."""
        if self._x_hat is None or self._inv_std is None:
            raise RuntimeError("BatchNorm2D.backward called before forward")
        if not self._training:
            # During frozen-backbone training, running stats stay fixed but
            # affine parameters may remain trainable when requested.
            self._dgamma = np.sum(dout * self._x_hat, axis=(0, 1, 2))
            self._dbeta = np.sum(dout, axis=(0, 1, 2))
            return dout * (self.gamma * self._inv_std)
        N, H, W, C = dout.shape
        M = float(N * H * W)
        x_hat = self._x_hat
        inv_std = self._inv_std
        self._dgamma = np.sum(dout * x_hat, axis=(0, 1, 2))
        self._dbeta = np.sum(dout, axis=(0, 1, 2))
        dx_hat = dout * self.gamma
        dx = (1.0 / M) * inv_std * (
            M * dx_hat
            - np.sum(dx_hat, axis=(0, 1, 2), keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=(0, 1, 2), keepdims=True)
        )
        return dx

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.gamma, getattr(self, "_dgamma", None)), (self.beta, getattr(self, "_dbeta", None))]


class DepthwiseConv2D:
    """
    Depthwise 2D convolution: one filter per channel. Channel-last.
    Input (N, H, W, C), kernel (kH, kW, C) -> output (N, out_H, out_W, C).
    Enables large-kernel spatial modeling with minimal parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        weight_scale: float | None = None,
    ):
        if stride <= 0:
            raise ValueError("stride must be >= 1")
        if padding < 0:
            raise ValueError("padding must be >= 0")
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kH, kW = kernel_size
        scale = np.sqrt(2.0 / (channels * kH * kW)) if weight_scale is None else weight_scale
        self.W = np.random.randn(kH, kW, channels).astype(np.float64) * scale
        self.b = np.zeros((channels,), dtype=np.float64)
        self._x = None
        self._col = None
        self._out_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, H, W, C). Returns (N, out_H, out_W, C)."""
        self._x = x
        col, self._out_shape = _im2col(x, self.kernel_size, self.stride, self.padding)
        self._col = col
        # col (N, out_H, out_W, kH, kW, C), W (kH, kW, C) -> sum over (kH, kW)
        out = np.sum(col * self.W, axis=(3, 4))
        out += self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, out_H, out_W, C). Returns gradient w.r.t. x."""
        # dW: (kH, kW, C)
        self._dW = np.sum(self._col * dout[:, :, :, np.newaxis, np.newaxis, :], axis=(0, 1, 2))
        self._db = np.sum(dout, axis=(0, 1, 2))
        # dcol (N, out_H, out_W, kH, kW, C) = dout broadcast * W
        dcol = dout[:, :, :, np.newaxis, np.newaxis, :] * self.W
        return _col2im(dcol, self._x.shape, self.kernel_size, self.stride, self.padding)

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, getattr(self, "_dW", None)), (self.b, getattr(self, "_db", None))]


class LayerNorm:
    """
    Layer normalization over the last axis (channels for N,H,W,C).
    out = gamma * (x - mean) / sqrt(var + eps) + beta.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones((num_features,), dtype=np.float64)
        self.beta = np.zeros((num_features,), dtype=np.float64)
        self._x = None
        self._mean = None
        self._var = None
        self._x_norm = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., C). Normalize over last axis. Returns same shape."""
        self._x = x
        self._mean = np.mean(x, axis=-1, keepdims=True)
        self._var = np.var(x, axis=-1, keepdims=True) + self.eps
        self._x_norm = (x - self._mean) * np.reciprocal(np.sqrt(self._var))
        return self.gamma * self._x_norm + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: same shape as forward output. Returns gradient w.r.t. x."""
        dx_norm = dout * self.gamma
        sigma = np.sqrt(self._var)
        mean_dx = np.mean(dx_norm, axis=-1, keepdims=True)
        mean_dx_xn = np.mean(dx_norm * self._x_norm, axis=-1, keepdims=True)
        dx = (dx_norm - mean_dx - self._x_norm * mean_dx_xn) * np.reciprocal(sigma)
        self._dgamma = np.sum(dout * self._x_norm, axis=tuple(range(dout.ndim - 1)))
        self._dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
        return dx

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.gamma, getattr(self, "_dgamma", None)), (self.beta, getattr(self, "_dbeta", None))]


class Dense:
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int):
        scale = 1.0 / np.sqrt(in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float64) * scale
        self.b = np.zeros((out_features,), dtype=np.float64)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, in_features). Returns (N, out_features)."""
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, out_features). Returns (N, in_features)."""
        self._dW = self._x.T @ dout
        self._db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, getattr(self, "_dW", None)), (self.b, getattr(self, "_db", None))]
