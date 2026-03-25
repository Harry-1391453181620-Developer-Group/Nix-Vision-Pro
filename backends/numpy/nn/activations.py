"""Activation functions and their backward passes. NumPy only."""

from typing import Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU: max(0, x). In-place safe for non-overlapping views.

    Args:
        x: ndarray of any shape.

    Returns:
        ndarray same shape, non-negative.
    """
    return np.maximum(0, x)


def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Backward of ReLU: gradient is dout where x > 0, else 0.

    Args:
        dout: gradient of loss w.r.t. ReLU output, same shape as x.
        x: input to ReLU (pre-activation).

    Returns:
        gradient of loss w.r.t. x, same shape.
    """
    return np.where(x > 0, dout, 0.0)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU: x * Phi(x). Approximate 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Better training dynamics than ReLU for modern vision models.
    """
    a = np.sqrt(2.0 / np.pi)
    b = 0.044715
    inner = a * (x + b * (x ** 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Backward of GELU (approximate). Gradient of loss w.r.t. pre-activation x.
    """
    a = np.sqrt(2.0 / np.pi)
    b = 0.044715
    inner = a * (x + b * (x ** 3))
    tanh_in = np.tanh(inner)
    d_inner = a * (1.0 + 3.0 * b * (x ** 2))
    dx = 0.5 * (1.0 + tanh_in) + 0.5 * x * (1.0 - tanh_in * tanh_in) * d_inner
    return dout * dx


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus: log(1 + exp(x)). Numerically stable for large |x|."""
    return np.where(x > 20.0, x, np.where(x < -20.0, 0.0, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)))


def softplus_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient of softplus: dout * sigmoid(x)."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
    return dout * sig


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax: exp(x - max) / sum(exp(x - max)) for numerical stability.

    Args:
        x: ndarray (e.g. logits).
        axis: axis over which to apply softmax (default last).

    Returns:
        ndarray same shape, sums to 1 along axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
