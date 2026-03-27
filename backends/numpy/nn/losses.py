"""Loss functions and their backward passes. NumPy only.

This module now supports both weighted cross entropy and focal loss using a
single soft-label formulation. The generalized form lets the NumPy backend keep
label smoothing while still computing stable gradients, and it also keeps the
loss normalization logic aligned with the PyTorch backend.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def _softmax_and_log_probs(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return numerically stable softmax probabilities and log-probabilities."""
    x_max = np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    log_probs = logits - (np.log(np.sum(exp_x, axis=1, keepdims=True)) + x_max)
    return softmax, log_probs


def _per_sample_class_weight(
    labels_onehot: np.ndarray,
    class_weights: np.ndarray | None,
) -> np.ndarray:
    """Map class-wise weights to per-sample weights via one-hot or soft labels."""
    if class_weights is None:
        return np.ones((labels_onehot.shape[0],), dtype=np.float64)
    weights = np.asarray(class_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != labels_onehot.shape[1]:
        raise ValueError(
            f"class_weights length must match num_classes ({labels_onehot.shape[1]}), got {weights.shape[0]}"
        )
    return labels_onehot @ weights


def cross_entropy_loss(
    logits: np.ndarray,
    labels_onehot: np.ndarray,
    class_weights: np.ndarray | None = None,
) -> float:
    """Compute weighted soft-label cross entropy and return the batch mean."""
    _, log_probs = _softmax_and_log_probs(logits)
    per_sample_nll = -np.sum(labels_onehot * log_probs, axis=1)
    sample_weights = _per_sample_class_weight(labels_onehot, class_weights)
    normalizer = np.sum(sample_weights) + EPS
    return float(np.sum(sample_weights * per_sample_nll) / normalizer)


def cross_entropy_loss_backward(
    logits: np.ndarray,
    labels_onehot: np.ndarray,
    class_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the gradient of weighted soft-label cross entropy w.r.t. logits."""
    softmax, _ = _softmax_and_log_probs(logits)
    d_logits = softmax - labels_onehot
    sample_weights = _per_sample_class_weight(labels_onehot, class_weights)
    normalizer = np.sum(sample_weights) + EPS
    d_logits *= sample_weights[:, np.newaxis] / normalizer
    return d_logits


def focal_loss(
    logits: np.ndarray,
    labels_onehot: np.ndarray,
    gamma: float = 1.5,
    alpha: np.ndarray | None = None,
) -> float:
    """Compute focal loss using a soft-label generalization.

    The implementation uses:
    - `pt = sum(y * p)`
    - `ce = -sum(y * log(p))`
    - `loss = alpha_t * (1 - pt)^gamma * ce`

    This form supports standard one-hot labels, label-smoothed labels, and other
    soft labels, while callers can still choose to disable focal loss for MixUp
    at the policy layer.
    """
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")
    probs, log_probs = _softmax_and_log_probs(logits)
    pt = np.sum(labels_onehot * probs, axis=1)
    ce = -np.sum(labels_onehot * log_probs, axis=1)
    focal_factor = np.power(np.clip(1.0 - pt, 0.0, 1.0), gamma)
    sample_alpha = _per_sample_class_weight(labels_onehot, alpha)
    normalizer = np.sum(sample_alpha) + EPS
    return float(np.sum(sample_alpha * focal_factor * ce) / normalizer)


def focal_loss_backward(
    logits: np.ndarray,
    labels_onehot: np.ndarray,
    gamma: float = 1.5,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Return the focal-loss gradient for one-hot or soft labels.

    For soft labels, `pt` is the expectation of the model probability under the
    target distribution. The derivative is expanded analytically so the NumPy
    backend can train focal loss without relying on autodiff.
    """
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")

    probs, log_probs = _softmax_and_log_probs(logits)
    ce = -np.sum(labels_onehot * log_probs, axis=1)
    pt = np.sum(labels_onehot * probs, axis=1)
    one_minus_pt = np.clip(1.0 - pt, 0.0, 1.0)
    focal_factor = np.power(one_minus_pt, gamma)
    sample_alpha = _per_sample_class_weight(labels_onehot, alpha)
    normalizer = np.sum(sample_alpha) + EPS

    # The first term is the usual cross-entropy gradient scaled by the focal factor.
    d_logits = focal_factor[:, np.newaxis] * (probs - labels_onehot)

    if gamma > 0.0:
        # The second term accounts for how the focal factor changes as pt changes.
        safe_one_minus_pt = np.clip(one_minus_pt, EPS, None)
        focal_adjust = (
            gamma
            * np.power(safe_one_minus_pt, gamma - 1.0)
            * ce
        )
        d_pt = probs * (labels_onehot - pt[:, np.newaxis])
        d_logits -= focal_adjust[:, np.newaxis] * d_pt

    d_logits *= sample_alpha[:, np.newaxis] / normalizer
    return d_logits
