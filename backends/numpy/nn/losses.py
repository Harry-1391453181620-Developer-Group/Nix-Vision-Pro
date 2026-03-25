"""Loss functions and their backward passes. NumPy only."""

import numpy as np


def _per_sample_class_weight(
    labels_onehot: np.ndarray,
    class_weights: np.ndarray | None,
) -> np.ndarray:
    """Map class-wise weights to per-sample weights via one-hot labels."""
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
    """
    Cross-entropy loss: -sum(labels_onehot * log(softmax(logits))).
    Numerically stable: log_softmax(logits) then dot with labels.

    Args:
        logits: (N, num_classes), unnormalized scores.
        labels_onehot: (N, num_classes), one-hot encoded labels.
        class_weights: (num_classes,), optional class balancing weights.

    Returns:
        Scalar loss (mean over batch).
    """
    N = logits.shape[0]
    x_max = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - x_max), axis=1)) + x_max.squeeze(axis=1)
    log_probs = logits - log_sum_exp[:, np.newaxis]
    per_sample_nll = -np.sum(labels_onehot * log_probs, axis=1)
    sample_weights = _per_sample_class_weight(labels_onehot, class_weights)
    normalizer = np.sum(sample_weights) + 1e-12
    return float(np.sum(sample_weights * per_sample_nll) / normalizer)


def cross_entropy_loss_backward(
    logits: np.ndarray,
    labels_onehot: np.ndarray,
    class_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Gradient of cross-entropy w.r.t. logits: (softmax(logits) - labels_onehot) / N.

    Args:
        logits: (N, num_classes).
        labels_onehot: (N, num_classes).
        class_weights: (num_classes,), optional class balancing weights.

    Returns:
        d_logits: (N, num_classes), gradient w.r.t. logits.
    """
    x_max = np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    d_logits = softmax - labels_onehot
    sample_weights = _per_sample_class_weight(labels_onehot, class_weights)
    normalizer = np.sum(sample_weights) + 1e-12
    d_logits *= sample_weights[:, np.newaxis] / normalizer
    return d_logits
