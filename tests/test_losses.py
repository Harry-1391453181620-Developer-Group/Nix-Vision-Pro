"""Unit tests for nn losses."""

import numpy as np
import pytest

from nn.losses import cross_entropy_loss, cross_entropy_loss_backward, focal_loss, focal_loss_backward


def test_cross_entropy_loss_non_negative():
    logits = np.random.randn(4, 5).astype(np.float64)
    labels = np.zeros((4, 5))
    labels[np.arange(4), np.random.randint(0, 5, 4)] = 1.0
    loss = cross_entropy_loss(logits, labels)
    assert loss >= 0


def test_cross_entropy_loss_backward_shape():
    logits = np.random.randn(3, 5).astype(np.float64)
    labels = np.zeros((3, 5))
    labels[np.arange(3), [0, 1, 2]] = 1.0
    dlogits = cross_entropy_loss_backward(logits, labels)
    assert dlogits.shape == logits.shape


def test_cross_entropy_loss_with_class_weights():
    logits = np.random.randn(6, 4).astype(np.float64)
    labels = np.zeros((6, 4), dtype=np.float64)
    labels[np.arange(6), [0, 0, 1, 1, 2, 3]] = 1.0
    class_weights = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float64)
    loss = cross_entropy_loss(logits, labels, class_weights=class_weights)
    assert loss >= 0.0


def test_cross_entropy_backward_with_class_weights_shape():
    logits = np.random.randn(5, 3).astype(np.float64)
    labels = np.zeros((5, 3), dtype=np.float64)
    labels[np.arange(5), [0, 1, 2, 0, 1]] = 1.0
    class_weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dlogits = cross_entropy_loss_backward(logits, labels, class_weights=class_weights)
    assert dlogits.shape == logits.shape


def test_focal_loss_with_gamma_zero_matches_cross_entropy():
    logits = np.random.randn(4, 5).astype(np.float64)
    labels = np.zeros((4, 5), dtype=np.float64)
    labels[np.arange(4), [0, 1, 2, 3]] = 1.0
    ce = cross_entropy_loss(logits, labels)
    focal = focal_loss(logits, labels, gamma=0.0)
    assert focal == pytest.approx(ce, rel=1e-10)


def test_focal_loss_backward_shape():
    logits = np.random.randn(3, 4).astype(np.float64)
    labels = np.zeros((3, 4), dtype=np.float64)
    labels[np.arange(3), [0, 1, 2]] = 1.0
    dlogits = focal_loss_backward(logits, labels, gamma=1.5)
    assert dlogits.shape == logits.shape
