"""Unit tests for optimizers."""

import numpy as np

from nn.optimizers import AdamW, SGD


def test_sgd_updates_params():
    param = np.array([1.0, -2.0], dtype=np.float64)
    grad = np.array([0.5, -0.5], dtype=np.float64)
    optimizer = SGD([(param, grad)], lr=0.1, momentum=0.9, nesterov=True)
    before = param.copy()
    optimizer.step()
    assert not np.allclose(before, param)


def test_adamw_updates_params():
    param = np.array([1.0, -2.0], dtype=np.float64)
    grad = np.array([0.1, -0.2], dtype=np.float64)
    optimizer = AdamW([(param, grad)], lr=0.01, weight_decay=0.0)
    before = param.copy()
    optimizer.step()
    assert not np.allclose(before, param)
