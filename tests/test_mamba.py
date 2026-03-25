"""Unit tests for Mamba block (selective SSM)."""

import numpy as np
import pytest

from nn.mamba import MambaBlock


def test_mamba_forward_shape():
    block = MambaBlock(in_dim=64, state_dim=16, seed=42)
    N, L, D = 2, 16, 64
    x = np.random.randn(N, L, D).astype(np.float64)
    out = block.forward(x)
    assert out.shape == (N, L, D)


def test_mamba_backward_shape():
    block = MambaBlock(in_dim=64, state_dim=16, seed=42)
    N, L, D = 2, 16, 64
    x = np.random.randn(N, L, D).astype(np.float64)
    block.forward(x)
    dout = np.random.randn(N, L, D).astype(np.float64)
    dx = block.backward(dout)
    assert dx.shape == (N, L, D)


def test_mamba_get_params():
    block = MambaBlock(in_dim=8, state_dim=4, seed=1)
    params = block.get_params()
    assert len(params) == 6
    for param, grad in params:
        assert param is not None
        assert param.dtype == np.float64


def test_mamba_invalid_dims_raises():
    with pytest.raises(ValueError):
        MambaBlock(in_dim=0, state_dim=4)
    with pytest.raises(ValueError):
        MambaBlock(in_dim=4, state_dim=0)
