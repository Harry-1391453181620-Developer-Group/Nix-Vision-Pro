"""Unit tests for nn layers and activations."""

import numpy as np
import pytest

from nn.activations import relu, relu_backward, softmax
from nn.layers import Conv2D, Dense, Dropout, GlobalAveragePool2D, MaxPool2D


def test_relu_forward():
    x = np.array([[-1.0, 0.0, 1.0]])
    out = relu(x)
    np.testing.assert_allclose(out, [[0.0, 0.0, 1.0]])


def test_relu_backward():
    dout = np.ones((2, 3))
    x = np.array([[1.0, -1.0, 0.0], [0.0, 2.0, -2.0]])
    dx = relu_backward(dout, x)
    np.testing.assert_allclose(dx, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_softmax_sums_to_one():
    x = np.random.randn(4, 5)
    out = softmax(x, axis=-1)
    np.testing.assert_allclose(np.sum(out, axis=1), np.ones(4))


def test_dense_forward_shape():
    layer = Dense(10, 7)
    x = np.random.randn(3, 10).astype(np.float64)
    out = layer.forward(x)
    assert out.shape == (3, 7)


def test_dense_backward_shape():
    layer = Dense(10, 7)
    x = np.random.randn(3, 10).astype(np.float64)
    layer.forward(x)
    dout = np.random.randn(3, 7).astype(np.float64)
    dx = layer.backward(dout)
    assert dx.shape == (3, 10)
    assert layer._dW.shape == (10, 7)
    assert layer._db.shape == (7,)


def test_conv2d_forward_shape():
    layer = Conv2D(3, 8, (3, 3), stride=1, padding=1)
    x = np.random.randn(2, 8, 8, 3).astype(np.float64)
    out = layer.forward(x)
    assert out.shape == (2, 8, 8, 8)


def test_conv2d_backward_shape():
    layer = Conv2D(3, 8, (3, 3), stride=1, padding=1)
    x = np.random.randn(2, 8, 8, 3).astype(np.float64)
    layer.forward(x)
    dout = np.random.randn(2, 8, 8, 8).astype(np.float64)
    dx = layer.backward(dout)
    assert dx.shape == x.shape
    assert layer._dW.shape == layer.W.shape
    assert layer._db.shape == layer.b.shape


def test_maxpool2d_forward_shape():
    pool = MaxPool2D((2, 2), stride=2)
    x = np.random.randn(2, 4, 6, 3).astype(np.float64)
    out = pool.forward(x)
    assert out.shape == (2, 2, 3, 3)


def test_maxpool2d_backward_shape():
    pool = MaxPool2D((2, 2), stride=2)
    x = np.random.randn(2, 4, 6, 3).astype(np.float64)
    pool.forward(x)
    dout = np.random.randn(2, 2, 3, 3).astype(np.float64)
    dx = pool.backward(dout)
    assert dx.shape == x.shape


def test_global_average_pool2d_shapes():
    gap = GlobalAveragePool2D()
    x = np.random.randn(3, 5, 7, 4).astype(np.float64)
    out = gap.forward(x)
    assert out.shape == (3, 4)
    dout = np.random.randn(3, 4).astype(np.float64)
    dx = gap.backward(dout)
    assert dx.shape == x.shape


def test_dropout_train_eval_behavior():
    np.random.seed(0)
    dropout = Dropout(p=0.5)
    x = np.ones((4, 6), dtype=np.float64)
    train_out = dropout.forward(x, training=True)
    eval_out = dropout.forward(x, training=False)
    assert train_out.shape == x.shape
    assert np.any(train_out == 0.0)
    np.testing.assert_allclose(eval_out, x)
