"""Unit tests for the legacy NumPy CNN integration behavior."""

from pathlib import Path

import numpy as np

from backends.numpy.model import CNN


def test_cnn_forward_shape():
    model = CNN(input_size=(32, 32), num_classes=10, seed=123)
    model.eval()
    x = np.random.randn(4, 32, 32, 3).astype(np.float64)
    logits = model.forward(x)
    assert logits.shape == (4, 10)


def test_cnn_checkpoint_round_trip():
    x = np.random.randn(2, 32, 32, 3).astype(np.float64)
    model = CNN(input_size=(32, 32), num_classes=10, seed=7)
    model.train()
    _ = model.forward(x)
    _ = model.forward(x * 0.5 + 0.25)
    model.eval()
    logits_before = model.forward(x)

    checkpoint = Path('.worktmp') / 'model_test_numpy.npz'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint)

    restored = CNN(input_size=(32, 32), num_classes=10, seed=999)
    restored.load_weights(checkpoint)
    restored.eval()
    logits_after = restored.forward(x)
    np.testing.assert_allclose(logits_before, logits_after, atol=1e-10)
