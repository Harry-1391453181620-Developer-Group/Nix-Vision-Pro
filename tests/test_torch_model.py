"""Focused smoke tests for the PyTorch backend."""

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from backends.torch.model import TorchCNN


def test_torch_cnn_forward_shape():
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=123)
    x = torch.randn(4, 32, 32, 3)
    logits = model(x)
    assert tuple(logits.shape) == (4, 10)


def test_torch_cnn_checkpoint_round_trip():
    x = torch.randn(2, 32, 32, 3)
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=7)
    model.train()
    _ = model(x)
    _ = model(x * 0.5 + 0.25)
    model.eval()
    logits_before = model(x).detach().cpu().numpy()

    checkpoint = Path('.worktmp') / 'model_test_torch.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint)

    restored = TorchCNN(input_size=(32, 32), num_classes=10, seed=999)
    restored.load_weights(checkpoint)
    restored.eval()
    logits_after = restored(x).detach().cpu().numpy()
    np.testing.assert_allclose(logits_before, logits_after, atol=1e-6)
