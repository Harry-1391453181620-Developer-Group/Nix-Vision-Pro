"""Focused smoke tests for the PyTorch backend."""

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from backends.torch.model import TorchCNN, load_checkpoint_state, resolve_checkpoint_runtime_config
from backends.torch.predict_backend import _validate_checkpoint_overrides


def test_torch_cnn_forward_shape():
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=123)
    x = torch.randn(4, 32, 32, 3)
    logits = model(x)
    assert tuple(logits.shape) == (4, 10)


def test_torch_cnn_checkpoint_round_trip_preserves_metadata():
    x = torch.randn(2, 32, 32, 3)
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=7)
    model.train()
    _ = model(x)
    _ = model(x * 0.5 + 0.25)
    model.eval()
    logits_before = model(x).detach().cpu().numpy()

    checkpoint = Path('.worktmp') / 'model_test_torch.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint, metadata={"is_ema": True, "ema_decay": 0.999})

    state, metadata = load_checkpoint_state(checkpoint, map_location='cpu')
    assert metadata["backend"] == "torch"
    assert metadata["checkpoint_version"] == 2
    assert metadata["is_ema"] is True
    assert metadata["ema_decay"] == pytest.approx(0.999)
    assert "conv1.weight" in state

    restored = TorchCNN(input_size=(32, 32), num_classes=10, seed=999)
    returned_metadata = restored.load_weights(checkpoint)
    assert returned_metadata["is_ema"] is True
    restored.eval()
    logits_after = restored(x).detach().cpu().numpy()
    np.testing.assert_allclose(logits_before, logits_after, atol=1e-6)


def test_torch_cnn_loads_legacy_state_dict_checkpoint():
    x = torch.randn(2, 32, 32, 3)
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=17)
    checkpoint = Path('.worktmp') / 'model_test_torch_legacy.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)

    restored = TorchCNN(input_size=(32, 32), num_classes=10, seed=19)
    metadata = restored.load_weights(checkpoint)
    model.eval()
    restored.eval()
    assert metadata == {}
    np.testing.assert_allclose(
        model(x).detach().cpu().numpy(),
        restored(x).detach().cpu().numpy(),
        atol=1e-6,
    )


def test_torch_checkpoint_runtime_config_round_trip_from_structured_checkpoint():
    checkpoint = Path('.worktmp') / 'model_test_torch_runtime_config.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    class_names = [f'class_{index}' for index in range(13)]
    model = TorchCNN(input_size=(32, 32), num_classes=13, seed=29, width_scale=1.5)
    model.save_weights(checkpoint, metadata={"class_names": class_names})

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location='cpu')

    assert runtime_config.num_classes == 13
    assert runtime_config.width_scale == pytest.approx(1.5)
    assert runtime_config.stage2_channels == model.stage2_channels
    assert runtime_config.input_size == (32, 32)
    assert runtime_config.class_names == class_names


def test_torch_checkpoint_runtime_config_infers_legacy_architecture():
    checkpoint = Path('.worktmp') / 'model_test_torch_runtime_config_legacy.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = TorchCNN(input_size=(32, 32), num_classes=13, seed=31, width_scale=1.0)
    torch.save(model.state_dict(), checkpoint)

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location='cpu')

    assert runtime_config.num_classes == 13
    assert runtime_config.width_scale == pytest.approx(1.0)
    assert runtime_config.stage2_channels == 64
    assert runtime_config.input_size == (32, 32)
    assert runtime_config.class_names is None


def test_torch_predict_override_validation_rejects_checkpoint_conflicts():
    with pytest.raises(SystemExit, match="class-count"):
        _validate_checkpoint_overrides(
            class_count_override=12,
            width_scale_override=None,
            checkpoint_num_classes=13,
            checkpoint_width_scale=1.0,
        )

    with pytest.raises(SystemExit, match="model-width-scale"):
        _validate_checkpoint_overrides(
            class_count_override=None,
            width_scale_override=0.75,
            checkpoint_num_classes=13,
            checkpoint_width_scale=1.0,
        )
