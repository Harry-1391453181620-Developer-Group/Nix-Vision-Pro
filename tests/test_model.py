"""Unit tests for the legacy NumPy CNN integration behavior."""

from pathlib import Path

import numpy as np
import pytest

from backends.numpy.model import CNN, load_checkpoint_state, resolve_checkpoint_runtime_config
from backends.numpy.predict_backend import _validate_checkpoint_overrides


def test_cnn_forward_shape():
    model = CNN(input_size=(32, 32), num_classes=10, seed=123)
    model.eval()
    x = np.random.randn(4, 32, 32, 3).astype(np.float64)
    logits = model.forward(x)
    assert logits.shape == (4, 10)


def test_cnn_checkpoint_round_trip_preserves_metadata():
    x = np.random.randn(2, 32, 32, 3).astype(np.float64)
    model = CNN(input_size=(32, 32), num_classes=10, seed=7)
    model.train()
    _ = model.forward(x)
    _ = model.forward(x * 0.5 + 0.25)
    model.eval()
    logits_before = model.forward(x)

    checkpoint = Path('.worktmp') / 'model_test_numpy.npz'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint, metadata={"is_ema": True, "ema_decay": 0.999})

    state, metadata = load_checkpoint_state(checkpoint)
    assert metadata["backend"] == "numpy"
    assert metadata["checkpoint_version"] == 2
    assert metadata["is_ema"] is True
    assert metadata["ema_decay"] == pytest.approx(0.999)
    assert "conv1.W" in state

    restored = CNN(input_size=(32, 32), num_classes=10, seed=999)
    returned_metadata = restored.load_weights(checkpoint)
    assert returned_metadata["is_ema"] is True
    restored.eval()
    logits_after = restored.forward(x)
    np.testing.assert_allclose(logits_before, logits_after, atol=1e-10)


def test_cnn_loads_legacy_npz_checkpoint():
    model = CNN(input_size=(32, 32), num_classes=10, seed=21)
    checkpoint = Path('.worktmp') / 'model_test_numpy_legacy.npz'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, **model.state_dict())

    restored = CNN(input_size=(32, 32), num_classes=10, seed=22)
    metadata = restored.load_weights(checkpoint)
    assert metadata == {}
    for key, value in model.state_dict().items():
        np.testing.assert_allclose(value, restored.state_dict()[key], atol=1e-10)


def test_numpy_checkpoint_runtime_config_round_trip_from_structured_checkpoint():
    checkpoint = Path('.worktmp') / 'model_test_numpy_runtime_config.npz'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    class_names = [f'class_{index}' for index in range(13)]
    model = CNN(input_size=(32, 32), num_classes=13, seed=41, width_scale=1.5)
    model.save_weights(checkpoint, metadata={"class_names": class_names})

    runtime_config = resolve_checkpoint_runtime_config(checkpoint)

    assert runtime_config.num_classes == 13
    assert runtime_config.width_scale == pytest.approx(1.5)
    assert runtime_config.stage2_channels == model.stage2_channels
    assert runtime_config.input_size == (32, 32)
    assert runtime_config.class_names == tuple(class_names)


def test_numpy_checkpoint_runtime_config_infers_legacy_architecture():
    checkpoint = Path('.worktmp') / 'model_test_numpy_runtime_config_legacy.npz'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = CNN(input_size=(32, 32), num_classes=13, seed=43, width_scale=1.0)
    np.savez(checkpoint, **model.state_dict())

    runtime_config = resolve_checkpoint_runtime_config(checkpoint)

    assert runtime_config.num_classes == 13
    assert runtime_config.width_scale == pytest.approx(1.0)
    assert runtime_config.stage2_channels == 64
    assert runtime_config.input_size == (32, 32)
    assert runtime_config.class_names == ()


def test_numpy_predict_override_validation_rejects_checkpoint_conflicts():
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
