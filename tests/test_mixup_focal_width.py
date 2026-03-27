"""Focused tests for MixUp, focal-policy helpers, and width-scaled models."""

from pathlib import Path

import numpy as np
import pytest

from backends.numpy.model import CNN
from backends.numpy.train_backend import load_weights_forgiving as numpy_load_weights_forgiving
from utils.training import (
    _apply_random_erasing,
    apply_mixup,
    validate_augmentation_args,
    validate_focal_gamma,
    validate_mixup_probability,
    validate_model_width_scale,
)


torch = pytest.importorskip("torch")

from backends.torch.model import TorchCNN
from backends.torch.train_backend import load_weights_forgiving as torch_load_weights_forgiving


def test_validate_new_training_args():
    config = validate_augmentation_args(12.0, 0.2, 0.3, 0.4)
    assert config.rotation == pytest.approx(12.0)
    assert validate_mixup_probability(0.5) == pytest.approx(0.5)
    assert validate_focal_gamma(1.5) == pytest.approx(1.5)
    assert validate_model_width_scale(0.75) == pytest.approx(0.75)

    with pytest.raises(ValueError):
        validate_augmentation_args(180.0, 0.2, 0.2, 0.2)
    with pytest.raises(ValueError):
        validate_mixup_probability(1.5)
    with pytest.raises(ValueError):
        validate_focal_gamma(-0.1)
    with pytest.raises(ValueError):
        validate_model_width_scale(0.0)


def test_apply_mixup_preserves_soft_label_simplex():
    x = np.arange(4 * 2 * 2 * 1, dtype=np.float32).reshape(4, 2, 2, 1)
    y = np.eye(4, dtype=np.float32)
    mixed_x, mixed_y, active, lam = apply_mixup(x, y, np.random.default_rng(7), prob=1.0, beta_alpha=0.2)

    assert active
    assert 0.0 <= lam <= 1.0
    assert mixed_x.shape == x.shape
    assert mixed_y.shape == y.shape
    np.testing.assert_allclose(np.sum(mixed_y, axis=1), np.ones((4,), dtype=np.float32), atol=1e-6)
    assert not np.allclose(mixed_x, x)


def test_random_erasing_uses_image_mean_fill():
    image = np.linspace(0.0, 1.0, num=12 * 12 * 3, dtype=np.float32).reshape(12, 12, 3)
    mean_value = np.mean(image, axis=(0, 1))
    erased = _apply_random_erasing(image, np.random.default_rng(11))
    changed_mask = np.any(np.abs(erased - image) > 1e-6, axis=2)

    assert np.any(changed_mask)
    expected = np.broadcast_to(mean_value, erased[changed_mask].shape)
    np.testing.assert_allclose(erased[changed_mask], expected, atol=1e-6)


def test_width_scale_changes_stage_two_channels_for_both_backends():
    numpy_model = CNN(input_size=(32, 32), num_classes=8, seed=3, width_scale=0.75)
    torch_model = TorchCNN(input_size=(32, 32), num_classes=8, seed=3, width_scale=0.75)

    assert numpy_model.stage2_channels == 48
    assert torch_model.stage2_channels == 48
    assert numpy_model.conv3.W.shape[-1] == 48
    assert tuple(torch_model.conv3.weight.shape)[:2] == (48, 32)


def test_forgiving_checkpoint_load_skips_resized_tensors_for_torch():
    checkpoint = Path('.worktmp') / 'torch_width_scale_1_0.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    original = TorchCNN(input_size=(32, 32), num_classes=8, seed=5, width_scale=1.0)
    original.save_weights(checkpoint)

    resized = TorchCNN(input_size=(32, 32), num_classes=8, seed=5, width_scale=0.75)
    loaded, skipped = torch_load_weights_forgiving(resized, checkpoint)

    assert any(key.startswith('conv3.') for key in skipped)
    assert any(key.startswith('conv1.') for key in loaded)


def test_forgiving_checkpoint_load_skips_resized_tensors_for_numpy():
    checkpoint = Path('.worktmp') / 'numpy_width_scale_1_0.npz'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    original = CNN(input_size=(32, 32), num_classes=8, seed=5, width_scale=1.0)
    original.save_weights(checkpoint)

    resized = CNN(input_size=(32, 32), num_classes=8, seed=5, width_scale=0.75)
    loaded, skipped = numpy_load_weights_forgiving(resized, checkpoint)

    assert any(key.startswith('conv3.') for key in skipped)
    assert any(key.startswith('conv1.') for key in loaded)
