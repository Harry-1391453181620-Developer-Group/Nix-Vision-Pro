"""Tests for shared MAOL training utilities."""

from __future__ import annotations

import numpy as np
import pytest

from utils.training import (
    ModelEMA,
    augment_batch,
    build_epoch_phase_map,
    compute_effective_learning_rate,
    compute_phase_learning_rate,
    parse_bool_flag,
    validate_augmentation_args,
    validate_ema_decay,
)


def test_augmentation_is_deterministic() -> None:
    x = np.linspace(0.0, 1.0, num=2 * 32 * 32 * 3, dtype=np.float32).reshape(2, 32, 32, 3)
    a = augment_batch(x, np.random.default_rng(1234))
    b = augment_batch(x, np.random.default_rng(1234))
    np.testing.assert_allclose(a, b, atol=1e-7)


def test_phase_map_and_schedules() -> None:
    phase_map = build_epoch_phase_map(5, 2)
    assert [item.phase_index for item in phase_map] == [0, 0, 0, 1, 1]
    assert [item.epoch_index_in_phase for item in phase_map] == [0, 1, 2, 0, 1]
    warmup_lr = compute_phase_learning_rate(
        base_lr=0.002,
        schedule="cosine",
        min_lr_ratio=0.2,
        gamma=0.5,
        step_size=10,
        warmup_epochs=1,
        epoch_index_in_phase=0,
        epochs_in_phase=3,
        batch_index=0,
        num_batches=2,
    )
    base_lr = compute_phase_learning_rate(
        base_lr=0.002,
        schedule="cosine",
        min_lr_ratio=0.2,
        gamma=0.5,
        step_size=10,
        warmup_epochs=1,
        epoch_index_in_phase=0,
        epochs_in_phase=3,
        batch_index=1,
        num_batches=2,
    )
    assert warmup_lr < base_lr
    assert compute_effective_learning_rate(0.00005, 0.0002, 0.002, 0.1) == pytest.approx(0.0002)


def test_ema_copies_skipped_statistics_exactly() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))
            self.register_buffer("stat", torch.zeros(2))

    model = Tiny()
    ema_model = Tiny()
    ema = ModelEMA(ema_model, decay=0.5, skip_prefixes=("stat",))
    ema.sync_from(model)
    with torch.no_grad():
        model.weight.add_(2.0)
        model.stat.add_(3.0)
    ema.update(model, step_in_phase=10)
    assert torch.allclose(ema.ema_model.weight, torch.ones(2))
    assert torch.allclose(ema.ema_model.stat, torch.full((2,), 3.0))


def test_validators_and_bool_parser() -> None:
    config = validate_augmentation_args(12.0, 0.2, 0.3, 0.4)
    assert config.rotation == pytest.approx(12.0)
    assert validate_ema_decay(0.999) == pytest.approx(0.999)
    assert parse_bool_flag("false") is False
    with pytest.raises(ValueError):
        validate_augmentation_args(180.0, 0.2, 0.2, 0.2)
