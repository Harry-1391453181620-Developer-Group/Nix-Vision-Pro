"""Focused tests for augmentation, phase schedules, and backbone freeze policy."""

from argparse import Namespace

import numpy as np
import pytest

from backends.numpy.model import CNN
from utils.training import (
    adjust_phase_lr_offset_after_unfreeze,
    augment_batch,
    build_epoch_phase_map,
    compute_effective_learning_rate,
    compute_phase_learning_rate,
    validate_freeze_cycle_args,
    validate_phase_learning_rates,
)


torch = pytest.importorskip("torch")
from torch import nn

from backends.torch.model import TorchCNN
from backends.torch.train_backend import _apply_backbone_freeze_state, _build_optimizer


def test_augment_batch_is_deterministic_for_same_seed():
    x = np.linspace(0.0, 1.0, num=2 * 32 * 32 * 3, dtype=np.float32).reshape(2, 32, 32, 3)
    out_a = augment_batch(x, np.random.default_rng(1234))
    out_b = augment_batch(x, np.random.default_rng(1234))
    np.testing.assert_allclose(out_a, out_b, atol=1e-7)


def test_phase_schedule_validation_and_warmup_restart_shape():
    phase_map = build_epoch_phase_map(5, 2)
    assert [config.phase_index for config in phase_map] == [0, 0, 0, 1, 1]
    assert [config.epoch_index_in_phase for config in phase_map] == [0, 1, 2, 0, 1]

    validate_phase_learning_rates([0.002, 0.0005], 2)
    with pytest.raises(ValueError):
        validate_phase_learning_rates([0.0005, 0.002], 2)

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
    end_lr = compute_phase_learning_rate(
        base_lr=0.002,
        schedule="cosine",
        min_lr_ratio=0.2,
        gamma=0.5,
        step_size=10,
        warmup_epochs=1,
        epoch_index_in_phase=2,
        epochs_in_phase=3,
        batch_index=1,
        num_batches=2,
    )
    phase_restart_lr = compute_phase_learning_rate(
        base_lr=0.0005,
        schedule="cosine",
        min_lr_ratio=0.2,
        gamma=0.5,
        step_size=10,
        warmup_epochs=1,
        epoch_index_in_phase=0,
        epochs_in_phase=2,
        batch_index=1,
        num_batches=2,
    )

    assert warmup_lr < base_lr
    assert end_lr < base_lr
    assert phase_restart_lr == pytest.approx(0.0005, rel=1e-6)


def test_effective_lr_clamps_to_phase_floor():
    effective_lr = compute_effective_learning_rate(
        scheduled_lr=0.00005,
        phase_lr_offset=0.0002,
        phase_base_lr=0.002,
        min_lr_ratio=0.1,
    )
    assert effective_lr == pytest.approx(0.0002)


def test_freeze_cycle_validation_and_post_unfreeze_offset_rules():
    freeze_patience, freeze_epoch_num, after_unfreeze_lr_change = validate_freeze_cycle_args(8, 10, 1e-4)
    assert freeze_patience == 8
    assert freeze_epoch_num == 10
    assert after_unfreeze_lr_change == pytest.approx(1e-4)

    with pytest.raises(ValueError):
        validate_freeze_cycle_args(0, 10, 1e-4)
    with pytest.raises(ValueError):
        validate_freeze_cycle_args(8, 0, 1e-4)
    with pytest.raises(ValueError):
        validate_freeze_cycle_args(8, 10, -1e-4)

    new_offset, applied = adjust_phase_lr_offset_after_unfreeze(
        current_effective_lr=0.0018,
        phase_lr_offset=0.0,
        after_unfreeze_lr_change=0.0001,
        phase_base_lr=0.002,
        min_lr_ratio=0.1,
        next_phase_start_lr=0.0005,
    )
    assert applied
    assert new_offset == pytest.approx(0.0001)

    # EPS guard: this should still pass even though the candidate is smaller than
    # the next phase LR by a tiny floating-point residue.
    new_offset, applied = adjust_phase_lr_offset_after_unfreeze(
        current_effective_lr=0.0006,
        phase_lr_offset=0.0,
        after_unfreeze_lr_change=0.0001000000000005,
        phase_base_lr=0.002,
        min_lr_ratio=0.1,
        next_phase_start_lr=0.0005,
    )
    assert applied
    assert new_offset == pytest.approx(0.0001000000000005)

    # Do not deduct when the effective LR is already too close to the floor.
    same_offset, applied = adjust_phase_lr_offset_after_unfreeze(
        current_effective_lr=0.00039,
        phase_lr_offset=0.0001,
        after_unfreeze_lr_change=0.0001,
        phase_base_lr=0.002,
        min_lr_ratio=0.1,
        next_phase_start_lr=0.0005,
    )
    assert not applied
    assert same_offset == pytest.approx(0.0001)

    # Cap cumulative offset so it never undercuts the next phase LR.
    capped_offset, applied = adjust_phase_lr_offset_after_unfreeze(
        current_effective_lr=0.0017,
        phase_lr_offset=0.00145,
        after_unfreeze_lr_change=0.0002,
        phase_base_lr=0.002,
        min_lr_ratio=0.1,
        next_phase_start_lr=0.0005,
    )
    assert applied
    assert capped_offset == pytest.approx(0.0015)

    # Final phase cap falls back to the scheduler floor.
    final_offset, applied = adjust_phase_lr_offset_after_unfreeze(
        current_effective_lr=0.0012,
        phase_lr_offset=0.0016,
        after_unfreeze_lr_change=0.0004,
        phase_base_lr=0.002,
        min_lr_ratio=0.1,
        next_phase_start_lr=None,
    )
    assert applied
    assert final_offset == pytest.approx(0.0018)


def test_numpy_freeze_keeps_bn_running_stats_fixed():
    model = CNN(input_size=(32, 32), num_classes=8, seed=7)
    x_a = np.random.randn(4, 32, 32, 3).astype(np.float64)
    x_b = np.random.randn(4, 32, 32, 3).astype(np.float64)

    model.train()
    model.set_backbone_frozen(False, freeze_bn_affine=True)
    model.forward(x_a)
    running_mean_before = model.bn1.running_mean.copy()

    model.train()
    model.set_backbone_frozen(True, freeze_bn_affine=True)
    model.forward(x_b)
    np.testing.assert_allclose(model.bn1.running_mean, running_mean_before, atol=1e-12)


def test_torch_freeze_transition_updates_expected_params_and_bn_modes():
    model = TorchCNN(input_size=(32, 32), num_classes=8, seed=11)
    args = Namespace(optimizer="adamw", momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 32, 32, 3)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=True)
    optimizer_full = _build_optimizer(args, model, lr_value=1e-3)
    conv_before = model.conv1.weight.detach().clone()
    optimizer_full.zero_grad(set_to_none=True)
    criterion(model(x), y).backward()
    optimizer_full.step()
    assert not torch.allclose(model.conv1.weight.detach(), conv_before)
    assert model.bn1.training

    model.train()
    _apply_backbone_freeze_state(model, backbone_frozen=True, freeze_bn_affine=True)
    optimizer_head = _build_optimizer(args, model, lr_value=1e-3)
    conv_before_freeze = model.conv1.weight.detach().clone()
    fc_before_freeze = model.fc1.weight.detach().clone()
    running_mean_before = model.bn1.running_mean.detach().clone()
    optimizer_head.zero_grad(set_to_none=True)
    criterion(model(x), y).backward()
    optimizer_head.step()
    assert optimizer_head is not optimizer_full
    assert torch.allclose(model.conv1.weight.detach(), conv_before_freeze)
    assert not torch.allclose(model.fc1.weight.detach(), fc_before_freeze)
    assert torch.allclose(model.bn1.running_mean.detach(), running_mean_before)
    assert not model.bn1.training

    model.train()
    _apply_backbone_freeze_state(model, backbone_frozen=False, freeze_bn_affine=True)
    optimizer_full_again = _build_optimizer(args, model, lr_value=1e-3)
    conv_before_unfreeze = model.conv1.weight.detach().clone()
    optimizer_full_again.zero_grad(set_to_none=True)
    criterion(model(x), y).backward()
    optimizer_full_again.step()
    assert optimizer_full_again is not optimizer_head
    assert not torch.allclose(model.conv1.weight.detach(), conv_before_unfreeze)
    assert model.bn1.training

    model.train()
    _apply_backbone_freeze_state(model, backbone_frozen=True, freeze_bn_affine=True)
    optimizer_head_again = _build_optimizer(args, model, lr_value=1e-3)
    assert optimizer_head_again is not optimizer_full_again
    assert not model.conv1.weight.requires_grad
    assert model.fc1.weight.requires_grad
    assert not model.bn1.training
