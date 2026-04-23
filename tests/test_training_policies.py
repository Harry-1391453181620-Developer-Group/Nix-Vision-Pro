"""Focused tests for RandAugment, EMA, schedules, and backbone freeze policy."""

from argparse import Namespace
import random

import numpy as np
import pytest

from backends.numpy.model import CNN
from utils.training import (
    ModelEMA,
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
from backends.torch.train_backend import (
    _apply_backbone_freeze_state,
    _apply_batch_mix_torch,
    _build_optimizer,
    _collate_image_batch,
    _make_grad_scaler,
    _make_target_distribution,
    _make_worker_init_fn,
    _resolve_amp_dtype,
    _resolve_num_workers,
)


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


def test_numpy_model_ema_tracks_parameters_and_running_stats():
    model = CNN(input_size=(32, 32), num_classes=8, seed=7)
    ema_model = CNN(input_size=(32, 32), num_classes=8, seed=9)
    ema = ModelEMA(ema_model, decay=0.5, phase_warmup_steps=0)
    ema.sync_from(model)

    initial_weight = ema.ema_model.conv1.W.copy()
    initial_running_mean = ema.ema_model.bn1.running_mean.copy()

    model.train()
    model.forward(np.random.randn(4, 32, 32, 3).astype(np.float64))
    model.conv1.W += 0.25
    ema.update(model, step_in_phase=10, mix_active=False)

    np.testing.assert_allclose(
        ema.ema_model.conv1.W,
        0.5 * initial_weight + 0.5 * model.conv1.W,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ema.ema_model.bn1.running_mean,
        0.5 * initial_running_mean + 0.5 * model.bn1.running_mean,
        atol=1e-10,
    )
    assert ema.num_updates == 1


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


def test_torch_model_ema_tracks_parameters_and_bn_buffers():
    model = TorchCNN(input_size=(32, 32), num_classes=8, seed=11)
    ema_model = TorchCNN(input_size=(32, 32), num_classes=8, seed=13)
    ema = ModelEMA(ema_model, decay=0.5, phase_warmup_steps=0)
    ema.sync_from(model)

    initial_weight = ema.ema_model.conv1.weight.detach().clone()
    initial_running_mean = ema.ema_model.bn1.running_mean.detach().clone()

    model.train()
    _ = model(torch.randn(4, 32, 32, 3))
    with torch.no_grad():
        model.conv1.weight.add_(0.25)
    ema.update(model, step_in_phase=10, mix_active=False)

    assert torch.allclose(
        ema.ema_model.conv1.weight.detach(),
        0.5 * initial_weight + 0.5 * model.conv1.weight.detach(),
        atol=1e-6,
    )
    assert torch.allclose(
        ema.ema_model.bn1.running_mean.detach(),
        0.5 * initial_running_mean + 0.5 * model.bn1.running_mean.detach(),
        atol=1e-6,
    )
    assert int(ema.ema_model.bn1.num_batches_tracked.item()) == int(model.bn1.num_batches_tracked.item())
    assert ema.num_updates == 1


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


def test_torch_collate_outputs_contiguous_tensors():
    batch = [
        (torch.randn(3, 8, 8, dtype=torch.float32), 1),
        (torch.randn(3, 8, 8, dtype=torch.float32), 2),
    ]

    images, labels = _collate_image_batch(batch)

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert images.is_contiguous()
    assert labels.is_contiguous()
    assert tuple(images.shape) == (2, 3, 8, 8)
    assert tuple(labels.tolist()) == (1, 2)


def test_torch_worker_init_fn_reseeds_numpy_and_random_per_worker():
    init_worker = _make_worker_init_fn(1234)

    init_worker(0)
    worker_zero_int = int(np.random.randint(0, 1_000_000))
    worker_zero_float = random.random()

    init_worker(0)
    worker_zero_int_repeat = int(np.random.randint(0, 1_000_000))
    worker_zero_float_repeat = random.random()

    init_worker(1)
    worker_one_int = int(np.random.randint(0, 1_000_000))
    worker_one_float = random.random()

    assert worker_zero_int == worker_zero_int_repeat
    assert worker_zero_float == pytest.approx(worker_zero_float_repeat)
    assert worker_one_int != worker_zero_int
    assert worker_one_float != worker_zero_float


def test_torch_num_worker_defaults_follow_streaming_mode():
    assert _resolve_num_workers(True, None) == 4
    assert _resolve_num_workers(False, None) == 0
    assert _resolve_num_workers(True, 2) == 2
    with pytest.raises(SystemExit):
        _resolve_num_workers(True, -1)


def test_torch_batch_mix_preserves_soft_label_simplex():
    x = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    targets = _make_target_distribution(y, num_classes=4, label_smoothing=0.0, dtype=torch.float32)

    mixed_x, mixed_targets, mix_mode, lam = _apply_batch_mix_torch(
        x,
        targets,
        prob=1.0,
        beta_alpha=0.2,
        cutmix_ratio=0.0,
    )

    assert mix_mode == "mixup"
    assert tuple(mixed_x.shape) == tuple(x.shape)
    assert tuple(mixed_targets.shape) == tuple(targets.shape)
    assert 0.0 <= float(lam) <= 1.0
    assert mixed_targets.dtype == torch.float32
    assert torch.allclose(mixed_targets.sum(dim=1), torch.ones((4,), dtype=torch.float32), atol=1e-6)


def test_amp_mode_and_grad_scaler_policy():
    cpu_device = torch.device("cpu")
    assert _resolve_amp_dtype(cpu_device, "off") is None
    assert _resolve_amp_dtype(cpu_device, "auto") is None
    with pytest.raises(SystemExit):
        _resolve_amp_dtype(cpu_device, "on")

    cpu_scaler = _make_grad_scaler(cpu_device, None)
    assert not cpu_scaler.is_enabled()

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        amp_dtype = _resolve_amp_dtype(cuda_device, "auto")
        assert amp_dtype in {torch.float16, torch.bfloat16}
        cuda_scaler = _make_grad_scaler(cuda_device, amp_dtype)
        assert cuda_scaler.is_enabled() is (amp_dtype == torch.float16)
