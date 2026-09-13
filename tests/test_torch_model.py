"""Focused smoke tests for the PyTorch backend."""

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from backends.torch.model import (
    ARCHITECTURE_LEGACY_CNN,
    ARCHITECTURE_LEGACY_TOKEN,
    ARCHITECTURE_VIT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_TOKEN_DIM,
    DEFAULT_VIT_DEPTH,
    TorchCNN,
    TorchLegacyCNN,
    load_checkpoint_state,
    resolve_checkpoint_runtime_config,
)
from backends.torch.predict_backend import _validate_checkpoint_overrides


def _vit_from_runtime_config(runtime_config):
    return TorchCNN(
        input_size=runtime_config.input_size,
        num_classes=runtime_config.num_classes,
        omega_enabled=runtime_config.omega_enabled,
        omega_projector_depth=runtime_config.omega_projector_depth or 1,
        omega_hidden_dim=runtime_config.omega_hidden_dim or DEFAULT_TOKEN_DIM,
        tokenize=runtime_config.tokenize,
        token_dim=runtime_config.token_dim or DEFAULT_TOKEN_DIM,
        transformer_depth=runtime_config.transformer_depth or runtime_config.vit_depth or DEFAULT_VIT_DEPTH,
        attention_heads=runtime_config.attention_heads or 4,
        transformer_mlp_ratio=runtime_config.transformer_mlp_ratio or 2.0,
        token_pool=runtime_config.token_pool or "cls",
        token_positional_encoding=runtime_config.token_positional_encoding or "learned",
        token_dropout=runtime_config.token_dropout if runtime_config.token_dropout is not None else 0.1,
        transformer_layernorm=runtime_config.transformer_layernorm or "pre",
        patch_size=runtime_config.patch_size or DEFAULT_PATCH_SIZE,
        pool_type=runtime_config.pool_type or runtime_config.token_pool or "cls",
    )


def test_torch_vit_forward_shape():
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=123)
    x = torch.randn(4, 32, 32, 3)
    logits = model(x)

    assert tuple(logits.shape) == (4, 10)
    assert model.architecture == ARCHITECTURE_VIT
    assert model.patch_size == DEFAULT_PATCH_SIZE
    assert model.vit_depth == DEFAULT_VIT_DEPTH
    assert model.token_dim == DEFAULT_TOKEN_DIM
    assert model.pool_type == "cls"
    assert not hasattr(model, "conv1")


def test_torch_vit_forward_with_cls_omega_shape():
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=123, omega_enabled=True)
    x = torch.randn(4, 32, 32, 3)
    logits, h = model.forward_with_representation(x)
    omega_logits, omega_h, t_h = model.forward_with_omega(x)

    assert tuple(logits.shape) == (4, 10)
    assert tuple(h.shape) == (4, DEFAULT_TOKEN_DIM)
    assert tuple(omega_logits.shape) == (4, 10)
    assert tuple(omega_h.shape) == (4, DEFAULT_TOKEN_DIM)
    assert tuple(t_h.shape) == (4, DEFAULT_TOKEN_DIM)


def test_torch_vit_layer_idsi_forward_uses_patch_and_transformer_names():
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=123, omega_enabled=True)
    x = torch.randn(4, 32, 32, 3)

    logits, h, t_h, layer_inputs, layer_outputs, metric_tokens = model.forward_with_omega_and_layer_idsi(x)

    assert model.idsi_layer_names == (
        "patch_embedding",
        "transformer_block_1",
        "transformer_block_2",
        "transformer_block_3",
        "transformer_block_4",
    )
    assert tuple(logits.shape) == (4, 10)
    assert tuple(h.shape) == (4, DEFAULT_TOKEN_DIM)
    assert tuple(t_h.shape) == (4, DEFAULT_TOKEN_DIM)
    assert tuple(metric_tokens.shape) == (4, 16, DEFAULT_TOKEN_DIM)
    assert len(layer_inputs) == len(model.idsi_layer_names)
    assert len(layer_outputs) == len(model.idsi_layer_names)
    assert tuple(layer_inputs[0].shape) == (4, 16, DEFAULT_TOKEN_DIM)
    assert tuple(layer_outputs[0].shape) == (4, 16, DEFAULT_TOKEN_DIM)
    for layer_input, layer_output in zip(layer_inputs[1:], layer_outputs[1:]):
        assert tuple(layer_input.shape) == tuple(layer_output.shape) == (4, 17, DEFAULT_TOKEN_DIM)


def test_torch_vit_token_diversity_excludes_cls_token():
    model = TorchCNN(
        input_size=(32, 32),
        num_classes=10,
        seed=123,
        transformer_depth=1,
        token_dim=128,
        attention_heads=4,
    )
    x = torch.randn(4, 32, 32, 3)

    logits, h, t_h, layer_inputs, layer_outputs, metric_tokens = (
        model.forward_with_token_dynamics_and_layer_idsi(x)
    )

    assert model.idsi_layer_names == ("patch_embedding", "transformer_block_1")
    assert tuple(logits.shape) == (4, 10)
    assert tuple(h.shape) == (4, 128)
    assert tuple(t_h.shape) == (4, 128)
    assert tuple(metric_tokens.shape) == (4, 16, 128)
    assert tuple(layer_outputs[-1].shape) == (4, 17, 128)


def test_torch_vit_checkpoint_round_trip_preserves_metadata(tmp_path: Path):
    x = torch.randn(2, 32, 32, 3)
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=7)
    model.train()
    _ = model(x)
    _ = model(x * 0.5 + 0.25)
    model.eval()
    logits_before = model(x).detach().cpu().numpy()

    checkpoint = tmp_path / "model_test_torch.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint, metadata={"is_ema": True, "ema_decay": 0.999})

    state, metadata = load_checkpoint_state(checkpoint, map_location="cpu")
    assert metadata["backend"] == "torch"
    assert metadata["checkpoint_version"] == 3
    assert metadata["architecture"] == ARCHITECTURE_VIT
    assert metadata["patch_size"] == DEFAULT_PATCH_SIZE
    assert metadata["vit_depth"] == DEFAULT_VIT_DEPTH
    assert metadata["token_dim"] == DEFAULT_TOKEN_DIM
    assert metadata["attention_heads"] == 4
    assert metadata["pool_type"] == "cls"
    assert metadata["is_ema"] is True
    assert metadata["ema_decay"] == pytest.approx(0.999)
    assert "patch_projection.weight" in state
    assert "cls_token" in state
    assert "conv1.weight" not in state

    restored = TorchCNN(input_size=(32, 32), num_classes=10, seed=999)
    returned_metadata = restored.load_weights(checkpoint)
    assert returned_metadata["is_ema"] is True
    restored.eval()
    logits_after = restored(x).detach().cpu().numpy()
    np.testing.assert_allclose(logits_before, logits_after, atol=1e-6)


def test_torch_vit_loads_raw_state_dict_checkpoint(tmp_path: Path):
    x = torch.randn(2, 32, 32, 3)
    model = TorchCNN(input_size=(32, 32), num_classes=10, seed=17)
    checkpoint = tmp_path / "model_test_torch_raw.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location="cpu")
    restored = _vit_from_runtime_config(runtime_config)
    metadata = restored.load_weights(checkpoint)
    model.eval()
    restored.eval()
    assert metadata == {}
    assert runtime_config.architecture == ARCHITECTURE_VIT
    np.testing.assert_allclose(
        model(x).detach().cpu().numpy(),
        restored(x).detach().cpu().numpy(),
        atol=1e-6,
    )


def test_torch_checkpoint_runtime_config_round_trip_from_structured_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "model_test_torch_runtime_config.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    class_names = [f"class_{index}" for index in range(13)]
    model = TorchCNN(input_size=(32, 32), num_classes=13, seed=29)
    model.save_weights(checkpoint, metadata={"class_names": class_names})

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location="cpu")

    assert runtime_config.architecture == ARCHITECTURE_VIT
    assert runtime_config.num_classes == 13
    assert runtime_config.width_scale == pytest.approx(0.0)
    assert runtime_config.stage2_channels == 0
    assert runtime_config.input_size == (32, 32)
    assert runtime_config.class_names == tuple(class_names)
    assert runtime_config.omega_enabled is False
    assert runtime_config.patch_size == DEFAULT_PATCH_SIZE
    assert runtime_config.vit_depth == DEFAULT_VIT_DEPTH
    assert runtime_config.token_dim == DEFAULT_TOKEN_DIM
    assert runtime_config.pool_type == "cls"


def test_torch_checkpoint_runtime_config_round_trip_with_omega_metadata(tmp_path: Path):
    checkpoint = tmp_path / "model_test_torch_runtime_config_omega.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = TorchCNN(
        input_size=(32, 32),
        num_classes=13,
        seed=29,
        omega_enabled=True,
        omega_projector_depth=2,
        omega_hidden_dim=128,
    )
    model.save_weights(checkpoint)

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location="cpu")
    restored = _vit_from_runtime_config(runtime_config)
    metadata = restored.load_weights(checkpoint, map_location="cpu")

    assert runtime_config.architecture == ARCHITECTURE_VIT
    assert runtime_config.omega_enabled is True
    assert runtime_config.omega_projector_depth == 2
    assert runtime_config.omega_hidden_dim == 128
    assert metadata["omega_enabled"] is True


def test_torch_checkpoint_runtime_config_round_trip_with_vit_token_metadata(tmp_path: Path):
    checkpoint = tmp_path / "model_test_torch_runtime_config_token.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = TorchCNN(
        input_size=(32, 32),
        num_classes=13,
        seed=29,
        token_dim=128,
        transformer_depth=2,
        attention_heads=4,
        transformer_mlp_ratio=2.0,
        token_pool="mean",
        token_positional_encoding="learned",
        token_dropout=0.1,
        transformer_layernorm="pre",
    )
    model.save_weights(checkpoint)

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location="cpu")
    restored = _vit_from_runtime_config(runtime_config)
    metadata = restored.load_weights(checkpoint, map_location="cpu")

    assert runtime_config.architecture == ARCHITECTURE_VIT
    assert runtime_config.tokenize is True
    assert runtime_config.token_dim == 128
    assert runtime_config.transformer_depth == 2
    assert runtime_config.vit_depth == 2
    assert runtime_config.attention_heads == 4
    assert runtime_config.transformer_mlp_ratio == pytest.approx(2.0)
    assert runtime_config.token_pool == "mean"
    assert runtime_config.pool_type == "mean"
    assert metadata["tokenize"] is True


def test_torch_checkpoint_runtime_config_infers_legacy_cnn_architecture(tmp_path: Path):
    checkpoint = tmp_path / "model_test_torch_runtime_config_legacy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = TorchLegacyCNN(input_size=(32, 32), num_classes=13, seed=31, width_scale=1.0)
    torch.save(model.state_dict(), checkpoint)

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location="cpu")

    assert runtime_config.architecture == ARCHITECTURE_LEGACY_CNN
    assert runtime_config.num_classes == 13
    assert runtime_config.width_scale == pytest.approx(1.0)
    assert runtime_config.stage2_channels == 64
    assert runtime_config.input_size == (32, 32)
    assert runtime_config.class_names == ()
    assert runtime_config.omega_enabled is False


def test_torch_phase2_token_checkpoint_reconstructs_legacy_model(tmp_path: Path):
    checkpoint = tmp_path / "model_test_torch_phase2_token.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    legacy = TorchLegacyCNN(
        input_size=(32, 32),
        num_classes=13,
        seed=31,
        width_scale=0.75,
        tokenize=True,
        token_dim=128,
        transformer_depth=1,
        attention_heads=4,
        transformer_mlp_ratio=2.0,
        token_pool="mean",
        token_positional_encoding="learned",
        token_dropout=0.1,
        transformer_layernorm="pre",
    )
    legacy.save_weights(checkpoint)

    runtime_config = resolve_checkpoint_runtime_config(checkpoint, map_location="cpu")
    restored = TorchLegacyCNN(
        input_size=runtime_config.input_size,
        num_classes=runtime_config.num_classes,
        width_scale=runtime_config.width_scale,
        tokenize=runtime_config.tokenize,
        token_dim=runtime_config.token_dim or 128,
        transformer_depth=runtime_config.transformer_depth or 1,
        attention_heads=runtime_config.attention_heads or 4,
        transformer_mlp_ratio=runtime_config.transformer_mlp_ratio or 2.0,
        token_pool=runtime_config.token_pool or "mean",
        token_positional_encoding=runtime_config.token_positional_encoding or "learned",
        token_dropout=runtime_config.token_dropout if runtime_config.token_dropout is not None else 0.1,
        transformer_layernorm=runtime_config.transformer_layernorm or "pre",
    )
    metadata = restored.load_weights(checkpoint, map_location="cpu")

    assert runtime_config.architecture == ARCHITECTURE_LEGACY_TOKEN
    assert runtime_config.tokenize is True
    assert runtime_config.width_scale == pytest.approx(0.75)
    assert metadata["architecture"] == ARCHITECTURE_LEGACY_TOKEN


def test_torch_predict_override_validation_rejects_checkpoint_conflicts():
    with pytest.raises(SystemExit, match="class-count"):
        _validate_checkpoint_overrides(
            checkpoint_architecture=ARCHITECTURE_VIT,
            class_count_override=12,
            width_scale_override=None,
            checkpoint_num_classes=13,
            checkpoint_width_scale=0.0,
        )

    _validate_checkpoint_overrides(
        checkpoint_architecture=ARCHITECTURE_VIT,
        class_count_override=None,
        width_scale_override=0.75,
        checkpoint_num_classes=13,
        checkpoint_width_scale=0.0,
    )

    with pytest.raises(SystemExit, match="model-width-scale"):
        _validate_checkpoint_overrides(
            checkpoint_architecture=ARCHITECTURE_LEGACY_CNN,
            class_count_override=None,
            width_scale_override=0.75,
            checkpoint_num_classes=13,
            checkpoint_width_scale=1.0,
        )
