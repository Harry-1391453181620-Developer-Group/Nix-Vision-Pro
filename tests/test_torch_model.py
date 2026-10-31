"""Tests for CNN + MAOL checkpoint integration."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from backends.torch.model import TorchCNN, load_checkpoint_state, resolve_checkpoint_runtime_config
from maol.tree import ClassTree
from tests.conftest import workspace_tmp_path


def _tree() -> ClassTree:
    return ClassTree.from_leaf_paths(["animal/cat", "animal/dog", "plant"])


def test_maol_model_forward_and_checkpoint_round_trip() -> None:
    tree = _tree()
    model = TorchCNN(
        input_size=(64, 64),
        num_classes=3,
        tree=tree,
        seed=7,
        width_scale=0.75,
        node_weights=[4, 3, 2, 1, 1],
    )
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    h = model.forward_embedding(x)
    assert tuple(logits.shape) == (2, 3)
    assert tuple(h.shape) == (2, 256)
    assert torch.allclose(torch.linalg.vector_norm(h, dim=-1), torch.ones(2), atol=1e-5)

    checkpoint = workspace_tmp_path("maol_models") / "model.pt"
    model.save_weights(checkpoint)
    state, metadata = load_checkpoint_state(checkpoint)
    assert metadata["head_type"] == "prototype"
    assert metadata["theory"] == "MAOL"
    assert metadata["class_names"] == ["cat", "dog", "plant"]
    assert "head.centers" in state

    runtime = resolve_checkpoint_runtime_config(checkpoint)
    assert runtime.tree is not None
    assert runtime.tree.to_dict() == tree.to_dict()
    assert runtime.input_size == (64, 64)
    assert runtime.num_classes == 3
    assert runtime.embedding_norm == "l2"

    restored = TorchCNN(
        input_size=runtime.input_size,
        num_classes=runtime.num_classes,
        tree=runtime.tree,
        width_scale=runtime.width_scale,
        node_weights=[4, 3, 2, 1, 1],
    )
    restored.load_weights(checkpoint)
    restored.eval()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x), atol=1e-6)
