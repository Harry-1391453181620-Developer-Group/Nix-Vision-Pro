"""Tests for the MAOL prototype head geometry and losses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from maol.head import PrototypeHead
from maol.losses import compute_maol_losses
from maol.tree import ClassTree


def _head() -> PrototypeHead:
    tree = ClassTree.from_leaf_paths(["animal/cat", "animal/dog", "plant"])
    head = PrototypeHead(tree, 2, node_weights=[4, 3, 2, 1, 1])
    with torch.no_grad():
        head.centers.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        head.raw_radii.zero_()
    return head


def test_parent_geometry_is_weighted_bottom_up() -> None:
    head = _head()
    centers, radii = head.node_geometry()
    # animal = (2*cat + 1*dog)/3 = (2/3, 1/3); root = (3*animal + plant)/4
    assert torch.allclose(centers[1], torch.tensor([2.0 / 3.0, 1.0 / 3.0]), atol=1e-6)
    assert torch.allclose(centers[0], torch.tensor([0.25, 0.25]), atol=1e-6)
    assert radii.shape == (5,)
    assert torch.all(radii > 0)


def test_logits_and_terms_are_finite() -> None:
    head = _head()
    h = torch.tensor([[0.1, 0.2], [1.0, -0.5], [-0.2, 1.5]])
    labels = torch.tensor([0, 1, 2])
    terms = compute_maol_losses(
        head,
        h,
        labels,
        lambdas={"intra": 1.0, "sep": 1.0, "nest": 1.0, "cov": 1.0, "center": 1.0},
    )
    assert tuple(terms.logits.shape) == (3, 3)
    for value in terms.named().values():
        assert torch.isfinite(value)


def test_zero_lambdas_make_total_equal_ce() -> None:
    head = _head()
    h = torch.tensor([[0.1, 0.2], [1.0, -0.5]])
    labels = torch.tensor([0, 1])
    terms = compute_maol_losses(
        head,
        h,
        labels,
        lambdas={"intra": 0.0, "sep": 0.0, "nest": 0.0, "cov": 0.0, "center": 0.0},
    )
    assert torch.allclose(terms.total, terms.ce)


def test_cov_lambda_is_added_after_ce() -> None:
    head = _head()
    h = torch.tensor([[0.1, 0.2], [1.0, -0.5]])
    labels = torch.tensor([0, 1])
    terms = compute_maol_losses(
        head,
        h,
        labels,
        lambdas={"intra": 0.0, "sep": 0.0, "nest": 0.0, "cov": 0.5, "center": 0.0},
    )
    assert torch.allclose(terms.total, terms.ce + 0.5 * terms.cov)


def test_ema_stats_update_only_with_enough_samples() -> None:
    head = _head()
    h = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [2.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 0, 0, 1])
    before = head.c_ema.clone()
    head.update_ema_stats(h, labels, n_min=5)
    assert not torch.allclose(head.c_ema[0], before[0])
    assert torch.allclose(head.c_ema[1], before[1])  # only one dog sample
