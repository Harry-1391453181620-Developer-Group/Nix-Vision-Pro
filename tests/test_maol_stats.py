"""Tests for warm-up statistics and margin utilities."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maol.stats import WarmupStats, compute_class_means, lambert_w


def test_compute_class_means_tau_uses_normalized_centers() -> None:
    samples = [
        (torch.tensor([[1.0, 0.0], [0.6, 0.8]]), torch.tensor([0, 0])),
        (torch.tensor([[0.0, 1.0], [0.8, 0.6], [-0.6, 0.8]]), torch.tensor([1, 1, 1])),
    ]
    means, counts, tau_warm = compute_class_means(
        lambda x: x,
        samples,
        torch.device("cpu"),
        num_classes=2,
        embedding_dim=2,
    )
    stacked = torch.cat([x for x, _ in samples])
    labels = torch.cat([y for _, y in samples])
    expected_means = torch.stack([stacked[labels == c].mean(dim=0) for c in range(2)])
    normalized = torch.nn.functional.normalize(expected_means, p=2, dim=-1)
    expected_tau = torch.mean(torch.sum((stacked - normalized[labels]) ** 2, dim=-1))
    assert torch.allclose(means, expected_means)
    assert torch.allclose(counts, torch.tensor([2.0, 3.0]))
    assert tau_warm == pytest.approx(float(expected_tau), rel=1e-6)


def test_lambert_w_inverts_k_times_exp_k() -> None:
    values = np.array([0.0, 0.1, 1.0, 5.0, 20.0], dtype=np.float64)
    result = lambert_w(values)
    np.testing.assert_allclose(result * np.exp(result), values, atol=1e-9)


def test_warmup_stats_metadata_round_trip() -> None:
    stats = WarmupStats(
        c_ema=torch.zeros((2, 3)),
        q=torch.ones(2),
        node_weights=torch.ones(4),
        d_intra=1.5,
        d_inter=2.5,
        margin_cap=3.0,
        m_base=1.2,
        m_max=2.4,
        k0=1.0,
        alpha=2.0,
        tau0=2.25,
        tau_warm=1.8,
        confusion=torch.zeros((2, 2), dtype=torch.long),
        w_vu=torch.ones(1),
        margins=torch.ones(1),
        sep_weights=torch.ones(1),
    )
    restored = WarmupStats.from_metadata(stats.to_metadata())
    assert restored.d_intra == pytest.approx(1.5)
    assert torch.allclose(restored.c_ema, stats.c_ema)
    assert torch.allclose(restored.margins, stats.margins)
