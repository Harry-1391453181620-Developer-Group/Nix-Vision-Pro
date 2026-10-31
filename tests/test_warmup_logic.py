"""Tests for warm-up tau fallback and the sanity gate."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from backends.torch.train_backend import (
    _build_warmup_diagnostics,
    _resolve_warmup_tau,
    _summarize_values,
)
from maol.stats import WarmupStats


def _stats(*, d_intra: float = 0.5, d_inter: float = 1.0, q: list[float] | None = None) -> WarmupStats:
    q_values = q or [0.4, 0.5, 0.6]
    return WarmupStats(
        c_ema=torch.zeros((len(q_values), 2)),
        q=torch.tensor(q_values, dtype=torch.float32),
        node_weights=torch.ones(4),
        d_intra=d_intra,
        d_inter=d_inter,
        margin_cap=1.0,
        m_base=0.8,
        m_max=0.9,
        k0=1.0,
        alpha=0.1,
        tau0=0.25,
        tau_warm=0.25,
        confusion=torch.zeros((len(q_values), len(q_values)), dtype=torch.long),
        w_vu=torch.ones(1),
        margins=torch.ones(1),
        sep_weights=torch.ones(1),
    )


def test_tau_falls_back_to_one_for_collapsed_estimate() -> None:
    used, raw, fallback = _resolve_warmup_tau(1e-6, 1e-2, global_step=0)
    assert used == 1.0
    assert raw == 1e-6
    assert fallback is True

    used, raw, fallback = _resolve_warmup_tau(1e-6, 1e-2, global_step=100)
    assert used == 1.0
    assert fallback is True

    used, raw, fallback = _resolve_warmup_tau(2.5, 1e-2, global_step=100)
    assert used == pytest.approx(2.5)
    assert fallback is False


def test_sanity_gate_accepts_healthy_geometry() -> None:
    diagnostics = _build_warmup_diagnostics(
        _stats(q=[0.4, 0.5, 0.6]),
        val_acc=0.5,
        num_classes=4,
        tau_floor=1e-2,
        acc_factor=2.0,
        max_q_ratio=10.0,
    )
    assert diagnostics["passed"] is True
    assert diagnostics["failures"] == []


def test_sanity_gate_rejects_bad_geometry() -> None:
    diagnostics = _build_warmup_diagnostics(
        _stats(d_intra=0.938, d_inter=0.755, q=[0.5, 0.5, 50.0]),
        val_acc=0.08,
        num_classes=18,
        tau_floor=1e-2,
        acc_factor=2.0,
        max_q_ratio=10.0,
    )
    assert diagnostics["passed"] is False
    assert any("val_acc" in failure for failure in diagnostics["failures"])
    assert any("d_inter" in failure for failure in diagnostics["failures"])
    assert any("q_max/q_median" in failure for failure in diagnostics["failures"])


def test_value_summary_reports_median_and_tail() -> None:
    summary = _summarize_values([1.0, 2.0, 3.0, 100.0])
    assert summary["median"] == pytest.approx(2.5)
    assert summary["max"] == pytest.approx(100.0)
    assert summary["p90"] > 3.0
