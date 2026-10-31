"""Tests for MAOL plotting helpers."""

from __future__ import annotations

import json

import pytest

from plot import PLOT_METRICS, read_metric_rows, resolve_plot_output_path
from tests.conftest import workspace_tmp_path


def test_plot_metrics_cover_maol_terms() -> None:
    for name in (
        "train_loss_ce",
        "train_loss_intra",
        "train_loss_sep",
        "train_loss_nest",
        "train_loss_cov",
        "train_loss_center",
        "val_acc",
        "tau",
        "d_intra",
        "d_inter",
        "margin_cap",
        "mean_leaf_radius",
    ):
        assert name in PLOT_METRICS
    assert "IDSI" not in PLOT_METRICS


def test_read_metric_rows_round_trip() -> None:
    root = workspace_tmp_path("plot_tests")
    path = root / "epoch_metrics.jsonl"
    path.write_text(
        json.dumps({"epoch": 1, "train_loss_ce": 1.25, "val_acc": 0.5}) + "\n",
        encoding="utf-8",
    )
    rows = read_metric_rows(path)
    assert rows[0]["train_loss_ce"] == pytest.approx(1.25)
    assert rows[0]["val_acc"] == pytest.approx(0.5)


def test_plot_output_path_and_format_validation() -> None:
    metrics_path = workspace_tmp_path("plot_tests") / "run" / "epoch_metrics.jsonl"
    assert resolve_plot_output_path(metrics_path, output_dir=None, output_format="jpg") == (
        metrics_path.parent / "epoch_metrics_plot.jpg"
    )
    with pytest.raises(ValueError, match="Unsupported plot output format"):
        resolve_plot_output_path(metrics_path, output_dir=None, output_format="svg")
