from __future__ import annotations

import json
from pathlib import Path

import pytest

from plot import PLOT_METRICS, read_metric_rows, resolve_plot_output_path


TEST_OUTPUT_ROOT = Path("runs") / "_plot_metric_tests"


def test_plot_metrics_use_train_h_var_names() -> None:
    assert "train_h_var_max" in PLOT_METRICS
    assert "train_h_var_mean" in PLOT_METRICS
    assert "train_h_var_min" in PLOT_METRICS
    assert "h_var_max" not in PLOT_METRICS
    assert "h_var_mean" not in PLOT_METRICS
    assert "h_var_min" not in PLOT_METRICS


def test_read_metric_rows_preserves_train_h_var_names() -> None:
    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = TEST_OUTPUT_ROOT / "epoch_metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "train_h_var_max": 3.0,
                "train_h_var_mean": 2.0,
                "train_h_var_min": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_metric_rows(metrics_path)

    assert rows[0]["train_h_var_max"] == pytest.approx(3.0)
    assert rows[0]["train_h_var_mean"] == pytest.approx(2.0)
    assert rows[0]["train_h_var_min"] == pytest.approx(1.0)


def test_resolve_plot_output_path_defaults_to_metrics_directory() -> None:
    metrics_path = TEST_OUTPUT_ROOT / "run" / "epoch_metrics.jsonl"

    output_path = resolve_plot_output_path(metrics_path, output_dir=None, output_format="jpg")

    assert output_path == TEST_OUTPUT_ROOT / "run" / "epoch_metrics_plot.jpg"


def test_resolve_plot_output_path_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported plot output format"):
        resolve_plot_output_path(TEST_OUTPUT_ROOT / "epoch_metrics.jsonl", output_dir=None, output_format="svg")
