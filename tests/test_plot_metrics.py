"""Tests for MAOL plotting helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from plot import (
    PLOT_METRICS,
    backend_is_interactive,
    read_metric_rows,
    resolve_plot_output_path,
)
from tests.conftest import workspace_tmp_path


def test_backend_is_interactive_classifies_gui_backends() -> None:
    # Regression: GUI backends end in "agg" (qtagg/tkagg/...), so a naive
    # "agg" substring test misclassified them as headless and disabled windows.
    for backend in ("qtagg", "tkagg", "qt5agg", "gtk3agg", "gtk4agg", "wxagg", "macosx", "webagg"):
        assert backend_is_interactive(backend) is True, backend
    for backend in ("agg", "pdf", "ps", "svg", "cairo", "pgf", "template", ""):
        assert backend_is_interactive(backend) is False, backend
    assert backend_is_interactive("module://matplotlib_inline.backend_inline") is False


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


def test_plotter_saves_png_with_headless_backend() -> None:
    import matplotlib

    from plot import EpochMetricsPlotter

    previous = matplotlib.get_backend()
    try:
        matplotlib.use("Agg", force=True)
        root = workspace_tmp_path("plot_tests")
        metrics_path = root / "epoch_metrics.jsonl"
        output_path = root / "epoch_metrics_plot.png"
        rows = [
            {"epoch": epoch, "train_loss_ce": 1.0 / epoch, "val_acc": 0.5 + 0.1 * epoch}
            for epoch in (1, 2, 3)
        ]
        plotter = EpochMetricsPlotter(metrics_path=metrics_path, output_path=output_path)
        try:
            assert plotter.interactive is False
            assert plotter.save(rows) == output_path
            assert output_path.is_file()
            assert output_path.stat().st_size > 0
            plotter.show(block=True)  # no-op for headless backends, must not raise
        finally:
            plotter.close()
    finally:
        matplotlib.use(previous, force=True)


class _FakePlotter:
    def __init__(self, *, interactive: bool, output_path: Path) -> None:
        self.interactive = interactive
        self.backend = "fakeagg"
        self.output_path = output_path
        self.save_calls = 0
        self.show_calls: list[bool] = []

    def save(self, rows: list[dict[str, object]]) -> Path:
        self.save_calls += 1
        return self.output_path


def test_finalize_plotting_saves_and_shows_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from backends.torch import train_backend as tb

    output_path = workspace_tmp_path("plot_tests") / "plot.png"
    plotter = _FakePlotter(interactive=True, output_path=output_path)
    calls: list[bool] = []

    def fake_save(rows):
        calls.append(True)
        return output_path

    def fake_show(*, block: bool) -> None:
        plotter.show_calls.append(block)

    plotter.save = fake_save  # type: ignore[method-assign]
    plotter.show = fake_show  # type: ignore[method-assign]
    args = argparse.Namespace(plot_once=True, plot_real_time=False)

    tb._finalize_plotting(plotter, [{"epoch": 1}], Path("metrics.jsonl"), args)
    assert calls == [True]
    assert plotter.show_calls == [True]


def test_finalize_plotting_headless_plot_once_saves_without_window() -> None:
    from backends.torch import train_backend as tb

    plotter = _FakePlotter(interactive=False, output_path=workspace_tmp_path("plot_tests") / "plot.png")
    args = argparse.Namespace(plot_once=True, plot_real_time=False)
    tb._finalize_plotting(plotter, [{"epoch": 1}], Path("metrics.jsonl"), args)
    assert plotter.save_calls == 1
    assert plotter.show_calls == []


def test_finalize_plotting_real_time_always_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    from backends.torch import train_backend as tb

    built: list[object] = []
    plotter = _FakePlotter(interactive=False, output_path=workspace_tmp_path("plot_tests") / "plot.png")

    def fake_build(metrics_path, args):
        built.append((metrics_path, args))
        return plotter

    monkeypatch.setattr(tb, "_build_plotter", fake_build)
    args = argparse.Namespace(plot_once=False, plot_real_time=True)
    tb._finalize_plotting(None, [{"epoch": 1}], Path("metrics.jsonl"), args)
    assert len(built) == 1
    assert plotter.save_calls == 1


def test_finalize_plotting_ignores_empty_rows_and_disabled_flags() -> None:
    from backends.torch import train_backend as tb

    plotter = _FakePlotter(interactive=True, output_path=workspace_tmp_path("plot_tests") / "plot.png")
    enabled = argparse.Namespace(plot_once=True, plot_real_time=True)
    tb._finalize_plotting(plotter, [], Path("metrics.jsonl"), enabled)
    assert plotter.save_calls == 0

    disabled = argparse.Namespace(plot_once=False, plot_real_time=False)
    tb._finalize_plotting(plotter, [{"epoch": 1}], Path("metrics.jsonl"), disabled)
    assert plotter.save_calls == 0

