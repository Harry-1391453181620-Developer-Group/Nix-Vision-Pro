"""Helper functions for plotting training epoch metrics.

This module is intentionally not a CLI. Use `train.py --plot-once` or
`train.py --plot-real-time` so training owns the plotting lifecycle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCALAR_PLOT_METRICS = [
    "generalization_gap",
    "train_h_var_max",
    "train_h_var_mean",
    "train_h_var_min",
    "lr",
    "train_acc",
    "train_loss_attr",
    "train_loss_ce",
    "train_loss_total",
    "val_acc",
    "val_h_var_max",
    "val_h_var_mean",
    "val_h_var_min",
    "val_loss_attr",
    "val_loss_ce",
    "val_loss_idsi",
    "val_loss_total",
    "train_loss_idsi",
    "gradient_norm",
    "hidden_norm",
]

GLOBAL_IDSI_METRICS = [
    "IDSI",
    "IDSI mean",
    "IDSI max",
    "IDSI std",
]

LAYER_IDSI_METRICS = [
    "layer_IDSI",
    "layer_IDSI_mean",
    "layer_IDSI_max",
    "layer_IDSI_std",
]

PLOT_METRICS = SCALAR_PLOT_METRICS + GLOBAL_IDSI_METRICS + LAYER_IDSI_METRICS
PLOT_OUTPUT_FORMATS = {"png", "jpg", "jpeg"}


def read_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def resolve_plot_output_path(
    metrics_path: Path,
    *,
    output_dir: Path | None,
    output_format: str,
) -> Path:
    suffix = output_format.lower().lstrip(".")
    if suffix not in PLOT_OUTPUT_FORMATS:
        allowed = ", ".join(sorted(PLOT_OUTPUT_FORMATS))
        raise ValueError(f"Unsupported plot output format: {output_format}. Expected one of: {allowed}")
    target_dir = Path(metrics_path).parent if output_dir is None else Path(output_dir)
    return target_dir / f"epoch_metrics_plot.{suffix}"


class EpochMetricsPlotter:
    def __init__(self, *, metrics_path: Path, output_path: Path, title: str = "Experiment Metrics") -> None:
        import matplotlib.pyplot as plt

        self.metrics_path = Path(metrics_path)
        self.output_path = Path(output_path)
        self.title = title
        self._plt = plt
        self._interactive = "agg" not in plt.get_backend().lower()
        self._figure = plt.figure(num="Experiment Metrics", figsize=(22, 18))
        self._axes: dict[str, Any] = {}
        self._lines: dict[tuple[str, str], Any] = {}
        self._layer_colors: dict[str, Any] = {}
        self._legend_line_counts: dict[str, int] = {}
        self._panel_specs = self._build_panel_specs()
        self._create_axes()
        if self._interactive:
            plt.ion()
            self._figure.show()
            plt.show(block=False)
            plt.pause(0.1)

    def update(self, rows: list[dict[str, Any]]) -> None:
        self._draw(rows)
        if self._interactive:
            self._figure.canvas.draw_idle()
            self._plt.pause(0.05)

    def save(self, rows: list[dict[str, Any]]) -> Path:
        self._draw(rows)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(self.output_path, dpi=150)
        return self.output_path

    def show(self, *, block: bool) -> None:
        if not self._interactive:
            return
        self._plt.ioff()
        self._plt.show(block=block)

    def close(self) -> None:
        self._plt.close(self._figure)

    def _draw(self, rows: list[dict[str, Any]]) -> None:
        from matplotlib.ticker import MaxNLocator

        epochs = [_as_float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]

        for panel_id, title, metric_names in self._panel_specs:
            axis = self._axes[panel_id]
            if panel_id.startswith("layer:"):
                self._draw_layer_panel(axis, panel_id, title, metric_names[0], rows, epochs)
            else:
                self._draw_scalar_panel(axis, panel_id, title, metric_names, rows, epochs)
            axis.grid(True, alpha=0.3)
            axis.set_xlabel("epoch")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, min_n_ticks=3))
            axis.tick_params(axis="x", labelsize=8, rotation=30)
            if epochs:
                left = min(epochs)
                right = max(epochs) if len(epochs) > 1 else left + 1
                axis.set_xlim(left, right)
            else:
                axis.set_xlim(0, 1)
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        self._figure.suptitle(f"{self.title}: {self.metrics_path}", fontsize=10)
        self._figure.tight_layout(rect=(0, 0, 1, 0.97))

    def _build_panel_specs(self) -> list[tuple[str, str, list[str]]]:
        specs = [(f"scalar:{metric_name}", metric_name, [metric_name]) for metric_name in SCALAR_PLOT_METRICS]
        specs.append(("global_idsi", "Global IDSI metrics", list(GLOBAL_IDSI_METRICS)))
        specs.extend((f"layer:{metric_name}", metric_name, [metric_name]) for metric_name in LAYER_IDSI_METRICS)
        return specs

    def _create_axes(self) -> None:
        cols = 4
        row_count = math.ceil(len(self._panel_specs) / cols)
        axes = self._figure.subplots(row_count, cols, squeeze=False).ravel()
        for index, (panel_id, title, _) in enumerate(self._panel_specs):
            axis = axes[index]
            axis.set_title(title)
            self._axes[panel_id] = axis
        for axis in axes[len(self._panel_specs) :]:
            axis.set_visible(False)

    def _draw_scalar_panel(
        self,
        axis: Any,
        panel_id: str,
        title: str,
        metric_names: list[str],
        rows: list[dict[str, Any]],
        epochs: list[float],
    ) -> None:
        axis.set_title(title)
        for metric_name in metric_names:
            values = [_as_float(row.get(metric_name)) for row in rows]
            line = self._get_line(axis, panel_id, metric_name, label=metric_name)
            line.set_data(epochs, values)
        if len(metric_names) > 1:
            self._refresh_legend(axis, panel_id)

    def _draw_layer_panel(
        self,
        axis: Any,
        panel_id: str,
        title: str,
        metric_name: str,
        rows: list[dict[str, Any]],
        epochs: list[float],
    ) -> None:
        axis.set_title(title)
        layer_names = self._resolve_layer_names(rows, metric_name)
        active_keys: set[tuple[str, str]] = set()
        for layer_index, layer_name in enumerate(layer_names):
            values = [_as_layer_float(row.get(metric_name), layer_index) for row in rows]
            line_key = (panel_id, layer_name)
            active_keys.add(line_key)
            line = self._get_line(
                axis,
                panel_id,
                layer_name,
                label=layer_name,
                color=self._resolve_layer_color(layer_name, layer_index),
            )
            line.set_data(epochs, values)
            line.set_visible(True)
        for line_key, line in self._lines.items():
            if line_key[0] == panel_id and line_key not in active_keys:
                line.set_data([], [])
                line.set_visible(False)
        if layer_names:
            self._refresh_legend(axis, panel_id)

    def _get_line(
        self,
        axis: Any,
        panel_id: str,
        metric_name: str,
        *,
        label: str,
        color: Any | None = None,
    ) -> Any:
        key = (panel_id, metric_name)
        if key not in self._lines:
            (line,) = axis.plot(
                [],
                [],
                marker="o",
                linewidth=1.4,
                markersize=4,
                label=label,
                color=color,
            )
            self._lines[key] = line
        return self._lines[key]

    def _refresh_legend(self, axis: Any, panel_id: str) -> None:
        visible_count = sum(
            1
            for (line_panel_id, _), line in self._lines.items()
            if line_panel_id == panel_id and line.get_visible()
        )
        if self._legend_line_counts.get(panel_id) == visible_count:
            return
        self._legend_line_counts[panel_id] = visible_count
        axis.legend(loc="best", fontsize=7)

    def _resolve_layer_names(self, rows: list[dict[str, Any]], metric_name: str) -> list[str]:
        for row in reversed(rows):
            names = row.get("layer_IDSI_names")
            if isinstance(names, list) and names:
                return [str(name) for name in names]
        max_count = 0
        for row in rows:
            value = row.get(metric_name)
            if isinstance(value, list):
                max_count = max(max_count, len(value))
        return [f"layer_{index + 1}" for index in range(max_count)]

    def _resolve_layer_color(self, layer_name: str, layer_index: int) -> Any:
        if layer_name not in self._layer_colors:
            cycle = self._plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            self._layer_colors[layer_name] = cycle[layer_index % len(cycle)] if cycle else None
        return self._layer_colors[layer_name]


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _as_layer_float(value: Any, index: int) -> float:
    if not isinstance(value, list) or index >= len(value):
        return math.nan
    return _as_float(value[index])


if __name__ == "__main__":
    raise SystemExit("plot.py is a helper module. Use train.py --plot-once or train.py --plot-real-time.")
