"""Helper functions for plotting training epoch metrics.

This module is intentionally not a CLI. Use `train.py --plot-once` or
`train.py --plot-real-time` so training owns the plotting lifecycle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PLOT_METRICS = [
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
    "val_loss_total",
    "IDSI",
]

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
        self._figure = plt.figure(num="Experiment Metrics", figsize=(20, 13))
        if self._interactive:
            plt.ion()
            self._figure.show()

    def update(self, rows: list[dict[str, Any]]) -> None:
        self._draw(rows)
        if self._interactive:
            self._plt.pause(0.001)

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

        self._figure.clear()
        cols = 4
        row_count = math.ceil(len(PLOT_METRICS) / cols)
        axes = self._figure.subplots(row_count, cols, squeeze=False).ravel()
        epochs = [_as_float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]

        for index, metric_name in enumerate(PLOT_METRICS):
            axis = axes[index]
            values = [_as_float(row.get(metric_name)) for row in rows]
            axis.plot(epochs, values, marker="o", linewidth=1.4, markersize=4)
            axis.set_title(metric_name)
            axis.set_xlabel("epoch")
            axis.grid(True, alpha=0.3)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, min_n_ticks=3))
            if epochs:
                axis.set_xlim(min(epochs), max(epochs) if len(epochs) > 1 else min(epochs) + 1)

        for axis in axes[len(PLOT_METRICS) :]:
            axis.set_visible(False)

        self._figure.suptitle(f"{self.title}: {self.metrics_path}", fontsize=10)
        self._figure.tight_layout(rect=(0, 0, 1, 0.97))


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


if __name__ == "__main__":
    raise SystemExit("plot.py is a helper module. Use train.py --plot-once or train.py --plot-real-time.")
