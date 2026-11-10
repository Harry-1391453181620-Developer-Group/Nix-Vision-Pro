"""MAOL training metric plotting.

The trainer owns the plotting lifecycle (``--plot-once`` / ``--plot-real-time``).
This module stays helper-only, but the old standalone picker is retained for
quick inspection of an existing JSONL file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# Matplotlib backends that cannot open a GUI window. GUI backends commonly end
# in "agg" (qtagg, tkagg, gtk3agg, ...), so the backend must be checked against
# this explicit set instead of a substring test.
HEADLESS_BACKENDS = frozenset({"agg", "pdf", "ps", "svg", "cairo", "pgf", "template"})


def backend_is_interactive(backend_name: str | None = None) -> bool:
    """Return whether ``backend_name`` can display a GUI window.

    ``None`` resolves the currently active matplotlib backend. Inline/notebook
    backends (``module://...``) are treated as non-interactive because the
    training script cannot own their widget lifecycle.
    """
    if backend_name is None:
        import matplotlib

        backend_name = matplotlib.get_backend()
    name = str(backend_name).strip().lower()
    if not name:
        return False
    if name.startswith("module://"):
        return False
    return name not in HEADLESS_BACKENDS


SCALAR_PLOT_METRICS = [
    "train_loss_total",
    "train_loss_ce",
    "train_loss_intra",
    "train_loss_sep",
    "train_loss_nest",
    "train_loss_cov",
    "train_loss_center",
    "val_loss_total",
    "val_loss_ce",
    "val_loss_intra",
    "val_loss_sep",
    "val_loss_nest",
    "val_loss_cov",
    "val_loss_center",
    "train_acc",
    "val_acc",
    "generalization_gap",
    "lr",
    "gradient_norm",
    "hidden_norm",
    "tau",
    "d_intra",
    "d_inter",
    "margin_cap",
    "mean_leaf_radius",
    "mean_leaf_center_norm",
]

PLOT_METRICS = list(SCALAR_PLOT_METRICS)
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
    def __init__(self, *, metrics_path: Path, output_path: Path, title: str = "MAOL Experiment Metrics") -> None:
        import matplotlib.pyplot as plt

        self.metrics_path = Path(metrics_path)
        self.output_path = Path(output_path)
        self.title = title
        self._plt = plt
        self.backend = plt.get_backend()
        self.interactive = backend_is_interactive(self.backend)
        self._figure = plt.figure(num="MAOL Experiment Metrics")
        self._axes: dict[str, Any] = {}
        self._lines: dict[tuple[str, str], Any] = {}
        self._panel_specs = [(f"scalar:{name}", name, [name]) for name in SCALAR_PLOT_METRICS]
        self._create_axes()
        if self.interactive:
            plt.ion()
            self._figure.show()
            plt.show(block=False)
            plt.pause(0.1)

    def update(self, rows: list[dict[str, Any]]) -> None:
        self._draw(rows)
        if self.interactive:
            self._figure.canvas.draw_idle()
            self._plt.pause(0.05)

    def save(self, rows: list[dict[str, Any]]) -> Path:
        self._draw(rows)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(self.output_path, dpi=150)
        return self.output_path

    def show(self, *, block: bool) -> None:
        if not self.interactive:
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
            axis.set_title(title)
            for metric_name in metric_names:
                values = [_as_float(row.get(metric_name)) for row in rows]
                line = self._get_line(axis, panel_id, metric_name, label=metric_name)
                line.set_data(epochs, values)
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
        self._figure.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.05,
            top=0.95,
            wspace=0.35,
            hspace=0.75,
        )

    def _create_axes(self) -> None:
        cols = 5
        row_count = math.ceil(len(self._panel_specs) / cols)
        self._figure.set_size_inches(cols * 4.0, row_count * 3.0, forward=True)
        axes = self._figure.subplots(row_count, cols, squeeze=False).ravel()
        for index, (panel_id, title, _) in enumerate(self._panel_specs):
            axis = axes[index]
            axis.set_title(title)
            self._axes[panel_id] = axis
        for axis in axes[len(self._panel_specs) :]:
            axis.set_visible(False)

    def _get_line(
        self,
        axis: Any,
        panel_id: str,
        metric_name: str,
        *,
        label: str,
    ) -> Any:
        key = (panel_id, metric_name)
        if key not in self._lines:
            (line,) = axis.plot([], [], marker="o", linewidth=1.4, markersize=4, label=label)
            self._lines[key] = line
        return self._lines[key]


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _choose_metrics_file() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        selected = filedialog.askopenfilename(
            title="Select MAOL epoch metrics JSONL",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _plot_selected_file() -> int:
    metrics_path = _choose_metrics_file()
    if metrics_path is None:
        return 0
    rows = read_metric_rows(metrics_path)
    if not rows:
        raise SystemExit(f"No valid JSON object rows found in: {metrics_path}")
    output_path = metrics_path.parent / "epoch_metrics_plot.png"
    plotter = EpochMetricsPlotter(metrics_path=metrics_path, output_path=output_path)
    try:
        plotter.save(rows)
    finally:
        plotter.close()
    print(f"Plot saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_plot_selected_file())
