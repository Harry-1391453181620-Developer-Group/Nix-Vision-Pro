"""Live plotter for Omega experiment epoch metrics."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


METRICS = [
    "generalization_gap",
    "h_var_max",
    "h_var_mean",
    "h_var_min",
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


def _find_latest_metrics_file(runs_dir: Path) -> Path:
    candidates = [path for path in runs_dir.rglob("epoch_metrics.jsonl") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No epoch_metrics.jsonl files found under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
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


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _plot_rows(metrics_path: Path, rows: list[dict[str, Any]], *, output: Path | None) -> None:
    import matplotlib.pyplot as plt

    cols = 4
    rows_count = math.ceil(len(METRICS) / cols)
    fig = plt.figure(num="Experiment Metrics", figsize=(18, 12), clear=True)
    axes = fig.subplots(rows_count, cols, squeeze=False).ravel()
    epochs = [_as_float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]

    for index, metric_name in enumerate(METRICS):
        axis = axes[index]
        values = [_as_float(row.get(metric_name)) for row in rows]
        axis.plot(epochs, values, marker="o", linewidth=1.4, markersize=4)
        axis.set_title(metric_name)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
        if epochs:
            axis.set_xticks(epochs)

    for axis in axes[len(METRICS) :]:
        axis.set_visible(False)

    fig.suptitle(str(metrics_path), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plot epoch metrics from runs/*/epoch_metrics.jsonl")
    parser.add_argument("metrics_file", nargs="?", type=Path, help="Path to an epoch_metrics.jsonl file")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Run artifact root used to find the latest metrics file")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Render one update and exit")
    parser.add_argument("--output", type=Path, help="Optional PNG path to write on each refresh")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    interactive_backend = "agg" not in plt.get_backend().lower()
    if interactive_backend:
        plt.ion()
    metrics_path = args.metrics_file or _find_latest_metrics_file(args.runs_dir)
    last_seen: tuple[int, int] | None = None

    while True:
        if args.metrics_file is None:
            metrics_path = _find_latest_metrics_file(args.runs_dir)
        rows = _read_metrics(metrics_path)
        stat = metrics_path.stat()
        seen = (int(stat.st_mtime_ns), len(rows))
        if seen != last_seen:
            _plot_rows(metrics_path, rows, output=args.output)
            if interactive_backend:
                plt.pause(0.001)
            last_seen = seen
        if args.once:
            break
        if interactive_backend:
            plt.pause(max(0.1, float(args.interval)))
        else:
            time.sleep(max(0.1, float(args.interval)))

    if not args.once and interactive_backend:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
