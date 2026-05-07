# 2026-05-07 History

## Request

Fixed the plotting lifecycle after `plot.py --once` proved insufficient, moved plot controls into `train.py`, renamed training representation variance metrics, simplified run artifact layout expectations, and included verified local `gui.py` plus `LICENSE` changes in the commit scope.

## Work completed

- Converted `plot.py` into a helper-only module with no standalone CLI.
- Added torch training CLI options:
  - `--plot-once`
  - `--plot-real-time`
  - `--json-dir`
  - `--plot-output-format`
  - `--plot-output-dir`
- Integrated plotting into the training lifecycle:
  - `--plot-real-time` opens the figure when training starts and refreshes after each epoch.
  - `--plot-once` opens the figure after training finishes.
  - Both modes save a final image file to the selected output directory.
- Changed JSONL and plotted train variance metric names to:
  - `train_h_var_max`
  - `train_h_var_mean`
  - `train_h_var_min`
- Kept run artifacts directly under `runs/<timestamp>-.../`.
- Refactored the torch trainer plotting integration into small helpers to keep `train_backend.py` stable and easier to inspect.
- Updated README documentation and added focused plot metric tests.

## Validation

- Ran no-bytecode syntax checks for `train.py`, `backends/torch/train_backend.py`, `plot.py`, and `tests/test_plot_metrics.py`.
- Verified `train.py --backend torch --help` exposes the new plot options.
- Ran focused tests for plot helpers and the full training policy module.
- Ran a one-epoch headless `train.py --plot-once` smoke; it wrote JSONL with `train_h_var_*` keys and saved `epoch_metrics_plot.png`.
