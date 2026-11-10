# 2026-11-10

## Task

Two parts requested by the user:

1. Check that the current code and training process fully align with the MAOL
   theory in `MAOL.docx`.
2. Debug plotting: training no longer produces a plot after finishing and no
   real-time window opens during training (`--plot-real-time`, `--plot-once`).

## Findings

### Plotting

- `EpochMetricsPlotter` decided interactivity with
  `"agg" not in plt.get_backend().lower()`. GUI backends also end in `agg`
  (`qtagg` is the default backend of the project venv), so on this machine the
  live window and the final `--plot-once` window were both silently disabled.
- `--plot-real-time` only drew in memory and never saved anything; a headless
  backend (DGX with `MPLBACKEND=Agg`) produced no warning and no artifact, which
  looked exactly like broken plotting.
- `--plot-once` after `--plot-real-time` built a second plotter/figure instead of
  reusing the existing one.

### MAOL theory audit

- Full theory text extracted from `MAOL.docx` (paragraphs + inline OMML math) to
  `.maol_tmp/MAOL_theory_wrapped.txt`; a read-only audit against
  `maol/head.py`, `maol/losses.py`, `maol/stats.py`, `maol/tree.py`,
  `backends/torch/model.py`, `backends/torch/train_backend.py` and
  `utils/training.py` is in progress (report to be written).

## Changes

- `plot.py`: added `HEADLESS_BACKENDS` and `backend_is_interactive()`; the
  plotter exposes `backend`/`interactive` and no longer uses the substring test.
- `backends/torch/train_backend.py`: `_build_plotter` now reports whether a live
  window is available; new `_finalize_plotting()` always saves the plot when
  either flag is set, reuses the real-time plotter, and only shows a blocking
  window for `--plot-once` when the backend supports one.
- `tests/test_plot_metrics.py`: regression test for GUI-backend classification,
  headless save test, and `_finalize_plotting` behavior tests.
- `README.md` plotting section and `best_train_commands.txt` notes updated.
- Added `Agent_History/docs/plans/2026-11-10-plotting-lifecycle.md`.

## Validation

- Full test suite: `48 passed`.
- Headless end-to-end smoke (`MPLBACKEND=Agg`, CPU, synthetic 5-class dataset):
  `--plot-real-time` and `--plot-once` both write
  `runs/<run>/epoch_metrics_plot.png` and print the expected diagnostics.
- `MPLBACKEND=TkAgg`: `interactive=True`, window created, figure saved.

## Pending

- Finish the MAOL theory-vs-code audit report and fix any confirmed mismatches
  in a separate commit.
