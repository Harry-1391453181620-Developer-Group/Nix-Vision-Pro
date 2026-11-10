# 2026-11-10 — Fix `--plot-real-time` / `--plot-once` plotting lifecycle

## Task

User report: after training, no plot is produced automatically; during training,
no real-time window opens. `--plot-real-time` and `--plot-once` "are not working
properly". Investigate with systematic debugging (root cause first), then fix.

## Phase 1 — Evidence

Code inspection (current HEAD `f25487f`):

- `plot.py`: `EpochMetricsPlotter.__init__` computed
  `self._interactive = "agg" not in plt.get_backend().lower()`. GUI backends
  also end in `agg` (`qtagg`, `tkagg`, `gtk3agg`, `gtk4agg`, `qt5agg`, `wxagg`),
  so this predicate is False for them. `update()` then never redraws on screen
  and `show()` returns immediately.
- `plot.py`: real-time updates only draw in memory; nothing is saved after
  training. Only the `--plot-once` branch calls `save()`.
- `train_backend.py`: `--plot-once` is handled at the end of `_run_training` /
  `_run_linear_training`; it builds a **second** plotter (second figure) even
  when a real-time plotter already exists, saves, and calls `show(block=True)`.
- `train_backend.py`: `--plot-real-time` creates the plotter at training start;
  no warning and no artifact if the backend is non-interactive.

Environment / reproduction:

- Project venv matplotlib default backend on this Windows machine is `qtagg`.
  Running `MPLBACKEND=qtagg python -c "... 'agg' not in backend"` prints
  `False`, i.e. the old code classified an interactive Qt backend as headless.
  With `MPLBACKEND=Agg`, `EpochMetricsPlotter(...).save()` writes a 368 KB PNG
  correctly, so the save path itself was sound.
- Training runs on the DGX Spark are headless and the runbook sets
  `MPLBACKEND=Agg`, so a window can never open there; the trainer neither warned
  nor saved the figure.

Root causes:

1. Interactive-backend detection via substring `"agg" in backend` is wrong for
   every GUI backend whose name ends in `agg`; both the live and the
   `--plot-once` windows were silently disabled.
2. Real-time runs never wrote a plot artifact; non-interactive runs were
   completely silent, so a headless run looked exactly like "plotting broken".
3. `--plot-once` after `--plot-real-time` created a duplicate figure instead of
   reusing the existing one.

## Phase 4 — Fix

- `plot.py`: added `HEADLESS_BACKENDS` (`agg`, `pdf`, `ps`, `svg`, `cairo`,
  `pgf`, `template`) and `backend_is_interactive(backend_name=None)`; notebook
  (`module://...`) backends are treated as non-interactive. The plotter now
  exposes `backend` and `interactive`; the buggy `_interactive` substring test
  is gone.
- `train_backend.py`: `_build_plotter` logs whether the live window is enabled
  or why it cannot be; new `_finalize_plotting` used by both training paths:
  - always saves `epoch_metrics_plot.<ext>` when either plot flag is set,
  - reuses the real-time plotter instead of creating a second figure,
  - `--plot-once` shows a blocking window only when the backend supports it,
  - headless `--plot-once` prints the saved path instead of failing silently.
- Inline `--plot-once`/`--plot-real-time` argparse help updated.
- `README.md` plotting section and local `best_train_commands.txt` note updated:
  DGX/batch runs are headless and now always get a PNG; local desktop runs get a
  live window.

## Validation

- `python -m pytest tests -q -p no:cacheprovider` → `48 passed`.
- New tests: GUI-backend classification regression (`qtagg`/`tkagg`/... True,
  `agg`/`pdf`/... False), headless plotter save, `_finalize_plotting` behavior
  (save always, window only when interactive, no-op when flags off/empty).
- End-to-end smoke (CPU, 5-class synthetic dataset, `MPLBACKEND=Agg`):
  - `--plot-real-time`: warning printed, then
    `runs/<run>/epoch_metrics_plot.png` written (229,412 bytes).
  - `--plot-once`: PNG written (229,905 bytes) and the headless warning printed.
- `MPLBACKEND=TkAgg` instantiation: `interactive=True`, window created, figure
  saved and closed without error.
- Not verified: a full-length training window on the user's desktop (would
  require a long training run); the detection path it depends on is covered by
  the classification test and the TkAgg smoke.
