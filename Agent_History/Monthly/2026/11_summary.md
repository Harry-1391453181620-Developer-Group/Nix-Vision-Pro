# 2026-11 Summary

## Focus

- Added the 17-class no-MAOL CE linear control.
- Added `--maol-mode {full,none}` and `--no-maol`.
- Enabled EMA consistently in the documented MAOL and CE training commands.

## Result

- `TorchCNN(..., maol=False)` uses raw `fc1+ReLU` and `Linear(256, N)`; no wells,
  radii, `c_EMA`, `tau`, warm-up, or geometry losses.
- `predict.py` supports both MAOL prototype and CE linear checkpoints.
- GUI remains MAOL-only.
- 38 tests pass, including linear-checkpoint metadata and raw-embedding checks.

## Next Work

- Run the 17-class CE linear control with EMA.
- Re-run MAOL with EMA on the same 17-class tree and compare best val accuracy,
  macro accuracy, and per-class errors.
- Keep the old 18-class 93.6% number out of the direct comparison; it used a
  different class count and a token-transformer head.
