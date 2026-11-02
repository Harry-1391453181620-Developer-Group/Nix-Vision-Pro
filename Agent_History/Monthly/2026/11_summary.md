# 2026-11 Summary

## Focus

- Added the 18-class no-MAOL CE linear control.
- Added `--maol-mode {full,none}` and `--no-maol`.
- Enabled EMA consistently in the documented MAOL and CE training commands.

## Result

- `TorchCNN(..., maol=False)` uses raw `fc1+ReLU` and `Linear(256, N)`; no wells,
  radii, `c_EMA`, `tau`, warm-up, or geometry losses.
- `predict.py` supports both MAOL prototype and CE linear checkpoints.
- GUI remains MAOL-only.
- 38 tests pass, including linear-checkpoint metadata and raw-embedding checks.

## Corrections

- All MAOL runs are 18-class; the old 93.6% MAOIDL model was 12-class with a
  token-transformer head, so those two numbers are not comparable.
- See `Agent_History/docs/plans/2026-11-02-class-count-correction.md`.

## Next Work

- Run the 18-class CE linear control with EMA.
- Re-run MAOL with EMA on the same 18-class tree and compare best val accuracy,
  macro accuracy, and per-class errors.
- Keep the old 93.6% number out of the direct comparison: that model was
  **12-class** with a token-transformer head, while all MAOL runs are 18-class.
