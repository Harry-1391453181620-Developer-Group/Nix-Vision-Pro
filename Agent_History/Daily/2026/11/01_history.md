# 2026-11-01

## Task

Add an 18-class no-MAOL CE control with a linear head, and use EMA on both sides
of the comparison.

## Changes

- Added `--maol-mode {full,none}` and the `--no-maol` alias.
- Added `TorchCNN(..., maol=False)`:
  - raw `ReLU(fc1(...))` embedding, no L2;
  - `fc2 = Linear(256, num_classes)`;
  - no prototype head, wells, radii, `c_EMA`, or `tau`.
- Added `_run_linear_training`:
  - CE only, same phase/cosine LR, augmentation, grad clip, early stop and EMA;
  - run tag `ce` and zero geometry fields for plotting compatibility;
  - checkpoint metadata `head_type=linear`, `embedding_norm=raw`, `maol=false`.
- `predict.py` now loads both MAOL and CE linear checkpoints; GUI stays MAOL-only.
- Rejected MAOL-only flags in `none` mode.
- Updated `README.md`, `best_train_commands.txt` (both `--maol-mode` and `--ema`),
  and added the CE control command.
- Added the design note
  `Agent_History/docs/plans/2026-11-01-ce-linear-control.md`.

## Validation

- 38 tests pass.
- CPU smoke 2 epochs, 4 classes, `--maol-mode none --ema`:
  - summary `maol_mode=none`, `head_type=linear`, `ema=true`;
  - checkpoint `fc2.weight` present, no `head.centers`;
  - `predict.py` loads it.
- CPU smoke 1 epoch MAOL with `--ema` from an L2 warm-up checkpoint.
- Existing MAOL checkpoint still predicts.

## Notes

- The CE control reads the same 18-class tree from `Dataset/`; no warm-up
  checkpoint is used in CE mode, so there is no MAOL tree mismatch.
- The old 93.6% model was **12-class** and used a token-transformer head, so it
  is not directly comparable with the 18-class MAOL runs.  The fair comparison
  is 18-class MAOL vs 18-class CE linear, both with EMA.
