# Daily History - 2026-10-31

## Summary

Replaced the retired MAOIDL implementation with MAOL (Multi-Attractor Optimized
Learning) on the `main` branch.  The work is CNN-only, uses the nested dataset
tree directly, and keeps the existing plotting and checkpoint infrastructure
where it still fits the theory.

## Request

- Re-read the updated `MAOL.docx` immediately before implementation.
- Remove the Transformer branch and all MAOIDL code from `main`.
- Delete the NumPy backend entirely.
- Remove MixUp, CutMix, focal loss, and label smoothing.
- Implement the document's warm-up exactly, but with at least 5 epochs.
- Implement the B1 variant: epoch-wise class means, frozen centers, CE only.
- Implement `tau=1` for the first 100 warm-up iterations, then `tau_warm`.
- Add a separate scale probe from a warm-up checkpoint (100 steps, JSON only).
- Keep plotting; redesign the GUI; GUI supports MAOL checkpoints only.
- Keep only `README.md` among the project guides; remove `Nix_Vision_Pro.md`.
- Do not move `Dataset`, `runs`, `.stfolder`, `.codex`, or `test_image`.

## Implementation

- Added `maol/tree.py`, `maol/head.py`, `maol/losses.py`, `maol/stats.py`.
- Added `data/nested.py` for recursive leaf discovery and stratified splits.
- Rewrote `backends/torch/model.py` as five-stage CNN + `PrototypeHead`.
- Rewrote `backends/torch/train_backend.py` with warm-up, probe, and full MAOL
  training.
- Rewrote `predict_backend.py` and the Tkinter `gui_backend.py` for MAOL-only
  checkpoints.
- Rewrote `plot.py` for MAOL loss/geometry panels.
- Removed `backends/numpy/`, top-level `nn/`, `Nix_Vision_Pro.md`,
  `dist_optimizer.py`, and the obsolete MAOIDL tests.
- Updated `README.md`, `.gitignore`, `.stignore`, and the local
  `best_train_commands.txt`; removed `CONTRIBUTING.md` so `README.md` is the only
  top-level project guide.
- Added the MAOL reconstruction design note and October summary.

## Validation

- `python -m pytest tests -q -p no:cacheprovider`: 32 passed.
- Warm-up smoke (5 epochs, tiny nested dataset): warm-up checkpoint written.
- Probe smoke (3 steps, shortened for CI): JSONL and summary written; no model
  file written.
- Full training smoke (1 epoch from warm-up checkpoint): epoch metrics and
  summary written.
- Prediction smoke: checkpoint loaded and top-k probabilities printed.
- `train.py --help` and module imports checked.

## Notes

- The old 93.6% MAOIDL checkpoint can initialize only the CNN/fc1 backbone via
  `--init-from`; its token/linear head is skipped.
- The GUI deliberately rejects old MAOIDL checkpoints.
- `MAOL.docx` remains untracked locally pending the user's upload; it is not
  added to `.gitignore` or `.stignore`.
