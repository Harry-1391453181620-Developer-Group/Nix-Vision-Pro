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

- `python -m pytest tests -q -p no:cacheprovider`: 37 passed.
- Warm-up smoke (5 epochs, tiny nested dataset): warm-up checkpoint written.
- Probe smoke (3 steps, shortened for CI): JSONL and summary written; no model
  file written.
- Full training smoke (1 epoch from warm-up checkpoint): epoch metrics and
  summary written.
- Prediction smoke: checkpoint loaded and top-k probabilities printed.
- `train.py --help` and module imports checked.

## Notes

- MAOL warm-up and probe do not use a pretrained `--init-from`; the backbone
  forms the unit-sphere geometry itself.
- The GUI deliberately rejects old MAOIDL checkpoints.
- `MAOL.docx` is tracked in git and not added to `.gitignore` or `.stignore`.

## Follow-up: warm-up / probe hardening

The first real warm-up exposed a collapsed tau (`1e-6`), an undertrained
checkpoint (`8.97%` val_acc), `d_inter < d_intra`, and `q_max/q_median=107.6`.
The probe also stepped with raw 253k gradients and reported a mean of 3444.

Changes:

- tau fallback to `1.0` when `tau_raw < 1e-2` or non-finite; log
  `tau_raw/tau_used/tau_fallback`.
- warm-up uses clean images by default; `--maol-warmup-augment` opts in.
- `--init-from` was removed from warm-up/probe; MAOL warm-up starts from the
  current backbone, default 15 epochs, LR `2*lr[0]`.
- hard sanity gate (`val_acc`, `d_inter/d_intra >= 1.3`, `q_max/q_median`,
  `tau_floor <= tau0 <= 10`) with exit code 2 and `warmup_rejected.json`; the
  same gate runs when loading an existing warm-up checkpoint.
- probe defaults to `--maol-probe-no-update`; records pre/post-clip norms,
  per-term gradients for backbone/fc1/centers/radii, and median/p90/max.
- smoke-verified abort path, no-update probe, and clipped update probe. 37 tests
  pass.

## Follow-up: L2-normalized embedding

The second real warm-up (18 classes) produced `tau0=3.37e6`,
`tau_warm=5.88e6`, `d_intra=1837`, `d_inter=2165`, and a sparse `c_EMA` with
row norms ~6000–6660.  This was feature-scale explosion, not a tau arithmetic
bug.  The model now applies parameter-free L2 normalization after `fc1+ReLU`;
the prototype head, centers, radii, q, margins and all five geometry losses use
the same unit-sphere `h`.  Centers are free parameters normalized on read.
Checkpoints now record `embedding_norm=l2`; older raw-scale checkpoints are
rejected and must be retrained.
