# 2026-10-31 MAOL Reconstruction

## Source of truth

- `MAOL.docx` (updated 2026-10-30 23:13, SHA256 `1FB1E226C1CFB87D2E0A4D0E625D9D8959DF1CE457832878D5F0D49273DDC47D`)
- Nested dataset layout; no JSON tree.
- The old MAOIDL repository and theory are retired.

## Decisions

- Keep only the five-stage CNN + SE backbone.  Remove the token/Transformer path,
  Omega loss/projector, IDSI loss/projections, MixUp, CutMix, focal loss, label
  smoothing, and the NumPy backend.
- Use leaf wells + deterministic parent geometry.  `h` is the 256-d
  `fc1 + ReLU` representation; the prototype head replaces the linear head.
- Warm-up follows the B1 variant:
  - at each epoch start, recompute leaf centers from the current backbone;
  - centers are frozen (`requires_grad=False`) during warm-up;
  - only `L_CE` is applied;
  - `tau=1` for the first 100 warm-up iterations, then
    `tau_warm = mean ||h_i - c_yi||^2`;
  - after warm-up, initialize `c_EMA`, `q`, `d_intra`, `d_inter`, confusion
    weights, margins, and `tau0 = d_intra^2`, then unfreeze.
- The scale probe is a separate 100-step run from the saved warm-up checkpoint.
  It records the six loss means, the six `grad_h` norms, and total parameter
  gradient norm, then exits without saving model weights.
- Lambda policy: start at `1.0`; use the probe to bring terms within one order of
  magnitude of CE.  `L_sep` may start at `0.1`; `L_cov`/`L_center` stay in
  `0.1-1.0`.
- GUI supports MAOL checkpoints only; old MAOIDL models are intentionally
  rejected.
- `plot.py` is retained and rewritten for MAOL metrics.
- `README.md` is the only top-level project guide; `Nix_Vision_Pro.md` removed.

## Implementation

- `maol/tree.py`: nested-directory class tree, ancestors, sibling pairs,
  subtree masks and `n_u`.
- `maol/head.py`: `PrototypeHead`, parent geometry, six terms.
- `maol/losses.py`: weighted total with the document's denominators.
- `maol/stats.py`: warm-up class means, `q`, `d_intra`, `d_inter`, confusion
  weights, Lambert-W margins.
- `data/nested.py`: recursive leaf discovery and stratified split.
- `backends/torch/model.py`: CNN backbone + prototype head, structured
  checkpoints.
- `backends/torch/train_backend.py`: warm-up, probe, full MAOL training.
- `backends/torch/gui_backend.py`: redesigned field-station GUI.

## Validation

- New test suite covers tree construction, nested discovery, head geometry,
  loss normalization, checkpoint round-trip, plotting, and training utilities.
- Smoke runs cover warm-up, 100-step-style probe (shortened for CI), full training
  from a warm-up checkpoint, and prediction.
- Follow-up warm-up/probe hardening (tau fallback, sanity abort, no-update probe,
  group gradients) is documented in
  `Agent_History/docs/plans/2026-10-31-warmup-probe-hardening.md`.
