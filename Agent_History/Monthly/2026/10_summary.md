# 2026-10 Monthly Summary

## Scope

October 2026 covers the final MAOIDL hybrid experiments, the decision to retire
MAOIDL, and the first full MAOL reconstruction on `main`.

## Main Outcomes

- Confirmed the 93.6% pure-CE CNN + token-transformer hybrid (12 classes) as the
  best MAOIDL result; the token path was still Transformer-heavy and the MAOIDL
  losses did not improve the final accuracy.
- Retired the MAOIDL theory and old repository.
- Implemented MAOL:
  - leaf wells with centers/radii;
  - deterministic parent geometry from subtree counts;
  - prototype-head classification;
  - six losses with effective-count normalization;
  - nested-directory dataset trees;
  - epoch-wise warm-up means, frozen centers, CE-only warm-up;
  - `tau=1` for 100 iterations, then `tau_warm`;
  - separate 100-step scale probe from a warm-up checkpoint.
- Removed the Transformer branch, Omega/IDSI, NumPy backend, MixUp/CutMix,
  focal loss, and label smoothing from `main`.
- Redesigned the GUI as a MAOL-only field-station interface.
- Kept `plot.py` and rewrote its panels for MAOL.

## Validation

- 37 focused tests passed.
- CPU smoke runs covered warm-up, probe, full training, and prediction.
- Warm-up/probe hardening: tau fallback, hard sanity abort, clean warm-up,
  no pretrained init, parameter-free L2 embedding normalization, measurement-only
  probe, per-group gradients.

## Next Work

- Run the long MAOL experiments on the DGX Spark with the nested ImageNet-expanded
  dataset.
- Use the probe JSON to tune the five lambdas.
- Compare MAOL (18-class) against the 18-class CE linear control with the same
  EMA; the old 93.6% model is 12-class and is not a direct comparison.
