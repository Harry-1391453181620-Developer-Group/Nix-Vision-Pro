# 2026-05-14 Phase1.2 Layer-IDSI Design

## Goal

Implement the MAOIDL Phase1.2 upgrade in the active torch backend without moving into Phase 2 architecture work.

## Approved Policy

- Keep the existing CNN proxy architecture and Phase 1 Omega projector.
- Add `--idsi-lambda` with default `0.005`; the Layer-IDSI loss is active only when `--omega-loss` is enabled.
- Keep `L_IDSI` small relative to CE during early training by using the raw squared relative ratio for loss and applying only the small lambda weight.
- Preserve the current `L_fp`/Omega attractor loss and add Layer-IDSI as an additional term:
  `L_total = L_CE + omega_lambda * L_fp + idsi_lambda * L_IDSI`.

## Matched Feature Spaces

Whole CNN stages change channels and spatial resolution, so Phase1.2 must not directly subtract incompatible stage input/output tensors.

The monitored matched-space transitions are:

- `stage1`: output of `conv1+bn1+relu` to output of `conv2+bn2+relu+SE`, before `pool1`.
- `stage2`: output of `conv3+bn3+relu` to output of `conv4+bn4+relu+SE`, before `pool2`.
- `stage3`: output of `conv5+bn5+relu` to output of `conv6+bn6+relu+SE`, before `pool3`.
- `classifier_pre_head`: `h` to `T(h)`.

This keeps the feature spaces shape-matched without adding projection parameters or changing checkpoint compatibility.

## Metrics

- Compute Layer-IDSI statistics from per-sample relative fluctuations before final aggregation.
- Log `IDSI`, `IDSI mean`, `IDSI max`, `IDSI std`, `layer_IDSI`, `layer_IDSI_mean`, `layer_IDSI_max`, `layer_IDSI_std`, and `layer_IDSI_names`.
- Log `gradient_norm` from the existing backward pass after AMP unscale and before clipping.
- Log `hidden_norm` as the mean L2 norm over monitored feature tensors.

## Plotting

- Keep `plot.py` helper-only.
- Let `train.py` own `--plot-once`, `--plot-real-time`, `--json-dir`, `--plot-output-format`, and `--plot-output-dir`.
- Use one matplotlib window with scalar panels, one global IDSI panel, and dynamic layer-wise IDSI panels.
- Preserve stable layer colors and line artists across real-time refreshes.

## Validation

- Run syntax checks on touched Python modules.
- Run focused tests for torch model Layer-IDSI outputs, training policy loss/metrics, and plot metric handling.
- Run a small CPU training smoke with Omega and Layer-IDSI enabled when feasible.
