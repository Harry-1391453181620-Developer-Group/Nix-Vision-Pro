# 2026-05-14 History

## Request

Implemented Phase1.2 for the MAOIDL upgrade after the user approved the design and clarified matched feature-space requirements for CNN Layer-IDSI.

## Work completed

- Added matched-space Layer-IDSI transitions to the torch CNN for `stage1`, `stage2`, `stage3`, and `classifier_pre_head`.
- Added `--idsi-lambda` to the torch trainer with default `0.005`, active for Omega-loss runs.
- Added `L_IDSI` to the training objective as a small layer-wise stability regularizer.
- Logged Phase1.2 metrics into `epoch_metrics.jsonl`:
  - `IDSI`
  - `IDSI mean`
  - `IDSI max`
  - `IDSI std`
  - `layer_IDSI`
  - `layer_IDSI_mean`
  - `layer_IDSI_max`
  - `layer_IDSI_std`
  - `layer_IDSI_names`
  - `gradient_norm`
  - `hidden_norm`
- Updated `plot.py` so the train-owned plot lifecycle dynamically adapts to monitored layers and keeps all scalar/global/layer IDSI panels in one matplotlib window.
- Updated `README.md` with the Phase1.2 CLI, metric schema, and plotting behavior.
- Wrote the design note:
  `Agent_History/docs/plans/2026-05-14-phase1-2-layer-idsi-design.md`.

## Validation

- Ran Python syntax checks on the touched modules.
- Ran focused pytest coverage for model, training policy, and plot metric behavior.
- Ran a one-epoch CPU smoke training command with `--omega-loss`, `--idsi-lambda 0.005`, and `--plot-once`.

## Notes

- `L_IDSI` uses raw squared relative fluctuation ratios for optimization, while logged IDSI values are scaled by `100` for readability and continuity with earlier run logs.
- Gradient norm is measured from the existing backward pass after AMP unscale and before clipping; no additional backward pass is introduced.
