# 2026-11-09 EMA BatchNorm Copy Regression

## Symptom

The 2026-11-04 MAOL training run was considered the worst of the recent
training attempts:

```text
runs/20261104-070057-maol-warmup-seed_42   warm-up:  val_acc 73.21%, diagnostics passed
runs/20261104-082820-maol-probe-seed_42    probe:   100 no-update steps, healthy scales
runs/20261104-122357-maol-seed_42          main:    best val_acc 73.28% @ epoch 1,
                                                   val_acc 44.24% @ epoch 51 (early stop)
```

The full run's epoch-1 validation accuracy was exactly the warm-up accuracy,
then validation accuracy fell every epoch while training accuracy rose from
58.3% to 95.4%. The run was configured with `ema=true`, `ema_decay=0.9999`.

## Evidence gathered

1. The 11-04 warm-up and probe are not the problem. Their geometry numbers are
   identical to the healthy 11-01 warm-up (`d_intra=0.4905`,
   `d_inter=1.1828`, `tau0=0.2406`), and the probe losses/gradients match the
   earlier probes. The 11-04 probe and main run actually inherited the old
   11-01 warm-up checkpoint (`warmup_diagnostics.json` shows
   `val_acc=0.741417`, `tau_fallback_epochs=0`); the new warm-up checkpoint was
   written later, after those runs had started. The two warm-ups are
   geometrically equivalent, so this does not affect the conclusion.
2. Training metrics (train accuracy, train CE, gradient norms) are bit
   identical across the 11-01 non-EMA run, the 11-02 legacy-EMA run, and the
   11-04 collapsed run. The live model training trajectory is therefore
   unchanged; only the evaluated/saved weights differ.
3. The old 11-02 EMA run used decay `0.9993` with the pre-14ecf02 `ModelEMA`
   that EMA-blended floating buffers together with parameters. It reached
   `best val_acc 89.93%` and `final val_acc 89.70%`.
4. Commit `14ecf02` ("Copy live buffers in EMA and accept EMA decay") changed
   `ModelEMA.update` to copy every registered buffer exactly from the live
   model — including BatchNorm `running_mean`/`running_var` — while continuing
   to EMA-blend parameters. The 11-04 run combined that change with decay
   `0.9999`.

## Decisive experiment

`checkpoints/maol_warmup.pt`, `checkpoints/maol_model.pt` (EMA epoch 1), and
`checkpoints/maol_best_model.pt` (non-EMA epoch 112) were evaluated on the
exact 6,845-image validation split used by training, then re-evaluated after
swapping BatchNorm buffers between checkpoints:

```text
warm-up checkpoint as-is                               74.083%
EMA epoch-1 checkpoint as-is                           73.455%
non-EMA epoch-112 checkpoint as-is                     89.934%
EMA epoch-1 weights + warm-up BatchNorm buffers        63.901%
non-EMA epoch-112 weights + warm-up BatchNorm buffers   4.222%   (near random 5.56%)
```

The one-epoch mismatch (`73.455% -> 63.901%`) matches the collapsed run's
epoch-2 validation accuracy (`66.02%`). BatchNorm running statistics are
extremely sensitive: they define the normalization used during evaluation, so
they must belong to the same training state as the weights.

## Root cause

`ModelEMA` copied the live model's BatchNorm statistics into an EMA model whose
parameters lag behind (`0.9999^214 ≈ 97.9%` of the previous epoch's EMA per
epoch; effective window about 10,000 steps ≈ 47 epochs). The evaluated model
therefore used weights from one training epoch and normalization statistics
from another. Validation accuracy degrades as the gap grows, which is exactly
the observed monotone collapse. The live model itself was healthy.

The commit `14ecf02` intent (avoid stale averaged statistics) was directionally
reasonable, but copying live statistics into a lagged model is the wrong
implementation: the correct invariant is that the evaluated EMA model is
internally self-consistent (EMA weights with EMA statistics).

## Fix

`utils/training.py`, class `ModelEMA`:

- removed the `buffer_keys` exact-copy branch;
- floating state (parameters and floating buffers, including BatchNorm
  `running_mean`/`running_var`) is EMA-blended with the same decay;
- non-floating counters (`num_batches_tracked`) are still copied exactly;
- `skip_prefixes` statistics (`head.c_ema`, `head.q`, `head.tau`,
  `head.node_weights`, `head.margins`, `head.sep_weights`) are still copied
  exactly, because they describe the current data distribution rather than the
  weights (and `tau` is fixed after warm-up).

The `--ema <decay>` CLI from `14ecf02` is kept.

Tests (`tests/test_training_utils.py`):

- `test_ema_blends_floating_buffers` (was `test_ema_copies_buffers_exactly`);
- `test_ema_blends_batchnorm_running_stats` (was
  `test_ema_copies_batchnorm_running_stats`);
- new `test_ema_copies_integer_counters_exactly`;
- `test_ema_copies_skipped_statistics_exactly` unchanged.

Validation: `42 passed`.

## Follow-up for the next remote run

- The fix restores the semantics of the healthy 11-02 EMA run, so rerun MAOL
  with `--ema 0.9993` (or `--ema 0.999`). A decay of `0.9999` is not required
  and implies a very long lag; keep it only as an explicit experiment.
- `checkpoints/maol_model.pt` currently contains the collapsed run's EMA epoch-1
  weights (`best_epoch=1`, 73.46% val accuracy). Rerun warm-up/full training or
  use `checkpoints/maol_best_model.pt` (non-EMA, 89.93%) until then.
- A paired evaluation (log both live and EMA validation metrics each epoch)
  would make this class of regression immediately visible; it was not added in
  this fix.

## Verification limits

No end-to-end training rerun was performed locally: training runs on the remote
DGX Spark, and the attempted local 2-epoch reproduction was interrupted by a
machine restart. The root cause is established by the checkpoint-swap
measurement and by the bit-identical training trajectories; the fix restores
the code path already validated by the 11-02 run.

## Security notes

- Checkpoint loading still uses `weights_only=True`.
- No new shell execution, network access, or arbitrary deserialization.
- `--ema` decay validation still enforces `0 < decay < 1`.
