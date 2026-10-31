# 2026-10-31 Warm-up / Probe Hardening

## Why this follow-up exists

The first MAOL warm-up/probe run exposed three separate problems:

1. **Warm-up was undertrained.**  It ran from scratch (`init_from=None`), with
   RandAugment on, for only 5 epochs.  Final val_acc was `8.97%` on 18 classes.
2. **Warm-up tau collapsed.**  Epoch 1 had `tau_raw=1e-6`; the old code used
   `max(tau_epoch, 1e-6)`, so logits exploded and the backbone was damaged.
3. **Probe was not measurement-only.**  It recorded the raw gradient norm and
   then called `optimizer.step()` without clipping.  A single step had
   `total_grad_norm=253279`; the reported mean `3444` was dominated by outliers.
   The probe also used `tau0=0.8804`, not `1.0`.

The warm-up checkpoint also had `d_intra=0.938 > d_inter=0.755` and
`q_max/q_median=107.6`, so the geometry was already degenerate before lambda
tuning.

## Decisions

### Warm-up

- If `tau_raw < tau_floor` (default `1e-2`) or non-finite, **fall back to
  `tau=1.0`**, not to a small floor.
- Log `tau_raw`, `tau_used`, `tau_fallback`, and the cumulative fallback count
  in `warmup_metrics.jsonl`.
- Use clean images by default (`--maol-warmup-augment` is off).  Class means are
  computed on the clean loader, so warm-up training should use the same
  distribution.
- `--init-from` loads **only** `conv*/bn*/se*/fc1*`; it never loads `fc2`,
  `token_*`, `cls_token`, or a prototype head.
- Warm-up LR: `lr[0]` with a pretrained init, `2 * lr[0]` from scratch; override
  with `--maol-warmup-lr`.
- Hard sanity gate at the end of warm-up:
  - `val_acc >= 2 / num_classes`
  - `d_inter >= d_intra`
  - `q_max / q_median <= 10`
  - `tau0 >= tau_floor`
- On failure: write `warmup_diagnostics.json` and `warmup_rejected.json`, print
  failures, exit code `2`, and do not save a valid warm-up checkpoint.
- The same gate runs when an existing warm-up checkpoint is loaded, before
  probe or full training.

### Probe

- Default is measurement-only (`--maol-probe-no-update`): no
  `optimizer.step()`, no BN/stat updates, so the warm-up state is unchanged.
- `--maol-probe-update` opts into a 100-step trajectory; gradients are clipped
  before the step.
- Record pre-clip and post-clip total gradient norms.
- Record mean/median/p90/min/max for losses and gradient norms.
- Record per-term gradients for `backbone`, `fc1`, `centers`, `radii`.
- Record `tau` and `lr` per step.
- Probe remains JSON-only; no model weights are written.

## Validation

- 37 focused tests pass.
- CPU smoke covers:
  - tau fallback (`tau_raw=0` -> `tau_used=1`)
  - warm-up abort with exit code `2` and `warmup_rejected.json`
  - no-update probe with group gradients
  - update probe with clipping (`postclip ~= 1.0`)
  - full training from a warm-up checkpoint
  - backbone-only init (92 tensors loaded, head ignored)
