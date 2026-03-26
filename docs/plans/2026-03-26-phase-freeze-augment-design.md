# 2026-03-26 Training Policy Upgrade Design

## Scope

Add three training-policy upgrades to both active backends:

1. Stronger but controlled data augmentation
2. Temporary backbone freezing on `val_acc` plateau
3. Multiphase learning-rate training with per-phase warmup

The architecture stays unchanged in this task.

## Augmentation

Training-time augmentation remains inside the batch augmentation path so
validation and inference preprocessing stay deterministic.

The transform order is fixed to:

1. random resized crop
2. horizontal flip
3. rotation
4. color jitter
5. cutout

Rotation uses reflection padding before rotation, centered rotation, and
bilinear interpolation. Rotation probability is `0.5` and the angle range is
`[-12°, +12°]`.

Color jitter probability is `0.5` with safe ranges:

- brightness: `+-0.2`
- contrast: `+-0.2`
- saturation: `+-0.2`

The augmentation helpers are centralized so both NumPy and PyTorch backends use
the same geometry and photometric policy.

## Multiphase Learning Rate

Add:

- `--phase-count`
- `--lr` as a list with one value per phase
- `--warmup-epochs`

Epoch membership for each phase is computed with:

`np.array_split(range(epochs), phase_count)`

This keeps remainder epochs distributed predictably.

Validation rules:

- number of LR values must equal `phase_count`
- LR list must be monotonically non-increasing

Existing schedulers remain available, but they operate inside each phase:

- `constant`
- `step`
- `cosine`

Cosine scheduling intentionally restarts each phase.

Warmup is optional and ramps linearly from `0.1 * base_lr` to `base_lr` at the
start of each phase.

## Temporary Backbone Freeze

Add:

- `--freeze-patience` with default `8`
- `--freeze-epoch-num` with default `10`
- `--after-unfreeze-lr-change` as an additive LR decrement

If `val_acc` fails to improve for `freeze-patience` consecutive epochs inside
the current phase:

- freeze the backbone temporarily
- keep training the head for exactly `freeze-epoch-num` epochs
- unfreeze automatically after that timed window

After unfreeze:

- keep the phase base LR fixed as the scheduler anchor
- track a cumulative `phase_lr_offset` inside the phase
- compute `effective_lr = max(scheduled_lr - phase_lr_offset, min_lr)` where
  `min_lr = phase_base_lr * min_lr_ratio`
- compute the unfreeze candidate from the current effective LR, not the phase
  base LR
- apply a new deduction only when:
  - `candidate_lr > 0`
  - if a next phase exists: `candidate_lr >= next_phase_lr - 1e-12`
  - `current_effective_lr > 2 * min_lr`
- cap the cumulative offset so the effective LR can never undercut the next
  phase start LR, or the scheduler floor in the final phase

At the next phase boundary:

- reset `phase_lr_offset = 0`
- keep using the explicit value from `--lr` as the new phase scheduler anchor

Backbone means convolution, SE, and BatchNorm feature layers.
Head means `fc1`, dropout, and `fc2`.

PyTorch behavior:

- freeze backbone parameters with `requires_grad=False`
- set backbone BN layers to `eval()` while frozen
- restore BN layers to `train()` on unfreeze
- rebuild the optimizer on each freeze transition

Advanced option:

- `--freeze-bn-affine false` keeps BN affine parameters trainable while BN
  running statistics stay frozen

NumPy behavior:

- backbone BN running statistics must not update while frozen
- optimizer parameter list switches to head-only, or head plus BN affine when
  that option is enabled

## Tests

Add focused tests for:

- deterministic augmentation under the same RNG seed
- BN running-stat stability during freeze
- freeze -> unfreeze -> freeze transition correctness
- phase assignment, LR validation, and per-phase warmup/cosine behavior
- effective LR clamp to the scheduler floor
- post-unfreeze cumulative offset with next-phase and final-phase caps
- epsilon-safe next-phase LR comparison

## Documentation

Update the Markdown guides and the daily history log after implementation.
