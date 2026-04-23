# 2026-03-27 Dual-Backend MixUp, Focal Loss, and Width-Scaling Design

## Scope

Apply the same training-policy upgrade to both maintained backends:

1. configurable augmentation strength
2. random erasing
3. optional MixUp
4. focal-loss replacement for balance sampling
5. configurable model width scaling

The project must keep both `torch` and `numpy` training paths behaviorally
aligned where practical, while allowing backend-specific implementations where
the math or gradient mechanics differ.

## Constraints and Decisions

### MixUp and Focal Loss

MixUp produces soft labels:

`y = lam * y1 + (1 - lam) * y2`

Classic focal loss is defined against a single target probability `p_t`, so
enabling focal loss together with MixUp would either require a soft-label focal
formulation or would create ambiguous and unstable behavior.

Decision:

- keep both features available
- if MixUp is active for a run, focal loss is disabled for that run
- when focal loss is disabled through MixUp, the loss falls back to cross
  entropy for both training and validation

This is intentionally conservative and keeps `val_loss` interpretable.

### Random-Erasing Fill Value

Random erasing remains the last transform in the training augmentation chain.

Decision:

- erased pixels are filled with the per-image channel mean instead of zeros

This avoids baking in a special black patch assumption and stays compatible with
future normalization schemes where zero may not represent a natural image value.

### Width Change

The requested stage-2 width reduction should not be hard-coded.

Decision:

- add `--model-width-scale`
- default value: `0.75`
- stage 2 uses `round(64 * scale)` with a minimum safe width floor
- stage 3 input width follows stage 2 output width
- stage 3 output width remains `128`

Default behavior therefore changes stage 2 from `64` channels to `48`.

Checkpoint compatibility is expected to change when the width scale differs, so
all model-construction entry points must accept the same width-scale argument.

## CLI Design

Both train backends gain:

- `--rotation`
- `--brightness`
- `--contrast`
- `--saturation`
- `--mixup / --no-mixup`
- `--mixup-prob`
- `--focal-loss / --no-focal-loss`
- `--focal-gamma`
- `--focal-alpha {auto,none}`
- `--model-width-scale`

Validation rules:

- `0 <= rotation < 180`
- `0 <= brightness <= 1`
- `0 <= contrast <= 1`
- `0 <= saturation <= 1`
- `0 <= mixup_prob <= 1`
- `focal_gamma >= 0`
- `model_width_scale > 0`

Defaults:

- `rotation = 12`
- `brightness = 0.2`
- `contrast = 0.2`
- `saturation = 0.2`
- `mixup = false`
- `mixup_prob = 0.5`
- `focal_loss = true`
- `focal_gamma = 1.5`
- `focal_alpha = auto`
- `model_width_scale = 0.75`

The documentation will explicitly recommend keeping rotation at or below
roughly `20` degrees for CIFAR-like datasets even though the CLI permits a
larger numeric range.

## Augmentation Policy

The shared transform order stays:

1. random resized crop
2. horizontal flip
3. rotation
4. brightness / contrast / saturation jitter
5. random erasing

Semantics:

- `rotation=x` means sample angle in `[-x, +x]`
- `brightness=x` means sample factor in `[1-x, 1+x]`
- `contrast=x` means sample factor in `[1-x, 1+x]`
- `saturation=x` means sample factor in `[1-x, 1+x]`

Random erasing:

- probability `0.3`
- erased area sampled uniformly from `10%` to `20%` of total image area
- use a bounded aspect-ratio range so masks are not always square
- erase by filling with the per-image channel mean

The shared augmentation helpers live in `utils/training.py` so both training
backends use the same geometry and photometric policy.

## MixUp Policy

MixUp is applied after batch loading and after standard augmentation, because it
operates on already prepared tensors / arrays and should not be followed by
independent destructive transforms per mixed sample.

Per batch:

- sample a Bernoulli decision using `mixup_prob`
- if active, sample `lam ~ Beta(0.2, 0.2)`
- pair samples using a random permutation of the batch
- mix both inputs and one-hot labels using the same `lam`

This policy is shared conceptually between backends, but implemented in the
native math path of each trainer.

## Loss Policy

### Torch

PyTorch uses backend-native focal loss with support for:

- hard labels when focal loss is active
- optional class-wise alpha weights from inverse-frequency class weights
- standard cross entropy when focal loss is disabled or MixUp is active

### NumPy

NumPy receives matching forward and backward implementations:

- focal-loss forward for reporting and validation
- focal-loss backward for training gradients
- cross-entropy fallback for MixUp or explicit `--no-focal-loss`

### Alpha Handling

`--focal-alpha auto` reuses the existing inverse-frequency class-weight
computation as focal alpha.

`--focal-alpha none` disables alpha weighting even if class-weighting remains
enabled elsewhere.

## Model Construction

Both `TorchCNN` and `CNN` gain a `width_scale` constructor argument.

Affected entry points:

- both train backends
- both predict backends
- both GUI backends

This keeps inference compatible with the training-time width decision and
prevents silently constructing a mismatched model shape at load time.

## Tests

Add or update focused tests for:

- augmentation determinism under the same RNG seed
- new augmentation argument validation
- random-erasing mean fill behavior
- MixUp label and batch-shape behavior
- focal-loss forward and backward behavior in NumPy
- focal-loss training compatibility in PyTorch
- width-scaled model tensor shapes for both backends
- checkpoint-loading tolerance when width-scaled tensors differ

## Documentation and History

After implementation:

- update `Image_Identify_CNN.md`
- update `README.md`
- append `Agent_History/Daily/2026/03/27_history.md`
- commit the design doc first
- commit the implementation after tests and document updates

## Notes

The `agent-memory-mcp` skill was requested, but no memory MCP resources are
configured in the current session. The approved design is therefore persisted in
this repository design document instead of an external memory store.
