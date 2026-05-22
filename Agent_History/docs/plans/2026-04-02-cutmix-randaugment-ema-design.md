# 2026-04-02 CutMix, RandAugment, and EMA Design

## Scope

Apply the next training-policy upgrade to both maintained backends:
- replace the fixed augmentation stack with a shared RandAugment-style policy
- add CutMix alongside MixUp under one probability gate
- add EMA with validation and best-checkpoint evaluation on EMA weights
- upgrade checkpoints to structured `model + meta` payloads while preserving legacy-load compatibility

## Shared Policy Decisions

### RandAugment

Keep augmentation shared in `utils/training.py` so both `torch` and `numpy` trainers see the same image policy.

Decision:
- `--augment` now means a shared RandAugment-style policy
- each image samples two ops with replacement from a fixed pool
- keep the existing user-facing magnitude controls for rotation, brightness, contrast, and saturation
- keep cutout / random erasing as one of the shared sampled ops
- validation remains preprocess-only

This keeps the backends behaviorally aligned without introducing backend-specific transform drift.

### Mix Routing

The user requested one probability gate and no stacking of MixUp and CutMix.

Decision:
- if `rand() >= mixup_prob`: no mixing
- else if `rand() < cutmix_ratio`: CutMix
- else: MixUp

Defaults:
- `mixup_prob = 0.4`
- `cutmix_ratio = 0.5`

Additional rules:
- validation disables MixUp and CutMix completely
- CutMix recomputes `lam` from the actual pasted patch area after clipping
- label smoothing is set to `0` on mixed batches
- focal loss runs only on non-mixed batches

### EMA

EMA must track the full model state, not only trainable parameters.

Decision:
- use a real EMA model instance per backend
- update after `optimizer.step()`
- blend the full `state_dict`, including floating buffers
- copy non-floating state entries such as BN counters exactly
- evaluate validation metrics on EMA weights when EMA is enabled
- save the best checkpoint from EMA weights when EMA is enabled

Because this project uses phase restarts and temporary freeze/unfreeze cycles, EMA also gets a short phase-start warmup so the shadow weights can catch up after abrupt optimizer-state changes.

### Checkpoints

The user chose option 1: structured checkpoints.

Decision:
- new checkpoints save `model` and `meta`
- `meta` records at least `checkpoint_version`, backend, `is_ema`, and `ema_decay`
- model loaders continue to accept older plain checkpoints
- `--init-from` loads the live model first, then synchronizes EMA from that loaded model

This preserves predictor compatibility while making future debugging easier because saved files now describe whether they contain EMA weights.

## Testing Plan

Add or update focused tests for:
- RandAugment determinism under a fixed RNG seed
- CutMix area-based lambda behavior
- batch-mix route selection
- EMA tracking of parameters and BN buffers
- structured-checkpoint metadata round trips
- legacy checkpoint compatibility

## Notes

- `agent-memory-mcp` was requested again, but no memory MCP resources are configured in this session.
- The `writing-plans` skill referenced by the brainstorming workflow is not installed in this environment, so the approved design is captured directly in this repository design doc before implementation and verification.
