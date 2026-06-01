# 2026-06-01 Phase 2 Tokenized Dynamics Implementation

## Source Instructions

This implementation follows `Agent_History/docs/plans/2026-5-22-FROM-USER-Phase-2-Implementation-Instructions.md`.

The core interpretation is:

- keep the CNN as the primary representation extractor
- add a lightweight tokenized dynamics path without converting the model into a ViT
- preserve Phase 1/1.2 training, logging, plotting, EMA, AMP, compile, checkpoint, and augmentation behavior
- keep token counts lightweight and monitor token diversity to avoid Omega/IDSI-driven collapse

The CFT material added after the IDSI explanation in `smart.docx` was intentionally excluded from this implementation.

## Implemented Scope

Phase 2 is implemented in the PyTorch backend only. The NumPy backend remains the legacy CNN comparison path.

The optional token path is enabled by `--tokenize`. With tokenization disabled, existing Phase 1 and Phase 1.2 commands remain valid.

The token path uses:

- final CNN feature map before global pooling
- adaptive average pooling only when needed to keep token count at or below 64
- spatial flattening from `[B, C, H, W]` to `[B, N, C]`
- lightweight `Linear(128 -> token_dim)` projection
- weak learned or sinusoidal positional encoding
- shallow residual transformer dynamics block
- mean token pooling by default for classification
- CLS pooling support only for ablation

Default Phase 2 settings:

- `token_dim=128`
- `transformer_depth=1`
- `attention_heads=4`
- `transformer_mlp_ratio=2.0`
- `token_pool=mean`
- `token_positional_encoding=learned`
- `token_dropout=0.1`
- `transformer_layernorm=pre`

## Omega And IDSI Behavior

Token Omega loss uses already-computed token states:

```text
L_attr = mean((H_tokens - sg(T(H_tokens)))^2)
```

The transformer branch target is detached for token Omega so the loss does not train the transformer to collapse tokens toward a single target.

Layer-IDSI in token mode monitors:

- `stage1`
- `stage2`
- `stage3`
- `token_projection`
- `transformer_token_block`

It does not monitor attention heads, attention projections, FFN sublayers, or LayerNorm modules separately.

## Metrics And Plotting

Existing Phase 1.2 metrics are preserved.

Token mode additionally logs:

- token variance
- inter-token variance
- token norm mean/min/max
- mean pairwise cosine similarity
- mean pairwise cosine distance
- token collapse score and warning

`plot.py` remains helper-only and backward compatible. Token metrics are added as optional scalar panels, while layer-wise IDSI color mapping remains stable through existing layer-name mapping.

## Checkpoint And Inference Compatibility

Structured torch checkpoints now include token metadata when tokenization is enabled:

- `tokenize`
- `token_dim`
- `transformer_depth`
- `attention_heads`
- `transformer_mlp_ratio`
- `token_pool`
- `token_positional_encoding`
- `token_dropout`
- `transformer_layernorm`
- `token_grid_size`

Torch inference reconstructs tokenized checkpoints before loading weights. Legacy raw checkpoints and non-token Phase 1/1.2 checkpoints remain supported.

## Validation

Validation performed:

- AST syntax parse for modified Python modules and tests
- focused tests for torch model, training policy/loss logic, and plot metrics
- full test suite
- torch training CLI help parse

Observed results:

- `32 passed` for the focused Phase 2-related test set
- `79 passed` for the full test suite
- CLI help includes all required Phase 2 arguments

## Notes

This implementation provides the required architecture and observability surface. It does not run the required scientific experiment matrix; the experiment configurations remain the next training step:

- tokenized architecture without Omega loss
- tokenized architecture with Omega loss
- best Phase 1 CNN baseline
- best Phase 1.2 CNN + Layer-IDSI baseline
