# You MUST FOLLOW EVERY INSTRUCTIONS IN THIS FILE

1. Read this file, D:\Programing_materials\Python\python_Projects\Nix_Vision_Pro\Agent_History\docs\plans\2026-4-24-FROM-USER-roadmap-after-4-24-VERY-IMPORTANT-GUIDE.txt, and the file that D:\Programing_materials\Python\python_Projects\Nix_Vision_Pro\smart.docx - 快捷方式.lnk is pointing to.

2. Make sure that you understand everything explained is any files

3. Make sure you know every edge of the current project before implement.

4. Despite a tiny attention block is added, the dataset remained unchanged.

5. Also read D:\Programing_materials\Python\python_Projects\Nix_Vision_Pro\Agent_History\docs\plans\2026-05-14-phase1-2-layer-idsi-design.md to make sure that you understand what we did recently. But you don't need to follow the instructions in that file.

6. Preserve the current architecture while adding new ones.

7. Layer-IDSI monitoring must include:
    - CNN backbone stages
    - token projection stage (if enabled)
    - transformer token block

8. The CNN backbone remains the primary representation extractor.

9. The transformer token block is a lightweight dynamics refinement module operating on already-extracted spatial features.

10. Do NOT redesign the architecture around transformer dominance.

11. Explainthe intention of 8, 9, 10: DON'T TURN IT INTO A tiny ViT!!!

12. Token count must remain lightweight.
    If spatial resolution becomes too large, apply lightweight spatial reduction before tokenization.

    Preferred methods:
    - adaptive average pooling
    - strided convolution
    - lightweight projection pooling

    Avoid excessive token counts that significantly increase attention complexity or VRAM usage.

13. YOU SHOULD KNOW THAT Omega loss + IDSI loss can VERY EASILY cause low token diversity, where tokens become identical. So be sure that:
    - The objective is NOT to force all tokens toward identical stable representations.
    - Healthy token diversity must be preserved.
    - Token stabilization should emerge while maintaining meaningful inter-token feature differentiation.

14. Token diversity should primarily reflect meaningful feature differentiation between spatial tokens.

    Preferred diversity indicators include:
    - inter-token variance
    - mean pairwise cosine distance
    - token covariance statistics

Avoid defining diversity using only raw activation magnitude.

## PHASE 2 — TOKENIZED INTERNAL DYNAMICS ARCHITECTURE

Phase 2 upgrades the current Phase1 CNN backbone into a lightweight tokenized dynamics architecture

The objective of Phase2 is NOT to build a full Vision Transformer.
The objective is to lift the representation space from a single pooled hidden vector into a structured token manifold, allowing Ω-loss and future dynamics objectives to operate on higher-dimensional internal trajectories.

The current Phase1/1.2 training pipeline, logging system, plotting system, EMA flow, AMP compatibility, compile compatibility, checkpoint compatibility, and augmentation pipeline must remain functional.

Backward compatibility with existing training commands is important.

### Core Architectural Change

The current CNN backbone should no longer collapse immediately into a single pooled vector.

Instead:

CNN feature map:
F ∈ ℝ^(B×C×H×W)

must be reshaped into tokens:

H_tokens ∈ ℝ^(B×N×D)

where:

N = spatial token count
D = token dimension

Recommended implementation:

- Use the final CNN feature map before global pooling
- Flatten spatial dimensions into tokens
- Use lightweight linear projection if needed

Example:

[B, C, H, W]
→ reshape
[B, H*W, C]

Optional lightweight projection:
Linear(C → D)

Tokenization should preserve local semantic continuity.

Neighboring spatial tokens should remain semantically correlated after tokenization.

Avoid aggressive token mixing before transformer dynamics refinement.

The token projection stage should remain lightweight.
Recommended initial token count range:
    - 16 ~ 64 tokens

Avoid large token grids during early Phase2 experiments.

Avoid:

- deep projection stacks
- large nonlinear projection towers
- excessive normalization layers

Projection is intended only for dimensional alignment and lightweight token adaptation.

DO NOT introduce expensive patch embedding systems.
DO NOT redesign the entire CNN backbone.

This is a lightweight tokenization upgrade.

### Lightweight Transformer Dynamics Block

Introduce a minimal transformer dynamics block:

T(H_tokens)

Requirements:

- single transformer layer only
- minimal attention heads
- low token dimension
- shallow MLP
- preserve training stability
- avoid large GPU overhead

Recommended defaults:

- transformer depth = 1
- attention heads = 2 or 4
- token dim = 128 or 256
- small FFN expansion ratio

The transformer is NOT intended to become the primary model.
It acts as a lightweight internal dynamics operator.

The transformer token block is not intended to replace CNN representation learning.

Primary semantic extraction should remain within the CNN backbone.

The transformer primarily serves:

- token interaction
- dynamics refinement
- attractor-space evolution

The transformer dynamics block should initially behave close to an identity-preserving refinement operator rather than a strong feature-rewriting module.

Implementation should favor stable residual behavior and gradual dynamics evolution during early training.

Avoid aggressive initialization or overly dominant transformer updates in early epochs.

The transformer dynamics block should initially behave close to an identity-preserving refinement operator rather than a strong feature-rewriting module.

Implementation should favor stable residual behavior and gradual dynamics evolution during early training.

Avoid aggressive initialization or overly dominant transformer updates in early epochs.
Implementation guidance for stable refinement behavior:

- Prefer residual-dominant transformer updates
- Keep residual path strength larger than transformer perturbation initially
- Avoid large attention logits during initialization
- Prefer small initialization std for transformer projections
- Avoid deep nonlinear amplification in FFN
- Preserve feature continuity between CNN features and refined token states

The transformer should refine existing representations rather than overwrite them.

### Classification Head

The classifier may use either:

1. mean pooled token representation
or
2. CLS token

Mean pooling is preferred initially for stability.

Avoid introducing complex ViT-style positional systems unless necessary.

If positional encoding is added:

- keep it lightweight
- avoid large learned embeddings initially

Positional encoding should remain weak relative to learned feature dynamics.

Avoid allowing positional embeddings to dominate token identity or replace semantic feature evolution.

Note that:

    - Mean pooling is the primary recommended configuration.
    - CLS token support exists mainly for future ablation studies and experimentation.
    - Do NOT redesign the architecture around CLS-token-centric behavior.
    - Avoid introducing:
        - large learned CLS embeddings
        - deep CLS interaction schemes
        - ViT-style CLS dominance
    - Avoid introducing complex ViT-style positional systems unless necessary.

### Positional Encoding Constraints

Positional encoding should remain weak relative to learned feature dynamics.

Avoid allowing positional embeddings to dominate token identity or replace semantic feature evolution.

Preferred behavior:

- positional encoding provides weak spatial guidance
- CNN features remain the primary semantic carrier
- transformer dynamics operate mainly on semantic feature interaction

Avoid:

- large learned positional embeddings
- high-magnitude positional injection
- deep positional processing stacks
- positional-dominant token representations

If learned positional encoding is used:

- initialize with small magnitude
- preserve stable early training dynamics
- avoid overpowering CNN-extracted semantics

### Ω-Loss Extension

Phase1 Ω-loss operated on a single hidden vector.

Phase2 extends Ω-loss into token space.

Current objective:

L_attr is computed as the mean squared token-space residual:

L_attr = E[ || H_tokens - sg(T(H_tokens)) ||² ]

averaged across:

- batch dimension
- token dimension
- feature dimension

applied across all tokens.

Important:

- stop-gradient must remain on the transformer branch target
- preserve AMP compatibility
- avoid duplicate forward passes
- reuse already-computed activations

The total loss becomes:

L_total = L_CE_mix + λ_omega * L_attr + λ_IDSI * L_IDSI

where:

- λ_omega initializes from best Phase1 value
- λ_IDSI initializes from best Phase1.2 value

### Layer-IDSI in Phase2

Phase1.2 introduced Layer-IDSI for CNN stages.

Phase2 extends this concept into token dynamics.

Layer-IDSI should continue monitoring:

- CNN backbone stages
- transformer token block
- token projection stage if meaningful

Do NOT compute Layer-IDSI per attention head.
Do NOT compute Layer-IDSI per token individually.

The optimization target remains stable relative hidden-state evolution.

Avoid suppressing useful feature transformation.

For transformer blocks, treat the entire transformer block as ONE monitored dynamics layer.

Do NOT separately monitor:

- attention projections
- attention heads
- FFN sublayers
- LayerNorm submodules

Note that the IDSI should still follow the concept explained in smart.docx

### Required Experiment Configurations

Phase2 experiments MUST include:

1. tokenized architecture WITHOUT Ω-loss
2. tokenized architecture WITH Ω-loss
3. best Phase1 CNN baseline
4. best Phase1.2 CNN + Layer-IDSI baseline

These runs must remain directly comparable.

### Required Metrics

Preserve all existing Phase1.2 metrics.

Additionally log:

- token variance
- inter-token variance
- token norm statistics
- transformer block IDSI
- token diversity statistics

Potential useful metrics:

- mean pairwise cosine similarity between tokens
- token collapse indicators
- token variance across epochs

### Visualization Requirements

plot.py must remain backward compatible.

Add optional token-space visualization panels:

- token variance
- transformer IDSI
- token diversity
- inter-token similarity

Layer-wise plots must preserve stable color mapping.

### New arguments added

- Note: keep all current arguments still valid and no change (if changes are necessary, report to me.).

#### MUST ADD

1. --tokenize
    - Model code: parser.add_argument("--tokenize", action=argparse, BooleanOptionalAction, default=False, help="Enable Phase2 tokenized feature representation")
    - effects:
        - maintain backward compatibility
        - Phase1 command will not become invalid
        - Phase2 command can open individually

2. --token-dim
    - Model code: parser.add_argument("--token-dim", type=int, default=128, help="Token embedding dimension for Phase2")

3. --transformer-depth
    - Model code: parser.add_argument("--transformer-depth", type=int, default=1, help="Number of lightweight transformer blocks")
    - note:
        - Currently 1, but future might extend (for ablation).

4. --attention-heads
    - Model code: parser.add_argument("--attention-heads", type=int, default=4, help="Attention head count for token transformer")
    - effects:
        - token diversity
        - dynamics complexity
        - stability

5. --transformer-mlp-ratio
    - Model code: parser.add_argument("--transformer-mlp-ratio", type=float, default=2.0, help="Expansion ratio for transformer FFN")
    - Note: DO NOT USE default for ViT (4.0), use 2.0 for default instead.

6. --token-pool
    - Model code: parser.add_argument("--token-pool", choices=["mean", "cls"], default="mean", help="Pooling strategy for token classification")
    - effects:
        - Prepare well for ablation

#### MUST ADD, BUT FOR DYNAMICS SPECIFICALLY

1. --token-positional-encoding
    - Model code: parser.add_argument("--token-positional-encoding", choices=["none", "learned", "sinusoidal"], default="learned", help="Positional encoding type for tokens")

2. --token-dropout
    - Model code: parser.add_argument("--token-dropout", type=float, default=0.1, help="Dropout inside transformer token block")

3. --transformer-layernorm
    parser.add_argument("--transformer-layernorm", choices=["pre", "post"], default="pre", help="Transformer LayerNorm placement")

#### MUST ADD, BUT FOR IDSI/ OMEGA EXTENSION SPECIFICALLY

1. --token-omega-loss
    - Model code: parser.add_argument("--token-omega-loss", action=argparse.BooleanOptionalAction, default=True, help="Apply Omega loss in token space")

2. --token-idsi
    - Model code: parser.add_argument("--token-idsi", action=argparse.BooleanOptionalAction, default=True, help="Enable Layer-IDSI monitoring for token transformer")

3. --token-diversity-monitor
    - Model code: parser.add_argument("--token-diversity-monitor", action=argparse.BooleanOptionalAction, default=True, help="Track token diversity metrics")

### Critical Constraints

DO NOT:

- convert entire system into a full ViT
- introduce deep transformer stacks
- introduce heavy positional embedding systems
- destroy Phase1 training stability
- significantly increase VRAM usage
- store full token histories
- introduce extra forward passes solely for metrics

Phase2 should remain practical for consumer GPU training environments, however, it will soon move to an advanced hardware system, so it will definitely expand in the future.

Architectural changes should prioritize:

- low VRAM overhead
- stable batch sizes
- efficient experimentation

The objective is controlled structural evolution, not architectural replacement.

### Success Conditions

Phase2 is successful if ANY of the following occur:

- improved validation accuracy
- improved generalization
- smoother dynamics
- improved stability
- improved token diversity
- improved attractor formation quality

even if gains are moderate.

### Failure Conditions

Failure conditions include:

- token collapse
- unstable transformer dynamics
- exploding Layer-IDSI
- degraded generalization
- degraded stability
- excessive overfitting
- degradation relative to both Phase1 baselines

If instability occurs:

- reduce λ
- reduce transformer dimension
- reduce attention heads
- reduce FFN expansion
- simplify tokenization

DO NOT increase architectural complexity prematurely.

Phase2 is an intermediate dynamics-transition architecture between CNN hidden-state dynamics and future multimodal token-space dynamics systems.

Implementation decisions should prioritize:

- dynamics observability
- stability
- extensibility
- low-complexity experimentation

rather than maximizing transformer capacity.
