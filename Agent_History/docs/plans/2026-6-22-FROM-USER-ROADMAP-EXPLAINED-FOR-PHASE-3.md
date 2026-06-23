# This technological file introduce you IMPORTANT giudelines for the WHOLE Phase3:Towards the VLM

## Phase 3 consists of 3 steps, each time you implement, you are ONLY implementing ONE SINGLE STEP that I point for you to implement

## Before you read further, READ THE WHOLE PROJECT CAREFULLY, MAKE SURE UNDERSTAND EVERY PART OF THE PROJECT. If you have any confusions, ASK THE USER IMMEDIATELY

## After you read the project, read specifically these two files:Agent_History\docs\plans\2026-5-22-FROM-USER-Phase-2-Implementation-Instructions.md AND Agent_History\docs\plans\2026-4-24-FROM-USER-roadmap-after-4-24-VERY-IMPORTANT-GUIDE.txt

## The Phase-2 one is what we did last time, and the roadmap one is the very basic instruction until Phase 3 is finished. There MIGHT be some differences between the roadmap file and this instruction file, but please follow 100% what I instructed in this file

## VERY IMPORTANT: This project have some unusual innovations, PLEASE 100% FOLLOW THE DEFINITIONS AND EXPLANATIONS IN THE TECHNOLOGICAL DOCUMENT, which is linked through:"smart.docx - 快捷方式.lnk"

## IGNORE THE CFT STUFF IN smark.docx IN THE WHOLE PHASE3, THAT IS FOR PHASE4, DO NOT IMPLEMENT CFT DURING PHASE3

## Note that the current best model is trained through best_train_commands.txt, NEVER MODIFY THIS FILE

## The project has evolved through several stages

### Phase 1

CNN Classifier

### Phase 1.2

CNN + Omega Loss + IDSI Monitoring

### Phase 2

CNN + Lightweight Transformer Tokens

### Phase 2.2

CNN + Transformer + Full MAOIDL Instrumentation (except for CFT)

## The current best-performing model consists of

CNN Feature Extractor
+
Token Projection
+
Lightweight Transformer
+
Classification Head

where: the CNN Feature Extractor is:

Stage1 CNN
Stage2 CNN
Stage3 CNN

## Current Research Problem

The project has produced an unexpected observation.

During Phase 2 training:

CNN layers and Transformer layers
show different IDSI behavior.

Observed trend:

CNN layers:
Higher IDSI sometimes correlates
with better validation accuracy.

Transformer layers:
Lower IDSI often correlates
with better stability.

This is important because:

IDSI was originally designed
for Transformer dynamics.

Therefore the current hybrid architecture introduces ambiguity.

We cannot determine whether:

IDSI is fundamentally useful for Transformers.

or

IDSI behavior is being distorted by CNN dynamics.

Purpose of Phase 3.1

Phase 3.1 is NOT intended to maximize classification accuracy.

Primary goal:

Remove CNN dynamics completely.

Observe MAOIDL behavior
inside a pure Transformer system.

Research questions:

Does IDSI still correlate
with validation accuracy?

Does Omega Loss still provide
representation stabilization?

How does Transformer-only
dynamics differ from CNN+Transformer?

## Summarative steps for Phase 3

### Phase3.1: Pure ViT Classifier

### Phase3.2: CLIP with dataset turn to image_caption

### Phase3.3: Vision Transformer+Language Model with dataset turn to image+conservation

### Phase3.1: CNN-Transformer Hybrid → Pure Vision Transformer

## Phase 3.1

### Purpose of Phase 3.1

Phase 3.1 is intended to research on the relationship between val_acc and IDSI, meanwhile, get the highest val_acc WITHOUT AFFECTING THE EXPERIMENTS.

#### Primary goal

Remove CNN dynamics completely and turn to a ViT

Observe MAOIDL behavior inside a pure Transformer system.

#### Research questions

Does IDSI still correlate with validation accuracy?

Does Omega Loss still provide representation stabilization?

How does Transformer-only dynamics differ from CNN+Transformer?

### Dataset

The structure of the dataset is UNCHANGED, but the content size is increased.

Classification task remains 13-class image classification.

No captions.

No language data.

No multimodal training.

### Detailed Architecture Requirements (VERY IMPORTANT)

note: torch backend only, numpy backend unchanged.

1. Replace the entire CNN backbone.
Current:
Image
↓
CNN
↓
Tokens
↓
Transformer
↓
Classifier

Target:
Image
↓
Patch Embedding
↓
Transformer Encoder
↓
Classifier (A pure ViT)
2. Recommended Initial ViT Configuration

note: Keep model intentionally small.

1. patch size: 8x8
2. token dimension: 128
3. transformer depth: 4
4. attention heads: 4
5. MLP ratio: 2.0
6. pooling: CLS token

### MAOIDL Compatibility Requirements

note: The following systems must remain functional.

#### Omega Loss

Must remain supported.

Current behavior:

Feature Representation
↓
Omega Projector
↓
Omega Loss

Pure ViT version:

CLS Token
↓
Omega Projector
↓
Omega Loss

#### IDSI Monitoring and IDSI loss

Must remain supported.

Current monitored layers:

stage1
stage2
stage3
token_projection
transformer_token_block

Pure ViT equivalent:

patch_embedding
transformer_block_1
transformer_block_2
transformer_block_3
transformer_block_4

These layer names must be used for metrics, plotting and monitoring.  IF transformer depth is not set to default 4, than the names should be changed accordingly.

#### Hidden Variance Metrics

Must remain functional.

Metrics currently used:

hidden_norm
hidden_variance
h_var_min
h_var_mean

These should continue to be logged.

#### Token Diversity Monitor

Already designed for token systems.

Should be retained without modification.

### Training Configuration

note: Initially preserve existing training regime.

Keep:

1. AdamW
2. Cosine LR
3. EMA
4. Label Smoothing
5. Augmentation
and their arguments EXCACTLY THE SAME.

Disable architectural experimentation until baseline ViT works.

Avoid introducing:

1. New losses
2. New regularizers
3. New attention variants

during initial implementation.

### Plotting

Keep plot ALL the current things, just modify the name of stages into the names mentioned in "IDSI Monitoring and IDSI loss"
Keep the arguments fro plotting.

### Checkpoint Compatibility

Must preserve current checkpoint system.
New checkpoints should additionally save:

1. patch_size
2. vit_depth
3. token_dim
4. attention_heads
5. pool_type

Inference must reconstruct architecture from checkpoint metadata.

Maintain backward compatibility.

### Training argument adjustments

DELETE:

1. --model-width-scale
2. --freeze-bn-affine
3. --freeze-patience
4. --freeze-epoch-num  (meaning that the freezing policy is deleted)

PRESERVE:

1. Omega
    1. --omega-loss
    2. --omega-lambda
    3. --omega-projector-depth
    4. --omega-hidden-dim
    The omega loss is now affecting CLS Token instead of CNN.
2. IDSI
    note: PRESERVE COMPLETELY
    1. --idsi-lambda
    2. --token-idsi
3. Position Encoding
    1. --token-positional-encoding learned

CHANGE:

1. Transformer arguments
    1. --token-dim, set default to 128
    2. --transformer-depth, set default to 4
    3. --attention-heads, set default to 4
    4. --transformer-mlp-ratio, set default to 2.0
2. Token pool
    1. --token-pool set to cls
3. Dropout
    1. --dropout set default to 0.10
    2. --token-dropout set default to 0.10
4. Learning rate
    1. --lr set default to 0.0003

### Evaluation Requirements

Phase 3.1 succeeds if:

1. Pure ViT training is stable.

2. Omega Loss operates correctly.

3. IDSI metrics can be collected.

4. Layer-wise dynamics are observable.

5. Results can be compared directly against Phase 2 hybrid models.

### Failure Conditions

note: You MUST check for these conditions before commit.

Removing Omega support

Removing Layer-IDSI

Removing token diversity monitoring

Breaking plots

Breaking checkpoints

Breaking AMP

Breaking EMA

Reintroducing CNN components

Implementing DeiT/Swin/Hybrid-ViT variants

### MOST IMPORTANTLY, you should

DO NOT ADD NEW RESEARCH IDEAS.

DO NOT IMPLEMENT CFT.

DO NOT IMPLEMENT MEMORY ATTRACTORS.

DO NOT IMPLEMENT SAAL.

DO NOT IMPLEMENT SAP.

## NO ADDITIONAL INNOVATIONS SUCH AS DeiT、Swin、ConvStem、Hybrid Patch Embedding ARE APPROVED, NEVER NEVER ADD THEM TO THE PROJECT

## NOTE THAT FROM HYBIRD CNN TO ViT IS A HUGE LEAP, WHICH LEADS TO THE DANGER OF COMPATIBILITY OF EACH COMPONENT OF THE WHOLE PROJECT. PLEASE BE VERY CAREFUL OF THE RELATIONSHIPS BETWEEN EACH FILE, METHOD, CLASS AND POLICIES

### COMPATIBILITY REQUIREMENTS, VERY VERY VERY IMPORTANT, BE CAREFUL AND MAKE SURE YOU UNDERSTAND THE FOLLOWING LIMITATIONS

1. This phase is an architecture replacement, NOT a project rewrite
2. PRESERVE:Training pipeline, Metrics pipeline, MAOIDL instrumentation, Experiment tracking, Plotting, Checkpointing, CLI interface as much as possible.
3. Existing functionality is considered stable, REDESIGN THE MEANING FOR TRANSFORMER ONLY RELYING ON smart.docx, DO NOT rename them unless absolutely required, DO NOT remove them.
4. Preserve the Training Systems
    1. EMA
    2. AdamW
    3. Cosine Scheduler
    4. Early Stopping
    5. Warmup
    6. Mixup/CutMix
    7. Augmentation Pipeline
Freezing system is DELETED as explained above
5. Preserve the Experiment Systems
    1. JSONL Metrics Logging
    2. Run Information Output (runs/)
    3. Plot Generation
    4. Checkpoint Saving
    5. Checkpoint Loading
6. Preserve the MAOIDL Systems
    1. Omega Loss
    2. Omega Projector
    3. IDSI Monitoring
    4. IDSI Loss
    5. Hidden Variance Monitoring
    6. Token Diversity Monitor
7. Metric Continuity Is More Important Than Architectural Purity
NEVER EVER BREAK Metric Continuity

8. All historical metric names must remain unchanged except:
    stage1
    stage2
    stage3
    which are replaced by:
    patch_embedding
    transformer_block_1
    transformer_block_2
    transformer_block_3
    transformer_block_4

9. You MUST allow Phase2 checkpoints to be loaded, but no need to be still able to train.
    At minimum:
    - metadata inspection
    - architecture detection
    - graceful loading
    must function correctly.
    Loading a Phase2 checkpoint must not crash.

10. MAOIDL Compatibility Has Priority Over ViT Optimization
Whenever there is a conflict between those two, choose MAOIDL Compatibility.

11. Note for the training pipeline:
    Phase 3.1 is a backbone replacement only.
    The training pipeline architecture itself is considered frozen.
    Do NOT redesign:
    - trainer structure
    - epoch loop
    - validation loop
    - metric logger flow
    - checkpoint flow
    unless required by the ViT backbone replacement.
    Backbone replacement should be implemented with the smallest possible code footprint.
12. REMEMBER TO VERIFY AND COMMIT!!
13. THE USER GUI BACKEND SHOULD BE COMPATIBLE FOR BOTH Phase2 and Phase3 Products.
14. IDSI for Transformer blocks must be measured on the block output after the final residual connection. The same hook location must be used consistently for all Transformer blocks.
15. CLS token must be excluded from token diversity calculations. Token diversity should operate only on patch tokens.
16. Historical run directories generated by Phase2 must remain readable by all analysis, plotting and GUI tools. The GUI must not assume that all runs are ViT runs.
17. WRITE TO HISTORY FILES AND UPDATE THE .md FILES BEFORE COMMIT.
18. Because the CNN backend in Phase2 will NOT be trainable anymore, the git history becomes very important

## Time to say goodbye to CNN. (Observe Silence.)

## Phase 3.2

Not yet written

## Phase 3.3

Not yet written
