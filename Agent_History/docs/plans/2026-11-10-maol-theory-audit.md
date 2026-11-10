# 2026-11-10 — MAOL.docx theory-vs-code audit

## Task

Check that the implementation and the training process fully align with the MAOL
theory in `MAOL.docx` (SHA256 `1FB1E226C1CFB87D2E0A4D0E625D9D8959DF1CE457832878D5F0D49273DDC47D`,
mtime 2026-10-30 23:13).

Scope: `maol/head.py`, `maol/losses.py`, `maol/stats.py`, `maol/tree.py`,
`backends/torch/model.py`, `backends/torch/train_backend.py`,
`utils/training.py`, `config.py`, `predict.py` / `train.py`.

Snapshot: post-fix tree on 2026-11-10; the two statistics fixes and this report
were committed together as `7aa34b9` after the audit draft (the audit itself was
read-only; the fixes came from a concurrent implementation pass).

## Method

1. Read the authoritative text and inline OMML equations (`.maol_tmp/MAOL_theory_wrapped.txt`,
   P00–P19). Because flat concatenation destroys bracket/superscript grouping,
   every formula involving `max(0, ...)`, squared norms and Lambert-W was
   re-extracted from the docx OMML tree with `python-docx` (walking `m:t`,
   `m:sup`, `m:sub`, fractions, n-ary operators and delimiters).
2. Walked the implementation file by file and matched each theory clause to
   specific lines; ran one read-only numerical probe of the head's geometry
   losses (no files created, no training).
3. Classified each difference as (a) docx-extraction ambiguity, (b) deliberate
   engineering deviation, or (c) real implementation mismatch.

## Verdict

**The implementation is theory-faithful; no major deviations.** The six losses,
the prototype head, the tree geometry, the warm-up procedure, the τ rules and
the margin/statistic derivation match `MAOL.docx`. Two formula suspicions from
the flat-text reading (that `L_sep` / `L_cov` used squared distances) were
extraction artifacts, not code bugs (§2).

Counts: **0 major / 3 minor / 1 theory defect / 10 aligned / 1 not-applicable**.

During the audit a concurrent pass fixed two statistics-update mismatches that
this audit also identified (`tau_warm` units, `n_min` gating of `c_EMA`); the
tree audited here is the post-fix state, and the two items are marked
"aligned (fixed during audit)". The auditor's own mandate was read-only, so
these edits are not attributed to this report.

## 1. Status matrix

| # | Theory point | Status | Evidence | Note |
|---|---|---|---|---|
| A | Prototype head replaces the linear head; only leaf centers in forward | Aligned | `backends/torch/model.py:309-318`, `maol/head.py:188-198` | `head=PrototypeHead(...)` in MAOL mode, `fc2` only for the CE control; `logits()` uses `centers_hat` only |
| B | L2 normalization of `h` and centers (not in the theory) | Minor deviation (user-approved) | `model.py:441-450`, `maol/head.py:100-103` | Approved earlier; not a bug, but it changes the unit of every distance formula (root cause of the old τ_warm error) |
| C | τ = 1.0 for 100 iterations, then `τ_warm = mean‖h−c_y‖²`; then `τ0 = d_intra²` fixed | Aligned (fixed during audit) | `train_backend.py:67,455-473,644-690`, `maol/stats.py:118-127,282` | `tau_warm` now measured to the normalized center (the fix); `τ0` fixed for the whole run; optional `{0.5,1,2}τ0` set not implemented (optional in docx) |
| D | Warm-up: centers recomputed each epoch, frozen, CE only; then unfreeze | Aligned | `train_backend.py:644-645,661-679,746-750` | `set_trainable(False)` before the loop, `set_centers(means)` each epoch, `set_trainable(True)` after |
| E | `L_intra` with unit leaf weight and ancestor chain excluding the leaf | Aligned | `maol/head.py:203-217`, `maol/tree.py` ancestor chain | Leaf term weight 1; chain starts at the parent; β = γ^Δdepth, γ=0.5 |
| F | `L_sep` hinge on the distance, margins via Lambert-W, `w_vu` from validation confusion | Minor deviation | `maol/head.py:219-251`, `maol/stats.py:238-259` | Functional form aligned; margins and `w_vu` are computed **once** after warm-up, while the docx says "we calculate W(k_vu) every epoch"; 8 Newton iterations vs 4–5 (harmless) |
| G | `L_cov = (1/|C_b)|Σ(max(0, q_c−r_c))²`; q EMA 0.9, `n_min=5`, q stop-grad | Aligned | `maol/head.py:268-274,304-311`, `maol/stats.py:190-206` | OMML confirms the squared hinge on the radius gap (not `q−r²`); q is a buffer so no grad |
| H | `L_nest` squared hinge with parent center/radius derived from children | Theory ambiguity / defect | `maol/head.py:152-183,253-266` | Code reproduces the formula exactly, but the docx parent-radius definition makes the term **identically zero**; `λ3` is inert (see §3.4) |
| I | `L_center = (1/|C_b|)Σ‖c_c − sg(c_c^EMA)‖²` | Aligned | `maol/head.py:276-281` | Denominator = present leaves |
| J | `L = L_CE + λ1L_intra + λ2L_sep + λ3L_nest + λ4L_cov + λ5L_center` | Aligned | `maol/losses.py:57-101`, `train_backend.py:60-66,1883-1887` | Defaults all 1.0; commands use 1/2/1/0.5/0.5, which the docx explicitly allows by tuning |
| K | Effective-count denominators | Aligned | `maol/losses.py:57-101`, `maol/head.py:251,274,281` | CE/intra over N, sep over valid pairs, cov/center over present leaves, nest over non-root nodes |
| L | `c_EMA ← m·c_EMA + (1−m)·mean`, only present classes; q gated on `n_min` | Aligned (fixed during audit) | `maol/head.py:312-320` | The gate now applies to `q` only, as the docx says |
| M | Tree from nested directories, node weights from train counts, only leaf geometry trainable, probe diagnostic | Aligned | `maol/tree.py`, `data/nested.py:44-105`, `head.py:42-43`, `train_backend.py:921-1078` | Probe is an implementation extra, not in the theory |
| N | Engineering additions not in the theory | Not applicable | `train_backend.py:510-598,455-473,1147-1175,1818,1856,1858` | Sanity gate, τ floor fallback, weight EMA (off; commands 0.9999 vs default 0.9993), clean-data warm-up, grad clip, optional class weighting, 2× warm-up LR |
| O | Metrics reported per epoch | Minor deviation (reporting only) | `train_backend.py:1374-1379,358-410` | `mean_leaf_center_norm` from raw `head.centers`; `d_intra`/`d_inter`/`margin_cap` are static; loader losses are batch-averaged |

Legend: Aligned = matches the docx (possibly after the concurrent fix);
Minor deviation = real but bounded difference; Theory ambiguity = docx problem,
code faithful; Not applicable = outside the theory's scope.

## 2. Formula verification

The flattened extraction merged sub/superscripts; OMML re-extraction fixed the
reading of the two formulas most at risk:

- `L_sep = (1/|P_valid|) Σ w_vu [ (max(0, m_vu − ‖h̄_v − h̄_u‖))² + η (max(0, m_vu − ‖c_v − c_u‖))² ]`
  — the hinge is on the **distance** and the hinge is squared; code
  `F.relu(margins − batch_distance) ** 2` matches exactly
  (`maol/head.py:247-250`).
- `L_cov = (1/|C_batch|) Σ (max(0, q_c − r_c))²` — squared hinge on the radius
  **gap**, not `q_c − r_c²`; code `(F.relu(gaps) ** 2).mean()` matches
  (`maol/head.py:268-274`).
- `w_vu = clip( w_vu / (q90(w_v'u' sibling pairs) + ε), 0, 1 )` — ε is in the
  denominator of q90, and the code is `clamp(w_bar/(q90 + 1e-12), 0, 1)`
  (`maol/stats.py:246-248`).
- `L_nest` = squared hinge with `sg(c_parent)`, `sg(r_parent)`; parent center
  `Σn_u c_u / Σn_u`, parent radius `max_u(‖c_u − c_parent‖ + r_u) + ε`
  (`maol/head.py:152-183,253-266`).

## 3. Detailed findings

### 3.1 B — L2 normalization (approved deviation)

`model.forward_embedding` returns `F.normalize(raw, p=2, dim=-1, eps=1e-6)` in
MAOL mode (`backends/torch/model.py:441-450`), and `centers_hat` normalizes the
centers on every read (`maol/head.py:100-103`); `set_centers`,
`initialize_from_stats` and `update_ema_stats` renormalize too
(`head.py:110-120,122-136,312-320`). The docx contains no normalization step,
so all theory formulas assume Euclidean distances between unconstrained
vectors. The effect is that every distance is a chord distance (`≤ 2`), radii
live in `[0, 2]`, and `d_intra`, `d_inter`, `q` and `τ0` are all in these units.
This was explicitly approved earlier (it fixed the 2026-10-31 feature-scale
explosion), so it is a deviation, not a bug; it is the reason `tau_warm` had to
be fixed.

### 3.2 C — τ rules

- `MAOL_WARMUP_TAU_STEPS = 100` (`train_backend.py:67`).
- Warm-up starts with `head.tau.fill_(1.0)` (`:645`); each epoch computes
  `means, counts, tau_epoch = compute_class_means(...)` (`:663-669`) and
  `_resolve_warmup_tau` returns 1.0 for the first 100 steps, then `τ_warm`
  (`:455-473`). The switch is applied inside the epoch at step 100 (`:689-690`).
- After warm-up `apply_warmup_stats` sets `τ = τ0 = max(d_intra², 1e-6)`
  (`maol/stats.py:282,306-309`); grep shows τ is never written again, so it is
  fixed as the docx requires.
- Checkpoint round-trip: `tau` is a persistent buffer (`maol/head.py:54`) and is
  also stored in `save_weights` metadata (`model.py:468-492`); `_load_warmup`
  requires `embedding_norm == "l2"` and re-applies `WarmupStats`
  (`train_backend.py:881-918`), so `τ`, `q`, `c_EMA`, margins and node weights
  all round-trip.
- The optional tuning set `{0.5τ0, τ0, 2τ0}` is not implemented (no CLI flag);
  the docx calls it optional, so this is not a mismatch.
- P08 says "During warm-up, τ is fixed at 1.0"; P12 says "fixed at 1.0 for the
  first 100 iterations; after that it is set to τ_warm". The code follows P12
  (the more specific paragraph). This is a docx-internal conflict, recorded in
  §5 item 1.
- `_resolve_warmup_tau` falls back to 1.0 when `τ_raw < τ_floor` (default
  `1e-2`, `:455-473,1859`) — an engineering guard for collapsed class means,
  not in the theory (item N).

### 3.3 F — separation margins are static

`compute_warmup_statistics` builds `w_vu` from a single validation confusion
matrix (`maol/stats.py:181-188,238-250`), converts it to margins with Lambert-W
(`:252-259`) and `apply_warmup_stats` installs them as buffers
(`maol/head.py:138-143`). Full training only reads those buffers
(`head.py:242-250`). The docx says *"we calculate W(k_vu) every epoch"*, with
`w_vu` defined from the validation confusion of the current model, so a strict
reading asks for a per-epoch refresh; the code deliberately computes both once
to avoid a validation pass per epoch. Consequence: the margin/weight schedule
does not follow the model's improving confusion. The functional form, scales
(`margin_scale=1.25`, `mbase_scale=0.9`, `mmax_scale=0.8`), `k0`, `α`, `q90`
normalization, clipping and the `η=0.1` center term all match; `λ2=2.0` in the
commands amplifies the static margins but is accepted tuning. `lambert_w` runs
8 Newton iterations (`stats.py:298`) versus "4–5" in the docx — an accuracy
improvement, converged values identical.

### 3.4 H — `L_nest` is identically zero (theory defect)

The docx defines the parent radius as
`r_parent = max_{u∈children}(‖c_u − c_parent‖ + r_u) + ε`. For any child `v`,

```
gap_v = ‖c_v − c_parent‖ + r_v − r_parent
      ≤ max_u(‖c_u − c_parent‖ + r_u) − r_parent
      = −ε
```

so `max(0, gap_v)² = 0` for every non-root node, including the leaf case the
docx highlights ("For a leaf node c, L_nest contributes through its role as a
child node"): the max includes the child itself, hence the leaf can never be
outside its parent's ball. The code reproduces the formula faithfully
(`head.py:152-183` radius; `:253-266` detached parent terms, `F.relu(gap)²`),
and a read-only probe on `["animal/cat","animal/dog","plant"]` returned
`nest_loss() == 0.0` with gaps `[-9.54e-07, -1.572, -9.54e-07, -1.716]`
(negative by construction). Therefore `λ3 L_nest` has no effect on gradients or
on the reported loss; this is a theory-side vacuity, not an implementation bug.
The docx also says "For non-leaf v, c_v and r_v are derived quantities;
gradients from L_nest therefore propagate to their children" and offers an
optional stop-gradient for non-leaf nodes — but with the current radius
definition no gradient can flow at all.

### 3.5 O — reporting-only observations

`train_backend.py:1374-1379` logs `mean_leaf_center_norm` from
`torch.linalg.vector_norm(model.head.centers)`, i.e. the **raw** parameter
(norm slightly below 1 because the head uses `centers_hat`), while the model
classifies with the normalized centers. The same block logs `d_intra`,
`d_inter` and `margin_cap` from the static `WarmupStats`; they do not change
during training by design. `evaluate_loader` (`:358-410`) averages per-batch
means (`totals / batches`) rather than per-sample values, so class-imbalanced
batches weight differently from the training losses. None of this affects
optimization.

### 3.6 N — engineering additions (outside the theory)

Listed for completeness, not as violations: warm-up sanity gate with
`SystemExit(2)` (`:510-598`, `--maol-warmup-sanity` default on, `:1860`);
τ-floor fallback (`:455-473`); warm-up on clean (non-augmented) data by default
(`:1858`, `:663-669`); model-weight EMA, off by default, `--ema-decay 0.9993`
(`:1147-1175`, `:1818`) while the documented commands use `0.9999`; gradient
clipping 1.0; optional class-weighted CE (`:1806`, default off); warm-up LR
`2·lr[0]` (`:646-650`, with a stale `--maol-warmup-lr` help text claiming
`lr[0]` from a pretrained init, `:1856`); AMP, RandAugment, phase LR schedules,
early stopping. No label smoothing, focal loss or MixUp/CutMix is active — the
only occurrence is a note at `:1420`.

## 4. Changes made during this audit (concurrent pass, not by this auditor)

Two real mismatches were found by both passes and fixed in the working tree at
10:04:49–10:04:58 (before this report was finalised):

1. **`tau_warm` measured to the raw class mean** — `maol/stats.py:118-127`.
   The code now computes
   `(Σ‖h‖² − 2Σ_c‖Σ_{i∈c}h_i‖ + N)/N`, i.e. the mean squared distance to the
   normalized class mean actually used by the head. Before the fix the value was
   `1−ρ²` instead of `2(1−ρ)`, a factor `(1+ρ)/2` (≤ 2×, largest when classes
   are not yet concentrated), making warm-up CE slightly sharper than intended.
   The fix affects warm-up only.
2. **`c_EMA` gated by `n_min`** — `maol/head.py:283-320`. The gate now guards
   only the quantile statistic `q` (`:304-311`); `c_EMA` is updated for every
   class present in the batch (`:312-320`), matching the docx, which says
   "otherwise q_c is left unchanged".

Both fixes, their tests and this report are committed as `7aa34b9`
("Align warm-up tau and c_EMA updates with MAOL theory").

Consequences for this report: line numbers in `maol/head.py` (322 → 329 lines),
`maol/stats.py` (309 → 318 lines) and `tests/` moved slightly; every citation
above refers to the post-fix tree. The two items are marked "aligned (fixed
during audit)" in §1. Since these are statistics changes, not loss-formula
changes, the formulas verified in §2 are unaffected; the next training run
should be treated as a new baseline.

## 5. Unverified / ambiguous

1. **P08 vs P12 (τ during warm-up)** — P08 says τ is fixed at 1.0 for the whole
   warm-up; P12 says 1.0 for the first 100 iterations and then `τ_warm`. The
   code follows P12. This is a docx-internal conflict, not a code bug; the
   choice affects warm-up dynamics only.
2. **`L_nest` intent** — the formula is vacuous as printed; it is impossible to
   tell from the docx whether the intended term was a margin-based hinge (e.g.
   `max(0, ‖c_v−c_p‖ + r_v + m − r_p)²`), a radius defined without the child
   itself in the max, or a term meant to be dropped. The code cannot be judged
   wrong against an unsatisfiable formula.
3. **Per-epoch margin refresh** — the docx says W is computed every epoch but
   does not say whether `w_vu` (and hence the validation confusion pass) is also
   refreshed. The code computes both once; whether a per-epoch refresh was
   intended is not stated.
4. **Order of q and c_EMA updates within a batch** — the docx does not specify
   whether the quantile is taken against the pre-update or post-update `c_EMA`.
   The code uses the pre-update value (`head.py:305-311`), which is the natural
   reading of "distances to `c_c^EMA`".
5. **`n_min` and `c_EMA`** — resolved: the docx gates only `q`; the code now
   does the same.
6. **Optional τ tuning set** `{0.5τ0, τ0, 2τ0}` — not implemented; optional in
   the docx, so not a mismatch, but no experiment can select it without code
   changes.
7. **Newton iteration count** — 8 vs "4–5"; convergence is not measured, only
   argued from the quadratic Newton rate (the values agree to float precision in
   the tested range).
8. **`d_intra` definition** — the docx defines `d_intra` as the mean distance
   and then sets `τ0 = d_intra²`, i.e. the square of a mean rather than a mean
   square; the code follows the docx literally (`stats.py:208,282`). This is a
   theory-definition quirk, not a deviation.
9. **Line-number drift** — `train_backend.py` was edited concurrently during the
   audit (2125 lines at the end); citations quote the code so they can be
   re-located if lines move. `plot.py` / plotting code was intentionally not
   inspected.
10. **Docx text vs OMML** — the flattened text of `L_sep`, `L_cov`, `w_vu` and
    the parent radius is misleading; all conclusions here use the OMML tree.
    The extraction script itself is not committed (read-only audit), so the
    OMML reading cannot be re-run from the repo alone.

## 6. Recommended minimal fixes

Only items that are real mismatches or defects; no refactors.

1. **Decide the P08/P12 conflict in the docx** (or state that P12 supersedes
   P08) — the code cannot satisfy both. Documentation-only.
2. **Fix or drop `L_nest` in the docx** — with the printed parent-radius
   definition the term is identically zero. If a nesting penalty is wanted,
   either add a positive margin, exclude the child itself from the max, or set
   `λ3 = 0` explicitly. Code change only if a non-vacuous term is defined; the
   current implementation is faithful.
3. **Refresh margins per epoch if the docx sentence is normative** — this needs
   a validation confusion pass per epoch (`stats.py:238-259` would have to move
   into the training loop) and re-tuning of `λ2`; otherwise document the
   one-shot choice as a deliberate deviation.
4. **Reporting: log `mean_leaf_center_norm` from `centers_hat`** (or label it
   "raw parameter norm") and either drop the static `d_intra`/`d_inter` from
   per-epoch metrics or mark them as constants (`train_backend.py:1374-1379`).
5. **Fix the stale `--maol-warmup-lr` help text** (`train_backend.py:1856`):
   the code always uses `2·lr[0]` when the flag is absent.
6. **Document the EMA default mismatch** (`--ema-decay 0.9993` vs the commands'
   `0.9999`) so a bare `--ema` run does not silently differ from the documented
   recipe (`train_backend.py:1818`).

## 7. Validation

- `python -m pytest tests -q -p no:cacheprovider` → `49 passed` (reported by the
  concurrent pass that made the two statistics fixes).
- The OMML reading was cross-checked against the flattened text for every
  formula cited above.
- One read-only numerical probe: `nest_loss() == 0.0` on a three-leaf tree;
  `L_intra` matched a hand computation for a two-leaf tree (leaf term 1 +
  `γ·0.5²·…`).
- No training run was executed.

## 8. Conclusion

The code is a faithful implementation of `MAOL.docx`, including the two formulas
whose flattened text looked wrong. The only theory-level defect is `L_nest`,
which is vacuous as printed and therefore inert in the code; the only real
engineering deviations are the approved L2 normalization, the static
margin/statistics schedule, and reporting-only metrics. The two statistics
mismatches (`tau_warm` units, `n_min` gating of `c_EMA`) were corrected in the
working tree during the audit; no loss formula needed changing.
