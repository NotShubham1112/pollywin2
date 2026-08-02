# PolyWin R2 — v5 Electronic Foundation Model (Design)

Date: 2026-08-03
Status: Approved (awaiting spec review)
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Problem context

7 polymer properties predicted from SMILES. Training data is severely imbalanced:

| Target | Samples | Share |
| ------ | ------: | ----: |
| tg | 4143 | 55.9% |
| egc | 2028 | 27.4% |
| egb | 337 | 4.5% |
| eps | 229 | 3.1% |
| nc | 229 | 3.1% |
| ei | 222 | 3.0% |
| eea | 221 | 3.0% |

The bottom five targets (egb, eps, nc, ei, eea) are data-starved (1238 rows total) and are
the highest-opportunity area: a smarter architecture can separate itself there because the
signal must be shared across targets.

Empirical correlations (computed on polymers with both targets measured):
- egc ↔ egb = 0.94, eps ↔ nc = 0.92
- egc ↔ eea = -0.83, egc ↔ nc = -0.76, egb ↔ nc = -0.82
- tg is statistically disconnected from the rest (shared-polymer overlap < 5%).

## 2. Why v5 exists (v4 defects)

v4 was contaminated:

1. **OOF leak in the multi-task NN.** `train_multitask` trained one model on ALL training rows,
   then computed `oof = model(X_train)` — i.e., in-sample predictions. Reported `mtnn tg = 15.81`
   and `stack tg = 14.99` are optimistic, not out-of-fold.
2. **Partially contaminated stack.** The level-2 meta-model trained on a mix of honest GBM OOF
   and leaky NN predictions, so all stack numbers are inflated.
3. **Degenerate GNN.** On Kaggle's P100 (no CUDA kernels for the GPU image's torch), the GNN fell
   back to CPU and produced near-constant predictions (RMSE 158–324), which the Ridge stack
   down-weighted. Harmless but wasted runtime and stack capacity.
4. **Fig-08 missing on Kaggle.** The cross-target correlation figure read a local file
   (`vault/figures/cross_target_corr.csv`) that does not exist in the Kaggle kernel, so it was
   silently skipped.

Goal of v5: obtain the **first honest OOF benchmark** and validate the Electronic Foundation
Model hypothesis. Every subsequent decision becomes evidence-driven instead of being influenced
by inflated v4 numbers.

## 3. Success criteria

- All base-model OOF scores are genuine GroupKFold out-of-fold predictions (GBMs already are;
  the NN branches become fold-wise).
- Honest tg stack RMSE will likely look *worse* than v4's "14.99" — expected and acceptable.
- Primary signal of success: **EFN beats GBM-only honest OOF on egb / eps / nc / ei / eea.**
- If EFN cannot beat GBMs on the small electronic targets in honest OOF, pivot to
  retrieval augmentation / pseudo-labeling rather than growing the network.
- One Kaggle validation run (~1h) after a local smoke run (bug-check only).

## 4. Architecture (v5)

### 4.1 GPU bootstrap (cell 1)

- `get_torch_device()` already runtime-probes CUDA with a tiny tensor.
- **Change:** on probe *failure* with CUDA reported available, attempt to repair:
  1. `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
     (cu121 supports Pascal sm_60 / P100; the GPU image's cu128 wheel does not).
  2. Hard re-import torch (drop torch modules from `sys.modules`, fresh import).
  3. Re-probe. If still failing, fall back to CPU with a printed warning.
- Best-effort only; wrapped in try/except. CPU fallback must always work.
- Do NOT gate on device capability alone — runtime validation only (Kaggle images change).

### 4.2 Fold consistency (layer 3 / validation harness)

- `folds` is a single global GroupKFold(n_splits=10) over all dedup rows, grouped by canonical
  SMILES. All targets share the same fold partition — this is what makes cross-target OOF
  features fold-safe.
- **Change:** save `folds` (plus the dedup index mapping) to `WORK/folds.csv` on first compute;
  if the file already exists, load it and never regenerate. Every downstream model (GBMs, EFN,
  tgnn, level-1.5 stack, level-2 meta) must use the identical fold assignment.
- **Single production fold definition:** `GLOBAL_FOLDS = 10` for EVERY model (GBMs, EFN, tgnn,
  level-1.5 stack, level-2 meta). Never mix a 5-fold EFN OOF with 10-fold GBM OOF in the same
  stack — that makes cross-target OOF, reliability features, and meta-model features impossible
  to reason about.
- Smoke runs only: `GLOBAL_FOLDS = 5` everywhere. Production: 10 folds always.

### 4.3 Auxiliary physics tasks (new cell after feature factory)

Ten chemistry-derived scores, computed from RDKit Mol objects (NOT from descriptor columns, to
avoid trivial copy-shortcuts), defined for **all 7406 training rows**:

| # | Task | Definition |
| - | ---- | ---------- |
| 1 | aromaticity_score | aromatic atoms / heavy atoms |
| 2 | conjugation_score | fraction of atoms in a conjugated system (RDKit conjugation flags) |
| 3 | sulfur_score | S atoms / heavy atoms |
| 4 | electronegativity_score | mean Pauling electronegativity over atoms |
| 5 | polarity_score | TPSA / heavy atoms |
| 6 | ring_density_score | ring atoms / heavy atoms |
| 7 | flexibility_score | rotatable bonds / heavy atoms |
| 8 | halogen_density | (F + Cl + Br + I) / heavy atoms |
| 9 | hbond_capacity | (HBD + HBA) / heavy atoms |
| 10 | heteroatom_fraction | non-C atoms / heavy atoms |

Each is standardized per fold (mean/std fit on fold-train rows only). Aux targets are used only
at train time to give the shared encoder dense supervision across all 7406 rows. Drop any aux
task with near-zero variance.

### 4.4 Electronic Foundation Network (replaces mtnn + gnn)

- Encoder: `1153 → 512 → 256 → 128` (BN + SiLU + Dropout 0.3) → "polymer state" z.
- 6 real heads (egc, egb, eps, nc, ei, eea) + 10 aux physics heads.
- **Tg excluded from this trunk entirely.**
- Loss: per-target inverse-σ weighted MSE (target standard deviations fit on fold-train rows),
  so every target — especially the small ones — receives real gradient signal. Total loss =
  weighted real-target MSE + λ_aux·aux MSE (λ_aux ≈ 0.3).
- **Per-head target-presence masking.** Each electronic target is present only on its own subset
  of rows; aux tasks are present on all rows. Implement per-head masks
  `mask = ~np.isnan(y_target)` and compute each head's loss only over its present labels.
  Missing target labels are **never imputed**.
- **Honest OOF:** one model per fold; fold-train → predict held-out fold; test predictions
  averaged across fold models. Same global fold partition as GBMs.
- EFN width is fixed at 512/256/128 — do not increase (7406 rows; a larger net is unlikely to help).

### 4.5 Tg isolation

- Dedicated single-target Tg NN (`tgnn`), encoder `256 → 128 → 64`, fold-wise OOF.
- Tg is treated completely separately: "Tg GBM stack + Tg NN + Tg meta".
- No shared trunk with the electronic model, no cross-target features for/from tg.
- tg stack = per-target level-1.5 Ridge over {lgb, cat, xgb, hgb, tgnn} OOF, then level-2.

### 4.6 Reliability features (new, level-2 input)

For every target, from the level-1 OOF predictions across {lgb, cat, xgb, hgb, efn} (and tgnn
for tg):

- `mean_prediction`, `std_prediction`, `max_prediction`, `min_prediction`

Disagreement across models ≈ uncertainty, which meta-models exploit well. Computed from
fold-safe OOF for training and from fold-averaged test predictions for the test set.

### 4.7 Cross-target OOF feature layer (electronic cluster only)

- Electronic cluster = {egc, egb, eps, nc, ei, eea}.
- **tg receives nothing and contributes nothing**, enforced unconditionally (overlap < 5%).
- Level 1: per-target GBMs + EFN + tgnn → honest OOF + fold-averaged test preds.
- Level 1.5: per-target Ridge stack on own level-1 OOF (as v4).
- Level 2: per-target meta-model (Ridge) on:
  - the target's own level-1 base-model OOFs,
  - reliability features (4.6),
  - cross-target level-1.5 stack OOFs for correlated targets,
  - missing-indicator column + mean imputation per cross feature.
- Cross-feature map (derived from data: |corr| ≥ 0.5 and coverage ≥ 0.5):

  | Target | Receives cross-target L1.5 stack OOFs from |
  | ------ | ------------------------------------------- |
  | eps | nc, egc, egb, eea |
  | nc | eps, egb, egc, ei |
  | egc | egb, eea, nc, eps, ei |
  | egb | egc, nc, eea, eps, ei |
  | ei | egc, egb, nc |
  | eea | egc, egb, eps |
  | tg | — |

- Fold-safety argument: `folds` is global and keyed by canon; a polymer with both targets
  measured lands in the same fold for both. Target A's fold-f OOF comes from level-1 models
  that never saw fold f, so using it as a level-2 feature for target B (whose fold-f model is
  also trained on folds ≠ f) leaks nothing.
- Level-2 output = final predictions.

### 4.8 GNN removal

- GNN removed from `BASE_MODELS` and from the runtime path. Code stays in the repo
  (`build_pipeline_nb.py` history / archived cell) for experiments; not in the Kaggle kernel.

### 4.9 Stacking / submission

- `BASE_MODELS` (electronic): `lgb, cat, xgb, hgb, efn`.
- `BASE_MODELS` (tg): `lgb, cat, xgb, hgb, tgnn`.
- `store_key` generalized to per-model target availability.
- Submission mechanics unchanged: level-2 final predictions + physics bounds
  (egc/egb/ei ≥ 0, eps ≥ 1, nc ∈ [1,3], tg/eea unconstrained).

### 4.10 Judge figures

- **Fig-08 fix:** compute the cross-target correlation matrix in-notebook from `Y` (pairwise
  complete observations over shared canonical polymers); save csv + figure. No dependence on
  local vault files.
- Fig-04 / Fig-09 model comparisons use the new model names and level-2 final RMSE.
- All other figures unchanged (01–07, 09).

### 4.11 Artifact persistence

Persist every intermediate OOF artifact to `WORK/` so post-hoc attribution analysis is
straightforward when leaderboard scores move:

    folds.csv                (global fold assignment + dedup index)
    oof_lgb.parquet          (per-target OOF + test preds for each base model)
    oof_cat.parquet
    oof_xgb.parquet
    oof_hgb.parquet
    oof_efn.parquet
    oof_tgnn.parquet
    l15_ridge.parquet        (level-1.5 stack OOF + test)
    final_meta.parquet       (level-2 final OOF + test)

These enable answering: which model added signal, which target improved, did EFN actually
help, did cross-target features help.

## 5. Deferred (post-v5, in priority order)

1. Retrieval augmentation / retrieval features (839/4940 test twins is potentially the largest
   untapped gain — revisit after honest EFN benchmark).
2. Pseudo-label expansion (PI1M, confidence-filtered).
3. Consistency-loss experiments (v5.1) — z → physics heads → reconstruct z. Deferred because it
   assumes the 10 engineered scores span the polymer manifold, which they probably don't, and
   it could suppress encoder information (sterics, topology, motifs).
4. Fragment auxiliary heads (20–50 binary fragment-presence tasks).

## 6. Validation plan

1. Local smoke run (reduced workload) — catches bugs, confirms fold-saving, aux-task variance,
   and level-2 wiring. Not for numbers.
2. One Kaggle run (~1h, GPU bootstrap best-effort).
3. Report honest OOF leaderboard: GBM-only (level-1.5) vs GBM+EFN+level-2 (final). Compare per
   target, with emphasis on egb/eps/nc/ei/eea.
