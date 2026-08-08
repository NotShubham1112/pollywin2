# PolyWin R2 — v16 Cross-Target Decoder (Design)

Date: 2026-08-08
Status: **DESIGN (pre-registered) — not yet implemented.**
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v16 exists

P14 (full-PI1M pretraining, LB **0.883**, blend OOF **0.8769**) is the frozen production
submission. The v13/v15/postmortem lines closed the fine-tune, blend-reweighting,
pretrain-scale, and pseudo-label levers:

- v15 loss-reweighting (x2 eps/nc): **FAIL** (−0.0051 mean) → STOP.
- Blend reweighting: **exhausted** — oracle best-arm (0.8723) < P14 ridge (0.8769).
- Pseudo-labeling (v8): OOF +0.017 → LB −0.019 → dropped.
- Pretraining scale: P14 *is* the full-995k-10-epoch run (+0.006).

A research report proposed target-specialist reweighting as the top remaining candidate;
that was audited and **already shipped in P14** (its per-target Ridge learns exactly those
weights), then proven to contribute nothing further. **None of these levers touch the
strongest unused signal in the data: the measured other-property values available at test
time for the same polymer.**

For the **small-five** targets (eps, nc, ei, eea, egb — precisely the two weakest, eps/nc,
plus the narrow-variance electron/bulk targets), between **88% and 98% of test rows** have
their exact canonical molecule present in **train under a different `target_type`** (quick
mines: eps 98.7%, nc 98.7%, ei 98.0%, eea 98.0%, egb 88.4% of test rows). The true values
of those sibling properties are therefore known at test time, with zero model error, using
**train labels only**.

Physics relations between the properties are strong in our own data:

| relation | n pairs | measured |
|----------|---------|----------|
| `egc ≈ ei − eea` | 59 | mean \|Δ\| = 0.072 |
| `egb ≈ egc` | 175 | corr **0.963** |
| `nc² ≈ eps` | 134 | corr **0.925** |

P14 blends only `{GNN, GBM}` per-target. It never feeds the true sibling values (its
"leak-safe twins" feed per-target LGBM **predictions**, not true labels). v16 adds a
**post-hoc, fold-safe Cross-Target Decoder stage** that converts the known-sibling lattice
+ physics into new blend arms. **Top-10 (0.898) is declared a stretch target; the honest
bar is ≥ 0.886 (top-20 zone).**

**Coverage asymmetry (pre-measured, drives the evaluation design):** train has only **6%**
multi-labeled polymers (~415/6565) while test has **92%** small-five coverage. The decoder
arms therefore carry most of their weight at **test time**; train-side OOF of the arms is
only measurable on the ~415 multi-labeled polymers. §3 therefore evaluates the arms on the
**arms-covered OOF subset explicitly**, and reports per-arm standalone R² on the pair
subsets (egb↔egc, nc↔eps, ei−eea↔egc) as a diagnostic — mirroring the v13 plan's
leakage-only baseline.

## 2. The single change

A **post-processing decoder stage glued between level-0 predictions and the per-target
Ridge blend** in the frozen P14 pipeline. GNN, GBM trio stack, folds (GroupKFold on
canonical smiles), GNN seeds 42/999/2025, descriptors, pretrained encoder, submission path
are all **bit-identical to P14**.

Three new candidate arms are produced and fed into the same per-target folded Ridge:

- **Arm 1 `physics_imp` — physics/learned-pair imputation.** For each small-five test row,
  if a physics-sibling value is known for the polymer from train:
  - `egc ← ei − eea` (where both known);
  - `egb ← linear_fit_on_pairs(egc)` (fitted on the 175 train pairs);
  - `eps ← linear_fit_on_pairs(nc²)` (fitted on the 134 train pairs);
  - otherwise → target train mean (inert placeholder).
  Fits are done on **train pairs only** (no test leakage).

- **Arm 2 `learned` — learned cross-target regressor.** Fold-safe per-target Ridge/LightGBM
  downstream on `{known sibling true values, physics-built candidate, RDKit descriptors}`,
  trained **only on the multi-labeled train polymers** with the **same GroupKFold** (all
  rows of a polymer stay in one fold → the sibling features for a held-out polymer are
  never in the training pivot). Test/inference uses the full-train pivot (legal: train
  labels only).

- **Arm 3 (optional, gated) — test-time physics self-consistency projection.** For the
  **1353 test rows / 548 molecules** that appear under multiple targets **inside test**,
  softly project the blended predictions so `egb≈egc`, `nc²≈eps`, `egc≈ei−eea`. Enabled
  **only** if a fold-safe offline mirror of the projection (applied to held-out OOF
  predictions) improves OOF; otherwise it is silent and skipped.

The **blend line** extends from `Ridge(X=[GNN, GBM])` to `Ridge(X=[GNN, GBM, physics,
learned])`, with alphas searched over the **same grid as P14** (`[0.1, 0.5, 1.0, 2.5,
5.0, 10.0, 25.0]`). Because weights are tuned on sibling folds, an arm that cannot help
is driven toward weight 0 → **the blended Ridge cannot regress below P14 (0.8769)** by
construction.

## 3. Baseline & success criteria (pre-registered)

- Reference: **v14** (its verbatim parent), same offline compare protocol as
  `vault/compare_p14.py` (GroupKFold(5) on canonical smiles, per-target R², ridge
  blend over the same alpha grid). The arms are evaluated **on the ~415 multi-labeled
  polymer rows** (the only rows where a held-out polymer's siblings exist within other
  folds) plus the physics-pair subsets reported separately.
- **Pass (honest):** small-five weighted-mean blend OOF on the multi-labeled-polymer rows
  improves by ≥ **+0.003** vs v14 AND no target regresses > −0.003. LB ≥ **0.886**
  confirms.
- **Stretch (declared, not gated):** public LB ≥ **0.898** (top-10). Explicitly not a
  fail condition.
- **Fail → STOP:** small-five-mean on covered rows < +0.003, or any target regresses
  > −0.003, or the physics arms contribute ≤ 0 at runtime. → keep **P14 (LB 0.883)**,
  no v16 submission slot spent.
- **No pseudo-labeling. No architecture change. No new libraries (OSI-approved only).**

## 4. Deliverables

- `build_v16_kaggle_nb.py` (fork of `build_v14_kaggle_nb.py` + the decoder stage inserted
  at the level-0→blend junction as a CORE substitution; arms built from train-only
  pivots; Ridge widened to 4 arms; submission path unchanged) →
  `PolyWin_R2_v16_cross_target_decoder.ipynb`.
- `tests/test_v16_kaggle_nb.py` (TDD; asserts the decoder stage is present AND that every
  non-decoder cell is bit-identical to v14; cells compile; smoke subset).
- Smoke run locally, then push to Kaggle (`polywin-r2-v16-cross-target-decoder`), full
  run, download, evaluate vs P14, decide per §3.

## 5. Result

- Kernel: `shubhamkambli11/polywin-r2-v16-cross-target-decoder` (private, notebook, GPU+Internet).
- Gate (offline, on arms-covered 1297 rows from the run's `blend_oof_test16.npz`):
  - small-five weighted gain **+0.0118** (need ≥ +0.003) — PASS
  - worst per-target delta **−0.0016** (need > −0.003) — PASS
  - per-target deltas: eea −0.0016, egb +0.0092, egc +0.0005, ei +0.0155, eps +0.0190, nc +0.0169, tg +0.0028.
  - Gate verdict: PASS → submitted v16.
- Public LB: **0.874** (vs P14 0.883, honest bar 0.886). **LB does NOT confirm the offline gate.**
- Arms at runtime (`v16_blend_report.csv`): learned arm weighted (w_LEARN) positive on all targets
  except eps (−0.1); physics arm (w_PH) positive on egb/eps only, ~0 elsewhere.
- **Decision: FAIL → P14 (0.883) stays production. v16 does not supersede.** Offline +0.0118 on the
  multi-labeled subset did not transfer to the full public LB, and the single v16 slot is spent.
- Lesson recorded in `docs/lab-postmortem-2026-08-08.md` §5: cross-target arms over-fit the
  multi-labeled subset (1297 rows); promise offline does not guarantee public LB at this scale.

No post-hoc edits to §3 gates beyond recording the run's numbers above.