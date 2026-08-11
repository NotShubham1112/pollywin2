# PolyWin R2 — v21 Leak-Safe Ridge Sibling Arm (Design)

Date: 2026-08-11
Status: **DESIGN (pre-registered) — not yet implemented.**
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v21 exists

P14 (full-PI1M pretrain, public LB **0.883**, honest OOF **0.8641**) is the frozen final
submission. The post-mortem (`vault/final_status.md`) closed every later attempt v15–v20:

- v15 loss-reweighting (x2 eps/nc): **FAIL** (−0.0051) → STOP.
- v16/v17/v18/v19 sibling/cross-target family: **FAIL on LB** (0.874/0.862/0.864/0.874) —
  root cause: the sibling pivot was built from **full train true labels** and reused
  unchanged inside GroupKFold val folds → label leakage. Leak-eligible val rows were
  ~100% for eps/nc/egb/ei/eea/egc (tg ~40%).
- v20 self-trained SMILES transformer: **FAIL** — 3rd arm near-collinear with GBM+GNN.

The forensics/gap analysis ranks a **leak-safe rebuild** of the sibling idea as the
single highest-EV remaining experiment: the cross-target correlational signal is real
(egc↔egb corr 0.963, nc²↔eps corr 0.925), every prior failure was a **leakage bug**, not
a no-signal result, and the fix is precisely diagnosed: **use model OOF predictions of the
other targets as the sibling features, never true labels**.

v21 therefore implements the sibling arm **leak-safe by construction**, as a purely
additive third column in P14's per-target Ridge blend. Base GBM trio + MT-GNN are
bit-identical to P14.

## 2. Rules scope (pre-audited)

- Sibling features = **model predictions only** (per-target LGBM OOF). No true train labels
  enter any feature path. No test-row same-polymer train-label lookup (no "physics
  recipes", no retention/imputation at test time). All of this stays strictly within the
  written rules (§4: no hand-labeling / human prediction of test records).
- No external data and no uploaded artifacts: everything (incl. PI1M pretraining) is
  produced inside one notebook run (§6.2.1, §6.2.4).
- Notebook-only, pinned-version, host-shared submission (§6.2.2, §7.1, §7.2).
- **Out of scope / rejected:** frozen external transformer embeddings (ChemBERTa-2 /
  MoLFormer — violates §6.2.4), GHM/loss-reweighting (falsified by v15), non-linear
  Level-2 meta (falsified by v7/v12), seed-scaling heterogeneous GNNs (contradicts the
  near-ceiling blend finding).

## 3. The single change

A **leak-safe sibling arm** computed between level-0 predictions and the per-target Ridge
blend, then added as a third column. GNN, GBM trio stack, folds (GroupKFold on canonical
SMILES), GNN seeds 42/999/2025, descriptors, pretrained encoder, submission path are all
**bit-identical to P14**.

**Sibling feature source (`twin_scores`).** Reuses `leak_safe_oof_scores()` already in
`mt_gnn_v2.py:285-312`: one global GroupKFold on canon assigns each row to a fold; for
each target `u`, a per-target LGBM is trained on the **out-of-fold** target-`u` rows and
scores every row in the in-fold. Result: a 7409×7 matrix where
`twin_scores[i, u]` = target-`u` LGBM's prediction on row `i`, trained **without** row
`i`'s canon group. Fold-bagged test predictions `lgb_test_te` (4940×7) provide the
test-time sibling features.

**Arm construction (per target `t`):**
- Train features (rows of target `t`): `twin_scores[:, u]` + miss flags for all `u ≠ t`
  (12 columns; no self column). NaN imputed with `TARGET_MEAN[u]`, miss flag = 1.
- Test features: `lgb_test_te[:, u]` + miss flags for `u ≠ t`, same imputation.
- Learner: per-target `Ridge`, alpha tuned by inner GroupKFold OOF over
  `ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]`, refit on full OOF for test.
- Own OOF produced under GroupKFold over target-`t` rows → `sib_oof[t]`, `sib_test[t]`.

**Blend line** extends from `Ridge(X=[GNN, GBM])` to `Ridge(X=[GNN, GBM, SIB])`, alphas
searched over the same grid as P14. Ridge self-regularizes: a collinear/no-value SIB arm
is shrunk toward weight 0 → the blend cannot regress below P14 by construction (verified
by the per-target `w_SIB` report).

## 4. Leak-safety (by construction + audited)

- `twin_scores[i, u]` is produced by a model that never saw row `i`'s fold for any target
  (GroupKFold on canon puts all 7 rows of a polymer in one fold) → a val-fold row's sibling
  features contain **no label of its own fold, for any target**. The v16–v19 pivot-leak is
  structurally impossible.
- The sibling Ridge's own OOF excludes the val fold's labels. No label path exists.
- **Audit (gate 1):** rerun the v19-style leak-eligibility check — for every val fold,
  exact-match count between sibling features and the polymer's true other-target labels
  must be **0**.

## 5. Baseline & success criteria (pre-registered)

Reference: **P14** (`vault/pipeline_out_pretrain/superblend_oof.npz`), same offline
compare protocol as `vault/compare_p14.py` (GroupKFold(5) on canonical smiles, per-target
R², Ridge blend over the same alpha grid).

- **Gate 1 — leak audit:** sibling-feature ↔ true-label exact-match count = **0** across all
  val folds.
- **Gate 2 — OOF gain (distribution-shift emphasis, the check v16 skipped):** blend OOF mean
  over **{eps, nc, ei}** ≥ P14 + **+0.003** (these are the starved targets the arm targets),
  **and** overall blend OOF mean ≥ P14 + **+0.003**.
- **Gate 3 — worst-target guard:** no per-target OOF regression > **−0.003** vs P14.
- **Confirmatory (not gated):** public LB ≥ **0.886** (top-20 zone) on the Kaggle run; the
  notebook's own gate report must match the local gate report before the score is trusted.
- **Fail → STOP:** any gate fails → keep **P14 (0.883)**, no v21 slot spent. Record the
  numbers; do not re-tune gates post-hoc.
- **No pseudo-labeling. No true-label sibling features. No test-row train-label lookup.
  No architecture change to P14 level-0. No new libraries (OSI-approved only).**

## 6. Deliverables

- `vault/r2_sibling_validate.py` — **local gate harness** (CPU, minutes): recompute
  `twin_scores`/`lgb_test_te` from `r2_train_feat.pkl`/`r2_test_feat.pkl`, build the SIB
  arm OOF, blend against cached P14 OOF, run gates 1–3. No GNN/pretrain retraining.
- `build_v21_kaggle_nb.py` (fork of `build_v14_kaggle_nb.py` + SIB arm cell inserted at
  the level-0→blend junction as a CORE substitution; blend widened to 3 arms; in-notebook
  gate report; submission path unchanged) → `PolyWin_R2_v21_sibling_arm.ipynb`.
- `tests/test_v21_kaggle_nb.py` (TDD; asserts the SIB cell is present, every non-SIB cell
  is bit-identical to v14, cells compile, smoke subset; gate harness unit tests).
- Smoke run locally (SMOKE=1), then full Kaggle GPU run, download, evaluate vs P14,
  decide per §5.

## 7. Error handling

- **NaN in `twin_scores`** (target with no out-fold rows → unscored): impute
  `TARGET_MEAN[u]` + miss flag (identical to the existing twin block).
- **Tiny-target Ridge** (12 features, ~220 rows): alpha tuned by inner OOF over the grid,
  clamped to grid bounds; Ridge penalizes regardless.
- **Collinear SIB arm** → `w_SIB ≈ 0`, blend ≈ P14 (safe by construction); `w_SIB` printed
  per target.
- **Row alignment:** assert `twin_scores` index order matches train.csv (mirror of the
  `superblend_oof.npz` alignment check in `run_v20_gate.py`).

## 8. Result (to be filled after run)

- [ ] Local gate harness: gates 1–3 pass/fail (report numbers).
- [ ] Notebook smoke run: cells compile, smoke subset matches harness.
- [ ] Kaggle full run: kernel URL, gate report, public LB, verdict.
