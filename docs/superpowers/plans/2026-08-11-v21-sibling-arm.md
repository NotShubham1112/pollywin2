# Implementation Plan — v21 Leak-Safe Ridge Sibling Arm

Date: 2026-08-11
Status: READY (plan) — derived from the approved design
`docs/superpowers/specs/2026-08-11-v21-sibling-arm-design.md`.
Design spec: read it first; this plan is the executable step sequence.

## Goal

Add a leak-safe sibling arm (SIB) as a third column in P14's per-target Ridge
blend. Base GBM trio stack + MT-GNN are bit-identical to P14. Only the blend
line changes: `Ridge(X=[GNN, GBM])` → `Ridge(X=[GNN, GBM, SIB])`, plus an
in-notebook gate report. P14 remains the fallback if gates fail.

## Success criteria (pre-registered — do NOT soften)

| Gate | Check | Threshold |
|------|-------|-----------|
| 0 (diagnostic) | per-target `sib_only_r2[t]` (SIB Ridge alone, own GroupKFold) | report only; ~0 on eps/nc/ei ⇒ strong stop signal |
| 1 (leak audit) | sibling-feature ↔ true-label exact-match count across all val folds | must be **0** |
| 2 (OOF gain) | blend mean over **{eps,nc,ei}** ≥ P14+δ AND overall mean ≥ P14+δ | soft δ=+0.0015; strong δ=+0.003 (both tiers both means) |
| 3 (worst-target) | every per-target OOF delta vs P14 | ≥ **−0.003** |

Fail ⇒ keep P14 (0.883), record numbers, do not re-tune gates.

## Pre-registered constants (mirror mt_gnn_v2.py exactly)

- `SEED = 42`, `EARLY_HOLDOUT = 0.15`, `GLOBAL_FOLDS = 2 if SMOKE else 5`
- `TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]` (sorted)
- `ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]`
- SIB features per target `t`: `twin_scores[:, u]` + miss flag for every `u ≠ t`
  (12 columns, no self column); NaN → `TARGET_MEAN[u]` + miss flag=1.
- Twin LGBM: `n_estimators=800, learning_rate=0.05, num_leaves=15,
  min_child_samples=10, subsample=0.8, colsample_bytree=0.8, random_state=SEED`.

---

## Task 1 — Local gate harness `vault/r2_sibling_validate.py` (CPU, minutes)

A self-contained script that recomputes the twin source from the feature pickles,
builds the SIB arm, blends against the cached P14 OOF, and runs gates 0–3.
No GNN / no pretrain retraining. Test-driven.

### 1a. Feature + twin-source recompute

Mirror `mt_gnn_v2.py:40-108` and `leak_safe_oof_scores()` (`mt_gnn_v2.py:285-312`):

- Load `r2_train_feat.pkl` / `r2_test_feat.pkl`.
- `FEAT_COLS = [c for c in trf.columns if c not in
  ('smiles','target','target_type','canon','inchikey','id')]`.
- `add_fingerprints` → morgan(2048), maccs(167), ap(1024), tt(1024).
- `X = hstack([D, mor, mc, ap, tt])`; `Xs = StandardScaler().fit(X).transform(X)`;
  `Xtes = scaler.transform(Xte)` (fit on TRAIN only — must match notebook).
- `Y, T, G = trf.target, trf.target_type, trf.canon`; `idx_of_target`,
  `TARGET_IDX`, `TARGET_MEAN`.
- Reimplement `leak_safe_oof_scores()` verbatim (one global GroupKFold on canon →
  `row_fold`; per target `u` train out-of-fold target-`u` LGBM, score all in-fold
  rows; bag `lgb_test_te`). Returns `twin_scores` (7409×7), `lgb_test_te` (4940×7).
- **Alignment guard** (mirror `run_v20_gate.py:75-83`):
  `assert twin_scores.shape[0] == len(trf)` and same for test.
- Smoke fast-path via env `SMOKE=1` (folds=2, `n_estimators=200`).

### 1b. SIB arm builder (pure, unit-testable)

```python
def build_sib_arm(twin_scores, lgb_test_te, TARGETS, TARGET_MEAN,
                  Y, T, G, idx_of_target, GLOBAL_FOLDS, ALPHA_GRID):
    sib_oof = np.full(len(Y), np.nan)
    sib_test = np.zeros(len(lgb_test_te))
    sib_only_r2 = {}
    for t in TARGETS:
        idx = idx_of_target[t]
        cols = [u for u in TARGETS if u != t]
        Xtr = build_feats(twin_scores[idx], cols, TARGET_MEAN)   # 12 cols
        Xte = build_feats(lgb_test_te, cols, TARGET_MEAN)
        yt = Y[idx].astype(np.float64)
        cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Xtr, yt, G[idx]))
        # alpha tuned by inner OOF; ridge refit on full OOF for test
        sib_oof[idx], sib_test, a_best = ridge_oof(Xtr, Xte, yt, cv, ALPHA_GRID)
        sib_only_r2[t] = float(r2_score(yt, sib_oof[idx]))
    return sib_oof, sib_test, sib_only_r2
```

`build_feats`: columns `[twin[:,u], miss(twin[:,u])]` for u≠t, NaN→TARGET_MEAN[u].

### 1c. Blend + gates (pure, unit-testable)

- Load `vault/pipeline_out_pretrain/superblend_oof.npz`
  (keys: `oof_gbm, oof_mt, test_gbm, test_mt, y_train, target_type_train,
  target_type_test`).
- **Row alignment** (mirror `run_v20_gate.py`): `target_type_train` == trf order,
  `target_type_test` == tef order, `corr(y_train, trf.target) > 0.999`.
- P14 2-arm reference: `_p14_2arm_oof(M2, y, g, n_splits=5)` — copy the exact
  function from `run_v20_gate.py:96-125` (fold-safe alpha scan, same grid).
- 3-arm blend: same protocol over `M3 = column_stack([gbm, mt, sib])`.
- `gate_1_leak_audit(twin_scores, trf, idx_of_target, folds)` → int count
  (v19-style: for each val fold, count rows where any twin feature exactly equals
  a true other-target label of that polymer). Must be 0.
- `gate_report(p14_r2, v21_r2, alphas, tier)` → dict
  `{gate0, gate1, gate2_soft, gate2_strong, gate3, pass}` using the table above.
- Print report (per-target R² deltas, `w_SIB` per target) and write
  `vault/pipeline_out_v21/v21_gate_report.csv`.

### 1d. Unit tests — `tests/test_v21_harness.py`

Small synthetic data (e.g. 3 targets × 40 rows, 2 folds) exercising:
- `build_feats` column count / miss-flag / NaN-impute behavior.
- `ridge_oof` returns finite OOF aligned to input order (mirror
  `test_v20_arm_cv.py:test_oof_alignment_and_shape`).
- `build_sib_arm` shapes: `sib_oof` len 7409-equivalent, `sib_test` len
  test-equivalent, `sib_only_r2` keys == TARGETS.
- `gate_1_leak_audit` returns 0 on leak-safe features, >0 when a leak is injected
  (force one twin feature == a true label).
- `gate_report` boundaries: exact +0.0015 / +0.003 / −0.003 equality passes;
  one step below fails; `alphas` all ≤ cap not required by v21 (Ridge alphas are
  unconstrained) — assert only mean/worst gates.

**Verification:** `python tests/test_v21_harness.py` and `pytest tests/test_v21_harness.py`.
Then a real-data smoke: `SMOKE=1 python vault/r2_sibling_validate.py` completes in minutes.

---

## Task 2 — Notebook builder `build_v21_kaggle_nb.py`

Fork `build_v14_kaggle_nb.py`. The CORE extraction is unchanged:

```
_A_START = _idx("# Graph featurization")
_A_END   = _idx("# Twin source:")
_B_END   = _idx("    stack_oof[t] = oof; stack_test[t] = te_pred") + 1
CORE_A = lines[_A_START:_A_END]
CORE_B = lines[_A_END:_B_END]
```

Cell list (bit-identical to v14 EXCEPT the two changes below):
1. **Keep verbatim:** intro, setup, imports, data load, CORE_A cell, pretrain
   cell, CORE_B (level-0) cell. Same `REPL` map
   (`@FOLDS@/@MAXEP@/@PATE@/@BS@/@PRTEP@/@PRTSMP@/@SUFFIX@/@GNNSEEDS@`).
2. **Insert a new SIB arm cell** at the level-0 → blend junction (after the CORE_B
   cell, before the blend cell). Inlines `build_sib_arm` from the harness
   (twin_scores/lgb_test_te are already in scope from CORE_B) →
   `sib_oof`, `sib_test`, `sib_only_r2`.
3. **Replace cell 5 (v13/P14 2-arm blend)** with the v21 3-arm blend cell:
   `M3 = column_stack([oof_gbm_global, oof_mt_global, sib_oof])` and
   `Mte3 = column_stack([test_gbm_global, test_mt_global, sib_test])`,
   same per-target alpha scan, `w_SIB` printed, gate report (gates 0–3) printed,
   submission written as `submission_v21.csv` (unchanged `id,target` writer).
4. **Drop v14 cell 6** (the leaky "conservative sibling-Ridge + physics eps"
   cell, `build_v14_kaggle_nb.py:704-889`). It re-introduces the exact full-train
   true-label pivot leak v21 exists to remove and overwrites the submission.
   v21 is its leak-safe replacement.
5. Markdown cells updated to describe v21 + gates (v14 intro/§5/§6 text replaced).

Output: `PolyWin_R2_v21_sibling_arm.ipynb` (full) and
`PolyWin_R2_v21_sibling_arm_smoke.ipynb` (SMOKE=1).

**Verification:** `python build_v21_kaggle_nb.py` then
`SMOKE=1 python build_v21_kaggle_nb.py`.

---

## Task 3 — TDD tests `tests/test_v21_kaggle_nb.py`

Modeled on `tests/test_v15_kaggle_nb.py` (build both notebooks via
`subprocess` + `_cell_list`/`_core_cells` helpers):

- `test_v21_sib_cell_present`: a code cell contains `def build_sib_arm` and
  `sib_only_r2`; the blend cell contains `column_stack([oof_gbm_global,
  oof_mt_global, sib_oof])` and `w_SIB`.
- `test_v21_core_a_core_b_bit_identical_to_v14`: extract CORE_A/CORE_B cells by
  the same markers as `test_v15_kaggle_nb.py`; assert exact string equality with
  v14's cells (build v14 fresh via `build_v14_kaggle_nb.py`).
- `test_v21_leaky_cell_absent`: v14 cell-6 markers absent
  (`train_pivot`, `get_sibs`, `submission_v17_final.csv`).
- `test_v21_gates_pre_registered`: code contains `+0.0015`, `+0.003`, `−0.003`,
  `leak` audit count, and the tier strings (soft/strong).
- `test_v21_submission_unchanged`: `submission_v21.csv`, `id,target`,
  `index=False`, `final_te` shape == len(tef).
- `test_v21_all_cells_compile`: `ast.parse` every code cell (v15 pattern).
- `test_v21_smoke_subset`: SMOKE build ⇒ `GLOBAL_FOLDS = 2`,
  `PRETRAIN_SAMPLE = 2000`, `GNN_SEEDS = "1"` (v15/v20 pattern).
- `test_v21_forbidden_refs`: no `superblend_oof.npz` / vault paths / URLs in the
  notebook source (self-contained on Kaggle, mirror `test_v20_nb.py:FORBIDDEN`).

**Verification:** `python tests/test_v21_kaggle_nb.py` and `pytest tests/`.

---

## Task 4 — Verification & smoke run

1. `pytest tests/test_v21_harness.py tests/test_v21_kaggle_nb.py` — all green.
2. `SMOKE=1 python vault/r2_sibling_validate.py` — gates 0–3 report printed;
   row-alignment asserts pass; runtime within a few minutes on CPU.
3. `python build_v21_kaggle_nb.py` + `SMOKE=1 python build_v21_kaggle_nb.py` —
   both notebooks build; smoke notebook compiles cell-by-cell.
4. Optional: execute the smoke notebook headlessly if a runner exists
   (skip otherwise — the harness + compile checks are the CI gate).
5. Report to the user:
   - `sib_only_r2` per target (gate 0 — is there signal at all?)
   - gate 1 exact-match count
   - gate 2 mean deltas at both tiers, gate 3 worst-target delta
   - verdict: P14 stays final vs v21 goes to a Kaggle run.

## Risks / notes

- `twin_scores` recompute must be a *verbatim* mirror of
  `leak_safe_oof_scores()`; any deviation (fold assignment, LGBM params, scaler
  fit-on-train) changes results. Guard with the unit tests + real-data smoke
  compared against the notebook's own numbers.
- The harness and the notebook must agree numerically; the Kaggle run's gate
  report is not trusted unless it matches the local one (design §5).
- Gate 2 checks BOTH {eps,nc,ei} mean and overall mean — implement both, not one.
- No new libraries. No pseudo-labels. No test-row train-label lookup.
