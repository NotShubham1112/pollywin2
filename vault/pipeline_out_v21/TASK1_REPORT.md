# Task 1 — v21 leak-safe gate harness

**Deliverable:** `vault/r2_sibling_validate.py` (self-contained, CPU-only; no GNN / no pretrain retraining). Report + data: `vault/pipeline_out_v21/`.

## What it does

1. Recomputes the twin source (`twin_scores` / `lgb_test_te`) from `r2_train_feat.pkl` / `r2_test_feat.pkl` with a verbatim mirror of `mt_gnn_v2.leak_safe_oof_scores()` (GroupKFold on canonical SMILES + early-holdout split; feature load mirrors `mt_gnn_v2.py:40-108`).
2. Builds the **SIB arm**: per-target Ridge over the *other-target* twin LGBM features plus miss flags (12 columns, no self column). `w_SIB` is read out of the final blend.
3. Blends the three arms (frozen P14 `oof_gbm`, `oof_mt` + `sib_oof`) per target with the P14 fold-safe alpha sweep (GroupKFold on SMILES, alpha tuned by inner OOF r²).
4. Runs **gates 0–3** against the frozen P14 baseline and writes the report CSV.

## Pre-registered gate table (unchanged, from design doc)

| Gate | Rule | Threshold |
|------|------|-----------|
| 0 | per-target `sib_only_r2` (diagnostic) | report only |
| 1 | exact-match leak audit (sibling feature == true other-target label, same canon group) | count = 0 |
| 2 soft | mean Δ over {eps,nc,ei} **and** overall mean ≥ P14 + δ | δ = +0.0015 |
| 2 strong | same, stronger | δ = +0.003 |
| 3 | worst per-target Δ (no target regresses) | Δ ≥ −0.003 |
| — | recomputed P14 mean within tolerance of frozen reference | |mean−0.8641| ≤ 0.005 |

P14 mean reproduced at 0.8643 (reference 0.8641) — assert passes.

## Verification (all green)

- `python tests/test_v21_harness.py` — **17 passed**, no warnings (synthetic 3-target data, ~4 s).
- `python -m pytest tests/test_v21_harness.py -q` — **17 passed**, pristine output.
- `SMOKE=1 python vault/r2_sibling_validate.py` — **completes in 26 s wall** (2 folds, 200 trees). Output below.

## SMOKE run (2 folds, 200 trees)

```
target   r2_p14   r2_v21   delta    alpha   w_sib  sib_only_r2
eea      0.9065   0.9053   -0.0012   2.50   +0.063   +0.6902
egb      0.9287   0.9297   +0.0011   5.00   +0.179   +0.8706
egc      0.9070   0.9075   +0.0005   10.00  -0.070   +0.7120
ei       0.8114   0.8132   +0.0018   10.00  +0.197   +0.6727
eps      0.7561   0.7573   +0.0012   10.00  +0.173   +0.6145
nc       0.8388   0.8376   -0.0012   0.50   +0.201   +0.7341
tg       0.9017   0.9018   +0.0001   25.00  -0.036   +0.4846
mean_v21 0.8646  mean_p14 0.8643  mean_delta +0.0003
eps/nc/ei mean delta +0.0006  worst_delta -0.0012

gate1 leak audit: 0 (must be 0)
gate2 soft  (('eps','nc','ei') + 0.0015 / overall): False
gate2 strong (('eps','nc','ei') + 0.003 / overall): False
gate3 worst-target >= -0.003 (delta): True
GATE: FAIL -> P14 stays final
```

**Interpretation:** in SMOKE the twin source is deliberately weak (2 folds, 200 trees), so per-target deltas are ≈ 0 and gates 2/overall fail — the gate correctly keeps P14 final. This is the expected outcome for a weak source and is what the full run guards against.

**Note on gate 0:** `sib_only_r2` is *not* ≈ 0 in smoke (eps 0.61, nc 0.73, ei 0.67) — the sibling arm carries real cross-target signal, and the leak audit (gate 1) is clean (0 exact matches). The authoritative gate-0 diagnostic will be the FULL run (5 folds, 800 trees).
