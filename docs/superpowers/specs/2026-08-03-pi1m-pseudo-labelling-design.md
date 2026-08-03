# PolyWin R2 — v8 PI1M Pseudo-Labelling (Design)

Date: 2026-08-04
Status: Approved in brainstorm (awaiting spec review)
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v8 exists

The v7 retrieval experiment was evaluated against its pre-registered criteria and came out a
**FAILURE** (see `2026-08-03-retrieval-augmented-gbm-design.md` §2): the FULL arm (BASE +
retrieval columns) is worse than BASE on **all 7 targets** in honest GroupKFold OOF
(tg +0.79, egc +0.038, egb +0.066, eps +0.034, nc +0.0014, ei +0.008, eea +0.022);
RETR-only is uniformly weak; and `retrieval_gain_share` shows FULL leans hard on retrieval
columns while OOF degrades — the neighbour signal overfits fold-to-fold. The v7 spec locked
the failure branch: **"v8 = pseudo-labeling immediately — no further retrieval expansion."**
v7 was not submitted; the standing public LB remains **0.847** (v6, `55216423`).

Pseudo-labelling is a different mechanism from retrieval: instead of adding neighbour-derived
*features*, it adds *training rows* (unlabeled PI1M polymers labelled by our own confident
predictions). The five small targets (egb 337, eps 229, nc 229, ei 222, eea 221 rows) are
data-starved, and the data-intelligence report (Phase 6) ranks high-confidence
pseudo-labelling as a cheap, safe +3–8% opportunity. PI1M (~996k polymers) is **explicitly
allowed** by the rules ("may be used for implementing advanced algorithms").

v8 answers one question with a clean two-arm experiment:

> Do confidence-filtered PI1M pseudo-rows add predictive signal beyond the v6 feature set —
> and if so, does the full stack trained on pseudo-augmented data beat the standing 0.847 LB?

## 2. Baseline & success criteria

- Reference baseline is the **BASE arm = v6 feature columns with NNs off and NO retrieval
  columns** (RUN_NNS=False; retrieval cols excluded on the evidence of v7). The v6 published
  0.847 LB remains the submission benchmark.
- **Locked success criteria (pre-registered, decided before coding):**

  | Outcome | Condition |
  | ------- | --------- |
  | Strong success | PSEUDO beats BASE on ≥4 targets **and** public LB gain ≥ +0.005 (≥ 0.852) |
  | Moderate success | PSEUDO beats BASE on 2–3 targets **and** public LB gain 0.002–0.005 (0.849–0.852) |
  | Failure | PSEUDO beats BASE on ≤1 target, or no LB movement → next roadmap step (v9) |

- Primary diagnostic: the per-target ablation table (BASE / PSEUDO OOF RMSE, LGB central
  table, identical folds).
- Secondary diagnostics: (a) selected-vs-rejected pseudo-row confidence distribution
  (fig 22); (b) per-target pseudo-row count vs OOF delta (did the small five benefit most?).
- Submit only on Strong/Moderate. Submission = the full stack (all 4 GBMs + L1.5 Ridge + L2
  meta, stacking identical to v6) trained on pseudo-augmented data. Compare LB vs 0.847.

## 3. Architecture (v8)

### 3.1 Runtime mode flag

```python
USE_PSEUDO = True    # v8: pseudo-labelling on
PSEUDO_SAMPLE = 200000   # seeded random PI1M sample scored for selection
PSEUDO_FRAC = 0.05       # per-target top-5% lowest-disagreement rows kept
PSEUDO_CAP_MULT = 2.0    # per-target cap = 2x real training count
```

Layer 8 exists in the notebook behind `USE_PSEUDO=False` (placeholder since v5).
v8 replaces the placeholder body with the real machinery below; `USE_PSEUDO=False`
still skips everything (CPU fallback for smoke runs).

### 3.2 Selection (confidence-filtered, cross-model disagreement)

1. **Sample:** seeded random 200k rows of PI1M (replaces the existing biased
   `nrows=400000` head-slice). Seed 42, shared with the rest of the pipeline.
2. **Canonicalize** via the existing `canon_key`; drop non-canonical rows.
3. **Features:** `build_features` on the sample (same BASE columns as the real arms);
   reindex to `Xtr.columns`, fill 0, clip to train min/max (existing Layer-8 behaviour).
4. **Score:** for each target `tt`, predict the sample with the **chosen expert** (best of
   {lgb, cat, xgb} by the existing `LEADERBOARD`) **plus the other two GBM families** —
   one fit per family (seed 42), std taken **across families** (inter-model disagreement),
   not across seeds. Model diversity is a stronger uncertainty signal than seed noise.
5. **Select:** keep the per-target top-5% lowest inter-model std rows (existing
   percentile logic, now over cross-family std). `conf = std`.

### 3.3 Integration (capped, full feature rebuild, expert retrain)

- Per-target cap `cap_tt = round(2.0 * real_count_tt)`; select only up to `cap_tt` rows
  from the target's sorted pool (tg ≈ 8.3k, egc ≈ 4.1k, small five ≈ 0.4–0.7k; total
  ≈ 15k pseudo rows). Real rows are never drowned (≤ 2:1 pseudo:real).
- **Full RDKit feature rebuild** for the selected SMILES via `build_features`
  (replaces the Layer-8 "placeholder" retrain stub).
- **Retrain:** each target's chosen expert is refit on (real + pseudo) rows with the
  real rows' `target` values and pseudo rows' `target` = mean prediction, `target_type`
  = `tt`. Pseudo rows have no fold membership → appended to **every fold's** training
  side (no leakage: they are unlabeled data, not test rows). OOF is scored on real rows
  only, with the same fold partition as BASE.
- The LGB central-table arm uses the same augmented rows (LGB refit per target).

### 3.4 Stacking & submission path

- On success, the full v6-identical stack (lgb/cat/xgb/hgb per fold on augmented rows,
  L1.5 Ridge, L2 meta) is rebuilt and the submission written. The experiment table
  (BASE vs PSEUDO, LGB) is computed first; the stack build reuses the same augmented
  row sets.

### 3.5 Compute strategy & runtime

- Dominant added costs vs the v7 run (~52 min): PI1M read + canon (~2 min),
  200k-row fingerprint build (~6–8 min), 3-family scoring of the sample
  (~3–4 min, 3 × 200k × ~340 cols), selected-row rebuild + retrain (~4–5 min).
  Projected total ≈ **65–75 min** (well under the 9h kernel limit; smoke keeps
  `USE_PSEUDO=False`).
- Hard caps: sample 200k, cap multiplier 2.0, frac 0.05 — no unbounded loops.

### 3.6 Artifact persistence

- `pseudo_labels.csv` — selected rows: `smiles`, per-target `target` (mean), `conf`
  (inter-family std), `target_type`.
- `ablation_pseudo.csv` — per-target BASE / PSEUDO OOF RMSE + delta.
- All existing artifacts (folds.csv, oof_*.parquet, l15_ridge.parquet, final_meta.parquet,
  submission.csv) regenerated as before.

### 3.7 Figures

Existing notebook figures 01–20 remain (v7's retrieval figures 10–19 stay as
evidence-of-failure; fig 20 `_lb` already annotates v7 as failed/not-submitted). v8 adds:

| Fig | File | Content | Purpose |
| --: | ---- | ------- | ------- |
| 21 | `21_pseudo_ablation.png` | grouped bars per target: BASE / PSEUDO (LGB) | **centerpiece**: does pseudo-labelling help? |
| 22 | `22_pseudo_conf_dist.png` | selected-vs-rejected inter-model std hist | is the filter separating signal from noise? |
| 23 | `23_pseudo_rows_per_target.png` | pseudo row count vs real count per target | integration balance |

Judge-ready 10 (final notebook, end): 01, 08 (+ overlap panel), 15 (v7 ablation kept as
evidence), 21, 22, 04, 17, 18, 20, and the ablation table.

## 4. Roadmap (frozen after v8)

| Version | Workstream |
| ------- | ---------- |
| v7 | Retrieval (FAILED, evidence recorded) |
| v8 | Pseudo-labeling (this design) |
| v9 | AtomPair on Pool C only (v7 spec roadmap; only if v8 fails) |
| v10 | Final ensemble tuning |

No new neural architectures unless retrieval **and** pseudo-labeling are exhausted.

## 5. Deferred to v9+ (explicitly out of scope for v8)

- kNN / label-propagation from PI1M → train neighbours (data report's "Do first" item is
  neighbour-based and shares the v7 failure mode; revisit only if v8 shows PI1M signal).
- AtomPair on Pool C only.
- TopologicalTorsion / ANN / embedding retrieval.
- Contrastive or masked-SMILES pretraining on PI1M (GPU available but 996k × tokenisation
  cost is not justified until pseudo-labelling is benchmarked).

## 6. Validation plan

1. Local smoke run (reduced workload, GLOBAL_FOLDS=5, `USE_PSEUDO=False`) — pipeline
   regression only, as before.
2. Tests: extend `tests/test_pipeline_nb.py` — Layer 8 wiring (USE_PSEUDO gate, sample
   size, per-target cap formula, cross-family std computation, rebuild-not-placeholder
   assertion), plus the existing 24 tests still pass.
3. One Kaggle run (`USE_PSEUDO=True`). Report the ablation table + conf-dist figure,
   then submit on Strong/Moderate and compare LB vs 0.847.
