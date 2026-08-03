# PolyWin R2 — v7 Retrieval-Augmented GBM Stack (Design)

Date: 2026-08-03
Status: Approved (awaiting spec review)
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v7 exists

v5/v6 delivered a clean, reproducible pipeline and an honest OOF benchmark. The EFN/TgNN
hypothesis was falsified: in honest GroupKFold OOF the neural models lose to GBMs on every
target (e.g. EFN egc 0.716 vs XGB 0.514; TgNN tg 47.39 vs LGB 35.12). Public LB stands at
**0.847** (v6, submitted `55216423`).

The largest identified untapped signal is **retrieval**: 839/4940 (~17%) of test polymers are
canonical twins of train polymers, and the overlap analysis shows the electronic targets share
substantial polymer overlap while Tg is largely isolated. A polymer repeatedly measured under
different target types means its nearest neighbor for an *eea* prediction may only have been
labeled *tg/egc/egb* — the current per-type retrieval (Layer 3) is blind to that.

v7 answers one question with a clean three-way experiment:

> Does kNN retrieval over polymers add predictive signal beyond the existing chemistry
> descriptors — and if so, how much of that signal lives inside retrieval alone?

## 2. Baseline & success criteria

- Reference baseline is the **v6 feature set with NNs off** (RUN_NNS=False changes the stack
  itself, so the v6 *published* 0.847 — which included NN OOFs — is not the baseline).
- **Locked success criteria (pre-registered, decided before coding):**

  | Outcome | Condition |
  | ------- | --------- |
  | Strong success | FULL beats BASE on ≥4 targets **and** public LB gain ≥ +0.005 |
  | Moderate success | FULL beats BASE on 2–3 targets **and** public LB gain 0.002–0.005 |
  | Failure | FULL ≈ BASE, RETR-only weak, no LB movement |

  On failure, v8 = pseudo-labeling immediately — no further retrieval expansion.
- Success is evidence-driven, not LB-only:

  | Case | BASE | FULL | RETR-only | Verdict |
  | ---- | ---: | ---: | --------: | ------- |
  | 1 | 0.50 | 0.46 | 0.70 | retrieval weak alone, complementary → keep |
  | 2 | 0.50 | 0.49 | 0.52 | retrieval duplicates existing signal → low value |
  | 3 | 0.50 | 0.40 | 0.43 | retrieval carries huge signal → dominant workstream |

- Primary diagnostic: the per-target ablation table (BASE / FULL / RETR-only OOF RMSE).
- Secondary diagnostics: (a) LGB `gain` importance split into retrieval vs non-retrieval share
  per target; (b) train rows split by `neighbor_density` into dense vs sparse neighborhoods,
  comparing FULL-vs-BASE gain per group (expect gain to correlate with density_90).
- Submit only the FULL config (all 4 GBMs + L1.5 Ridge + L2 meta, stacking identical to v6).
  Compare LB vs 0.847.

## 3. Architecture (v7)

### 3.1 Runtime mode flag

```python
RUN_NNS = False   # default; EFN/TgNN code archived in notebook, not executed
```

- v7 runs **pure GBM**: base models = lgb, cat, xgb, hgb. Stacking stays **exactly as v6**:
  per-target L1.5 Ridge over own base OOFs, then L2 meta (reliability features +
  cross-target L1.5 OOF features, electronic cluster only). The v6 submission (0.847) came
  from the L2 `final_meta`, so dropping L2 would regress the baseline — L2 is retained for
  baseline parity, with EFN/TgNN OOFs simply removed from `BASE_MODELS`/`store_key`.
- NN code stays in the repo (archived cell); flipping `RUN_NNS=True` re-enables it for a later
  final-ensemble benchmark without resurrecting old code.

### 3.2 Neighbor pools (all Morgan r2/512, Tanimoto, fold-safe)

Fold-safety is free: candidates are always `fold != f`; because `folds` is one global
GroupKFold(10) keyed on canonical SMILES (build_pipeline_nb.py:151-168), all rows of the same
polymer across every target type land in the same fold and are excluded together.

- **Pool A — Global chemistry.** Candidates = all train rows (all target types). Answers
  "where am I in polymer space?".
- **Pool B — Same-target.** Candidates = train rows of the target's own type. Replaces Layer 3's
  6-column block with a richer set.
- **Pool C — Cross-target property priors.** Neighbor set = Pool A's; labels come from all 7
  targets, each using only its available rows. This is the primary new signal: a neighbor
  labeled only *tg/egc/egb* still contributes priors for an *eea* prediction.

### 3.3 Feature grid (Morgan r2/512, 57 new columns: 12 Pool A + 10 Pool B + 35 Pool C)

**Pool A (global chemistry):**

| Column | Definition |
| ------ | ---------- |
| `g_top1_sim` | NN Tanimoto |
| `g_top3_sim`, `g_top5_sim`, `g_top10_sim` | top-k similarity |
| `g_top5_mean`, `g_top10_mean` | mean of top-k similarities |
| `g_gap` | top1 − top2 similarity |
| `g_std` | std of top-5 similarities |
| `g_density_95/90/85` | fraction of pool with sim > 0.95/0.90/0.85 |
| `g_exact_twin` | `top1_sim >= 0.999` (binary) |

**Pool B (same-target):**

| Column | Definition |
| ------ | ---------- |
| `st_top1_sim`, `st_top3_sim`, `st_top5_sim` | top-k similarity within target |
| `st_density_95/90/85` | density within target pool |
| `st_tgt_mean/median/std/wmean_sq` | this-target neighbor target prior; `wmean_sq` weights = `max(sim,0)^2` |

**Pool C (cross-target priors, ×7 targets):** for each `t in {tg,egc,egb,eps,nc,ei,eea}`:

| Column | Definition |
| ------ | ---------- |
| `ct_<t>_mean` | mean target value of neighbors with a `t` label |
| `ct_<t>_median` | median |
| `ct_<t>_std` | std |
| `ct_<t>_wmean_sq` | sim²-weighted mean |
| `ct_<t>_count` | number of neighbors contributing a `t` label |

A prior built from 10 neighbors is very different from one built from 1 — the count column
lets the model know. Total Pool C = 35 columns (5 × 7).

### 3.4 Weighting (approved modification)

Weighted priors use **squared similarity**: `weights = max(sim, 0)^2`. This makes the nearest
neighbor dominate (0.95→0.90, 0.90→0.81, 0.70→0.49, 0.50→0.25) and reflects retrieval
intuition better than raw Tanimoto. `wmean_sq` is the primary weighted prior.

### 3.5 Compute strategy

- Compute the **global jaccard matrix once**, then fold-mask:

  | Block | Pairs | Float32 |
  | ----- | ----: | -----: |
  | train–train | 7406² ≈ 54.8M | ≈ 220 MB |
  | test–train | 4940 × 7406 ≈ 36.6M | ≈ 146 MB |

  ≈ 364 MB total — comfortably in memory. "Compute once, mask many times" replaces the
  per-fold cdist re-computation in the current Layer 3.
- Runtime projection: retrieval adds ≈ 10–18 min on CPU, partially offset by dropping EFN/TgNN.
- k grid locked at K = {1, 3, 5, 10}. No top20/top50: retrieval hypothesis is about twins and
  very local neighborhoods, not broad manifold averaging.

### 3.6 Experiment wiring

- Three arms per target, **identical folds, identical GBM hyperparameters**:
  - BASE = v6 feature columns (no retrieval)
  - FULL = BASE + retrieval (submitted config)
  - RETR-only = retrieval columns alone
- Central table uses **LGB** for all three arms (cheapest, strongest per benchmark). The
  submitted stack uses all 4 GBMs + L1.5 Ridge + L2 meta on the FULL feature set (stacking
  identical to v6 per §3.1).
- Layer 3 rewritten: compute Pool A/B/C for the FULL column set; BASE/RETR arms are feature
  subsets of the same model fit (no extra retrieval compute).

### 3.7 Artifact persistence

- Persist `Xtr_retr.parquet` / `Xte_retr.parquet` (retrieval columns only) and the FULL Xtr/Xte
  for post-hoc attribution.
- Persist `ablation_lgb.csv` (per-target BASE/FULL/RETR-only OOF RMSE + `gain` importance
  splits) and a density-split diagnostic table.
- Persist `retrieval_audit.csv` — per row (both train and test): `id`, `target_type`,
  `top1_sim`, `top3_sim`, `top5_sim`, `g_exact_twin`, `g_density_95/90/85`. Enables
  post-LB attribution ("did retrieval help on twins? dense? sparse?") without rerunning
  anything. Tiny storage, high debugging value.
- All existing v6 artifacts (folds.csv, oof_*.parquet, l15_ridge.parquet, final_meta.parquet,
  submission.csv) regenerated as before.

### 3.8 Figures

Existing notebook figures 01–09 remain (target balance, histograms, chemistry drivers, model
comparison, pred-vs-actual, residuals, feature importance, cross-target corr, stack
improvement). v7 adds a retrieval-diagnostics block (10–14), the ablation centerpiece
(15–16), and stack/importance/LB visuals (17–20):

| Fig | File | Content | Purpose |
| --: | ---- | ------- | ------- |
| 10 | `10_similarity_dist.png` | hist of `g_top1_sim` | twins / near-twins regime (Phase 2 #7) |
| 11 | `11_neighbor_density.png` | hist of `g_density_90` | where are dense neighborhoods? (#8) |
| 12 | `12_exact_twin_freq.png` | train vs test bars of `sim >= 0.999` | quantify twin regime (#9) |
| 13 | `13_retrieval_gain_vs_density.png` | FULL−BASE OOF gain by density bucket | **key diagnostic**: expect gain↑ with density_90 (#10) |
| 14 | `14_sim_vs_oof_error.png` | `g_top1_sim` vs |OOF error| scatter | do close neighbors predict better? (#11) |
| 15 | `15_ablation_base_full_retr.png` | grouped bars per target: BASE / FULL / RETR-only (LGB) | **centerpiece**: does retrieval help? (#13) |
| 16 | `16_retrieval_delta.png` | FULL−BASE bar per target | sign/scale of retrieval contribution (#14) |
| 17 | `17_oof_pred_corr.png` | corr heatmap of lgb/cat/xgb/hgb OOF preds | model diversity → stack benefit (#15) |
| 18 | `18_retrieval_feat_importance.png` | top retrieval cols by LGB `gain` (FULL) | which retrieval cols carry signal (#19) |
| 19 | `19_pool_contribution.png` | stacked bar of Pool A/B/C gain share | global vs same-target vs cross-target (#20) |
| 20 | `20_lb_progression.png` | LB score vs submission # (v4/v5/v6/v7) | leaderboard impact timeline (#22) |

Gaps vs the 24-figure plan, resolved:
- **Overlap heatmap (#3)** — not currently produced anywhere. Add it to the Fig-08 cell as a
  second panel: shared-polymer count matrix across the 7 target pairs (canon-keyed), which
  supports the Pool-C hypothesis directly.
- **Test/train distribution projection (#5, PCA/UMAP)** — deferred to v8; adds a projection
  hyperparameter axis and is not required to test retrieval.
- **Reliability-feature importance (#18)** — folded into existing Fig-07 (L2 meta importance
  already shown); no new figure.

Judge-ready 10 (final notebook, end): 01, 08 (+ overlap panel), 10, 13, 15, 04, 17, 18, 20,
and the ablation table.

## 4. Roadmap (frozen after v7)

| Version | Workstream |
| ------- | ---------- |
| v7 | Retrieval (this design) |
| v8 | Pseudo-labeling (PI1M) |
| v9 | AtomPair on Pool C only |
| v10 | Final ensemble tuning |

No new neural architectures unless retrieval **and** pseudo-labeling are exhausted.

## 5. Deferred to v8+ (explicitly out of scope for v7)

- AtomPair on Pool C only (add only if v7 shows signal).
- TopologicalTorsion retrieval.
- Approximate nearest neighbors / embedding retrieval.
- Pseudo-labeling with PI1M (Phase 5 — only after retrieval is benchmarked).

## 6. Validation plan

1. Local smoke run (reduced workload, GLOBAL_FOLDS=5) — bug-check only: retrieval columns
   compute, ablation arms wire up, submission format unchanged (id,target, all 4940 test ids).
2. Tests: extend `tests/test_pipeline_nb.py` with markers for the new Layer 3 (pools A/B/C
   column counts, fold-safety of cross-target priors, wmean_sq, exact-twin, count columns).
3. One Kaggle run (RUN_NNS=False). Report the ablation table + importance + density split,
   then submit FULL and compare LB vs 0.847.
