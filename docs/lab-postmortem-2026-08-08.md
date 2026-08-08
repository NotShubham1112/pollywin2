# PolyWin R2 — Lab Postmortem

Date: 2026-08-08
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)
Metric: unweighted mean R2 over 7 targets (eea, egb, egc, ei, eps, nc, tg) -- each target
counts equally regardless of sample size.
Final submission: **P14 (public LB 0.883)**. Working tree clean; P14 frozen as production.

---

## 1. What was tested

Three hypotheses were put on the line. Each was pre-registered, had a clear pass/fail
gate, and was judged on the public LB (37% test slice; final standings use the other 63%).

| # | Hypothesis | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | GBM + MT-GNN blend beats GBM alone | LB 0.852 -> **0.877** | **Accepted** |
| 2 | Larger PI1M pretraining improves the model | LB 0.877 -> **0.883** | **Accepted** |
| 3 | Extra EPS/NC loss emphasis improves EPS/NC | eps -0.0053, nc -0.0234 | **Rejected** |

## 2. Leaderboard progression (verified refs)

| ref | wave | config | public LB | delta | notes |
|-----|------|--------|-----------|-------|-------|
| 55194181 | v4 | baseline stack (lgb/cat/xgb/hgb + FFN) | 0.828 | -- | start |
| 55216423 | v6 | honest OOF stack (l15 + meta), no NNs | 0.847 | +0.019 | standing best |
| (--) | v7 | retrieval FULL | not submitted | -- | failed OOF |
| 55246041 | v8 | PI1M pseudo labelling | 0.828 | -0.019 | regression, dropped |
| (--) | v10 | pretrained GNN | 0.830 | -- | |
| 55286407 | v11 | blend (fold-safe weights) | 0.852 | +0.022 | best standing |
| 55305403 | v12 | chemistry bucket-MoE | 0.849 | -0.003 | OOF gain did not transfer |
| 55342412 | v13 | GBM + MT-GNN blend, multiseed (42/999/2025) | **0.877** | +0.025 | Hypothesis 1 |
| 55346358 | P14 | + full-PI1M pretrain (995k, 10 epochs) | **0.883** | +0.006 | Hypothesis 2, FINAL |
| v1 | v15 | EPS = NC x2 focus weight | OOF -0.0051 | -- | Hypothesis 3, rejected |

## 3. The six findings

### 3.1 Retrieval adds little once a strong GNN exists
v7 (retrieval augmentation) failed OOF and was never submitted. Once the encoder captured
molecular similarity, hand-built retrieval features were redundant.

### 3.2 GBM + GNN blending works (Hypothesis 1)
v13 combined the GBM trio with the MT-GNN via a per-target Ridge blend and gained +0.025
over v11 (0.852 -> 0.877). The gains were additive, not a fluke of one fold.

### 3.3 Full-scale pretraining works (Hypothesis 2)
P14 scaled the pretrain corpus from the 20k-molecule budget to the full deduplicated PI1M
archive (~995k molecules, 10 epochs) and gained +0.006 (0.877 -> 0.883). Bigger, better
shared representations directly transferred downstream.

### 3.4 Correlation reduction was NOT the mechanism
The v14 hypothesis (Criterion A: adding diversity via a second model) was falsified -- the
correlation between GBM and GNN OOF actually rose (0.9513 -> 0.9552), yet the blend still
got better. The gain came from a stronger encoder, not from decorrelation.

### 3.5 EPS/NC carried most of the gains
P14's improvement was almost entirely eps (0.7749 -> 0.8009, +0.0260) and nc
(0.8417 -> 0.8657, +0.0240). The two weakest targets were the highest-leverage rows.

### 3.6 Loss reweighting for EPS/NC failed (Hypothesis 3)
v15 doubled the sample weight for eps/nc rows (TGT_FOCUS = {"eps": 2.0, "nc": 2.0}). Both targets got WORSE:

```
EPS: 0.8009 -> 0.7956  (-0.0053)
NC : 0.8657 -> 0.8423  (-0.0234)
mean: 0.8768 -> 0.8718 (-0.0051)   <-- pre-registered gate FAIL
```

Doubling the weight made the fine-tune overfit the tiny eps/nc folds (only ~229 rows
each) while stealing gradient from the dominant targets. The P14 gain came from stronger
shared features, not from loss reweighting. v15 was falsified, stopped, and P14 was kept.

## 4. What it cost and what it bought

- Total experiments: v4 baseline through v15; three hypotheses, two accepted, one rejected
  cleanly.
- Public LB floor improved 0.828 -> 0.883 and ended in the top-20 (0.886) / top-10 (0.898)
  gap: roughly 0.003 from top-20, 0.015 from top-10.
- Each experiment was a single change, pre-registered, with a stopping gate. No blind
  tuning after signal died.

## 5. Recommendation for the future

- Submit P14 (0.883) as the final answer wherever allowed; keep it frozen as production.
- Do not overwrite P14; it is the best confirmed model.
- Only a quantitative step-change (a real representation-size increase, or a principled
  regularization method) justifies further tuning against the closing LB.