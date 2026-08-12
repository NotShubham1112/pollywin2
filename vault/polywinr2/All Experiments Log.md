# 🧪 PolyWin R2 — Complete Experiment Log

> **Master log of every experiment run on AISEHack 2.0 Round 2 (`ppp-round-2`, Kaggle),**
> with public-LB scores, OOF numbers, and a verdict + explanation per entry.
> Last updated: 2026-08-11.

## Competition context (the rules every run obeys)

- **Metric:** unweighted mean R² over the 7 targets — `eea, egb, egc, ei, eps, nc, tg`.
  Each target counts equally regardless of sample size.
- **Data:** `train.csv` (7,409 rows), `test.csv` (4,940 rows), `PI1M.csv` (~995k
  unlabeled polymers — allowed only as a **pretraining / feature** source, never as labels).
- **Protocol:** 5-fold **GroupKFold on canonical SMILES** (all 7 target rows of one
  polymer land in the same fold → no sibling leakage) + **per-target Ridge blend** over
  `ALPHA_GRID`. Every experiment = one pre-registered change with a stopping gate.
- **Standing / FINAL submission: P14 = 0.883 public LB.**

---

## 📊 Leaderboard progression (all submitted waves)

| Wave | Config | Public LB | Δ vs parent | Verdict |
|------|--------|-----------|-------------|---------|
| [[v4 Baseline Stack]] | GBM trio (LGBM/Cat/XGB/HGB) + FFN, RDKit feats | **0.828** | — | start (sub `55194181`) |
| [[v6 Honest OOF Stack]] | leak-safe OOF meta-stack, no NNs | **0.847** | +0.019 | accepted → superseded (sub `55216423`) |
| [[v7 Retrieval Augmented GBM]] | retrieval `FULL` features | *not submitted* | — | FAIL (OOF, never shipped) |
| [[v8 PI1M Pseudo Labelling]] | PI1M pseudo-labels from OOF | **0.828** | −0.019 | FAIL → dropped (sub `55246041`) |
| [[v10 Pretrained GNN]] | self-supervised MT-GNN (GINE) | **0.830** | — | superseded (sub `55246047`) |
| [[v11 Fold-Safe Blend]] | fold-safe per-target blend weights | **0.852** | +0.022 | best standing (sub `55286407`) |
| [[v12 Chemistry Bucket-MoE]] | KMeans chemistry routing + per-cluster GBMs | **0.849** | −0.003 | FAIL — OOF gain didn't transfer (sub `55305403`) |
| [[v13 GBM + MT-GNN Blend]] | 4-arm per-target Ridge, 5-seed MT-GNN (42/999/2025) | **0.877** | +0.025 | ✅ Hypothesis 1 (sub `55342412`) |
| **[[P14 Full-PI1M Pretrain]]** | + full 995k PI1M pretrain (10 ep), ridge over [GBM, MT] | **0.883** | +0.006 | ✅ Hypothesis 2 — **FINAL** (sub `55346358`) |
| [[v15 EPS/NC Focus]] | EPS/NC ×2 loss weight | OOF −0.0051 | — | ✘ Hypothesis 3, never submitted |
| [[v16 Cross-Target Decoder]] | physics + learned decoder arms | **0.874** | −0.009 | FAIL — gate passed, LB regressed (sub `55289004`) |
| v17 Sib-Ridge + physics eps | sibling-target Ridge arm | **0.862** | −0.021 | FAIL — dropped |
| v18 Hard-gate sib(halved)+phys | half-weight sibling gate | **0.864** | −0.019 | FAIL — dropped |
| v19 CT-PGCN | same-engine CT-PGCN, sib+phys blend (alphas ≤ 0.30) | **0.874** | −0.009 | FAIL — OOF leakage artifact, not shipped |
| v20 Self-Trained SMILES Encoder | RoBERTa-style masked SMILES encoder arm | *not submitted* | — | FAIL — local gate (OOF −0.0004) |

**Best public LB: 0.883 (P14).** Final standings use the private 63% slice; the
competition's top-20 boundary was ≈0.886 and top-10 ≈0.898.

---

## 1. v4 — Baseline Stack (the floor)

- **Model:** GBM trio (LightGBM / CatBoost / XGBoost / HistGradientBoosting) + a small FFN.
- **Features:** RDKit descriptors + Morgan/MACCS fingerprints (22 + 2,215 cols).
- **Public LB: 0.828** (submission `55194181`).
- **Explanation:** the reference floor for the whole campaign. Had a leak: OOF packing was
  not GroupKFold-safe, so scores looked better than they were. Not a contender, just the
  start line.
- **Per-target (later honest v6-style stack, see `final_leaderboard.csv`):** tg 0.4266 /
  egc 0.6365 / egb 0.7878 / eps 0.6149 / nc 0.1215 / ei 0.5332 / eea 0.4968 (LGB column).

---

## 2. v6 — Honest OOF Stack (first real gain)

- **Model:** level-1 GBM trio → level-2 meta-regressor, all OOF-packed via GroupKFold on
  canonical SMILES. No neural nets.
- **Public LB: 0.847**, Δ **+0.019** (submission `55216423`).
- **Explanation:** leak-safe OOF packing alone bought +0.019. This is the template every
  later blend reuses (fold-safe weights from v11 onward). **Lesson:** honest validation
  converts directly into leaderboard gain.

---

## 3. v7 — Retrieval Augmented GBM (dead end)

- **Idea:** hand-built molecular-similarity retrieval features from train neighbors on top
  of the GBM stack (`FULL` config).
- **Result:** FAILED honest OOF — retrieval was worse than BASE on **all 7 targets**
  (`exp2_retrieval_feats.csv`), e.g. eea 0.869→0.851, nc 0.775→0.692, eps 0.719→0.714.
- **Never submitted.** Verdict **FAIL**.
- **Explanation (Finding 1):** once a strong encoder exists, hand-built retrieval adds
  little — the GNN already captures molecular similarity implicitly. Confirmed later by
  the superblend ablation (`exp4_superblend.csv`): GBM+GNN+Retr mean 0.8621 ≈ GBM+GNN
  mean 0.8632 — retrieval rows don't help on top of the two model families.

---

## 4. v8 — PI1M Pseudo Labelling (regression, dropped)

- **Idea:** train on the large unlabeled PI1M corpus using OOF pseudo-labels as extra
  training targets.
- **Public LB: 0.828**, Δ **−0.019** vs v6 (submission `55246041`) — regression.
- **Explanation:** clean OOF +0.017 translated to public-LB −0.019. Pseudo-labels
  **amplified the model's existing errors** — the label noise is correlated with what the
  base model already gets wrong. **Lesson:** PI1M is reused later only as a
  **pretraining/feature** source (P14), never as labels.

---

## 5. v10 — Pretrained GNN (first deep branch)

- **Model:** multi-task graph neural net (GINE trunk, masked atom/bond reconstruction on
  PI1M) → per-target heads, trained in-notebook.
- **Public LB: 0.830** — *weaker alone* than the v6 stack (0.847).
- **Explanation (Finding 4):** the GNN alone underperformed the tabular stack; its value
  only appears when **blended** with the GBM trio (v11 → v13). Single-arm comparisons can
  mislead; the payoff of a GNN here is complementarity, not standalone accuracy.
- **A/B evidence (`pretrain_ab.csv`):** pretrained init beat scratch on most targets in
  OOF + trust-check.

---

## 6. v11 — Fold-Safe Blend (regime shift)

- **Model:** GBM-trio OOF + MT-GNN OOF combined with **fold-safe** per-target weights
  (weights tuned on OOF inside the same GroupKFold that produced them).
- **Public LB: 0.852**, Δ **+0.022** over v6 (submission `55286407`).
- **Explanation:** first real dual-model blend; Hypothesis 1 (GBM + GNN > GBM alone)
  started paying here. v11 OOF per-target (v12 smoke table): tg 0.9020 / egc 0.9081 /
  egb 0.9163 / eps 0.7778 / nc 0.8468 / ei 0.7939 / eea 0.8929.

---

## 7. v12 — Chemistry Bucket-MoE (OOF ≠ LB)

- **Model:** split the GBM training set by chemistry (KMeans) buckets, route rows through
  bucket-specialist GBM arms + a router; fold-safe per-cluster weights.
- **Public LB: 0.849**, Δ **−0.003** vs v11 (submission `55305403`) — dropped.
- **OOF:** bucket-MoE 0.8626 vs v11 blend 0.8625 (win on tg/egc/eea, lose 4/7, worst
  −0.0016) — marginal gain that **did not transfer**.
- **Explanation (same failure shape as v7):** subset-logic gating overfits the OOF
  partitioning; the field does not need per-bucket experts once GNN arms exist. Routing
  idea shelved.

---

## 8. v13 — GBM + MT-GNN Blend ✅ Hypothesis 1

- **Model:** GBM trio OOF + **5-seed MT-GNN OOF** (seeds 42/999/2025), per-target Ridge
  over `ALPHA_GRID`.
- **Public LB: 0.877**, Δ **+0.025** vs v11 (submission `55342412`) — first line past 0.87.
- **OOF per-target (`kernel-v13-multiseed-full/out/v13_blend_report.csv`):**
  eea 0.9151 / egb 0.9224 / egc 0.9033 / ei 0.8248 / eps 0.7749 / nc 0.8417 / tg 0.9003.
  Learned weights lean GNN on egb (0.83), eps (0.60), nc (0.68), eea (0.70); lean GBM on tg (0.81).
- **Explanation (Finding 2 + 4):** correlation between arms actually **rose**
  (0.9513 → 0.9552) yet the blend still gained — so the mechanism was NOT decorrelation;
  it was adding a genuinely complementary model family with a stronger encoder.

---

## 9. P14 — Full-PI1M Pretrain (THE FINAL) ✅ Hypothesis 2

- **Model:** v13 blend + MT-GNN encoder pretrained on the **full deduplicated PI1M
  archive (~995k molecules, 10 epochs)** before fine-tuning. Per-target Ridge over
  [GBM, MT].
- **Public LB: 0.883**, Δ **+0.006** vs v13 (submission `55346358`) — **FINAL, locked.**
- **Honest equal-weight OOF:** 0.8641 mean R².
- **OOF per-target (`kernel-v14-p1m/out/v14_blend_report.csv`):**
  eea 0.9133 / egb 0.9250 / egc 0.9070 / ei 0.8239 / eps 0.8009 / nc 0.8657 / tg 0.9020.
- **Explanation (Finding 3 + 5):** bigger, better shared representations transfer
  directly downstream. The +0.006 gain came **almost entirely from the two weakest
  targets**: eps 0.7749 → 0.8009 (+0.026) and nc 0.8417 → 0.8657 (+0.024). This is the
  "pseudo-labelling done right" — features/pretraining, never labels.
- **Survived every later attack (v15, v16, v17, v18, v19, v20).**

---

## 10. v15 — EPS/NC Focus ✘ Hypothesis 3 (rejected)

- **Change:** doubled sample weight for eps/nc rows in the fine-tune loss
  (`TGT_FOCUS = {"eps": 2.0, "nc": 2.0}`).
- **OOF: −0.0051** weighted score vs P14 — **never submitted**.
- **Per-target (`kernel-v15-epsnc-focus/out/v15_blend_report.csv`):**
  eps 0.8009 → 0.7956 (−0.0053), nc 0.8657 → 0.8423 (−0.0234). Both **got worse**.
- **Explanation (Finding 6):** doubling the weight made the fine-tune overfit the tiny
  eps/nc folds (~229 rows each) while stealing gradient from the dominant targets. The P14
  gain came from stronger shared features, **not** loss reweighting. Branch closed.

---

## 11. v16 — Cross-Target Decoder (gate passed, LB failed)

- **Model:** P14 GNN extended with a physics-imputed arm (egc=ei−eea; egb=f(egc); eps=f(nc))
  plus a learned per-target Ridge over the 6 sibling targets → 4-arm blend.
- **Offline gate (pre-registered, 1,297 covered rows): PASS** — weighted gain +0.0118
  (≥ +0.003), worst target −0.0016 (> −0.003). **7/7 gates passed.**
- **Public LB: 0.874**, Δ **−0.009** vs P14 (submission `55289004`) — **FAIL, rejected**.
- **Explanation:** the covered 1,297-row subset was **NOT representative** of the public
  test distribution. An offline gain of +0.0118 — larger than the entire historical
  improvement range — should have raised skepticism, not lowered it. **Lesson:** at this
  scale, a +0.01 gain on a small multi-labeled subset is a small-sample artifact. The
  v16 cross-target family is closed.

---

## 12. v17 / v18 — Sibling-target blending (regressions)

- **v17** — sibling-target Ridge arm + physics eps: **LB 0.862** (Δ −0.021 vs P14).
- **v18** — hard-gate sib(halved) + physics: **LB 0.864** (Δ −0.019 vs P14).
- **Explanation:** same cross-target family as v16. The sibling pivot was built from FULL
  train and used unchanged inside GroupKFold val folds → **label leakage** (see v19).
  All OOF gains from sibling features failed to transfer; closed for this competition.

---

## 13. v19 — CT-PGCN (leakage artifact, not shipped)

- **Model:** same-engine CT-PGCN; sib + physics blend with alphas ≤ 0.30.
- **Local OOF: 0.8706** (+0.0063 over P14, **gates passed**) but **public LB 0.874**
  (Δ −0.009 vs P14) → **REGRESSION. Never shipped.**
- **ROOT CAUSE — OOF sibling leakage (same failure as v16):** the sibling pivot is built
  from FULL train and used unchanged inside GroupKFold val folds. GroupKFold groups on
  SMILES, so all 7 target rows of one polymer land in the same fold; a val-fold
  eps/nc/egb/ei/eea prediction uses that polymer's **TRUE other-target labels from the
  same val fold**. Measured leak-eligible val rows: eea 216/216, egb 274/274, egc 101/101,
  ei 216/216, eps 223/223 (~100%), nc 222/222, tg 4/10 (~40%).
- **The v19 eps OOF jump (+0.031) is ~entirely this artifact.** P14 avoids it (twin arms
  use model OOF predictions, not true labels) — which is exactly why P14 OOF tracks the LB
  and the sib/decoder family (v16 0.874, v17 0.862, v18 0.864, v19 0.874) does not.
- Artifacts kept for reference (not shipped): `vault/ctpgcn_v19.py`,
  `vault/ctpgcn_submission_v19.csv`.

---

## 14. v20 — Self-Trained SMILES Encoder (gate FAIL, not submitted)

- **Model:** compact RoBERTa-style masked-region SMILES encoder (~5–15M params) pretrained
  from scratch on PI1M *in-notebook* (pure-Python tokenizer, no `transformers`), frozen
  pooled embeddings → fold-safe per-target Ridge heads → 3-arm blend
  [GBM, MT-GNN, Transformer].
- **Local full-run gate (`v20_gate_report.csv`): FAIL.**
  mean R² 0.8643 (P14) → 0.8639 (v20), **Δ −0.0004** (needed ≥ +0.003); worst target **nc
  −0.0033** (needed > −0.003). Per-target: eea −0.0001 / egb +0.0001 / egc +0.0009 /
  ei −0.0001 / eps −0.0006 / nc −0.0033 / tg +0.0002.
- **Explanation:** the self-trained SMILES encoder arm added no signal on top of the GBM +
  GNN pair — the 3rd arm is near-collinear with the existing representation. Per
  pre-registration, no submission slot was spent. (Two Kaggle smoke runs also failed for
  environmental reasons — rdkit install / input path — before reaching the gate.)
- **Verdict: P14 remains FINAL.**

---

## 15. Supporting internal ablations (never submitted)

| Experiment | File | Result |
|-----------|------|--------|
| Retrieval feature ablation (v7 deep-dive) | `pipeline_out_pretrain/exp2_retrieval_feats.csv` | retrieval hurts or ties on all 7 targets |
| GBM + MT-GNN + stack (exp3) | `exp3_blend.csv` | stacking gains ≈ +0.01 over GBM only, no extra |
| Superblend (exp4, 2^3 combo) | `exp4_superblend.csv` | GBM+GNN mean 0.8632 best; +Retr 0.8621 (no gain) |
| MT-GNN vs GBM stack OOF | `mtgnn_v2_compare.csv` | MT-GNN better on egb/eps/nc/eea, worse on tg/egc/ei |
| v13 leak-only probe | `v13_leak_only_compare.csv` | leak-only OOF strongly negative (eps −0.10, nc −0.32) → confirms folds are clean |
| P14 vs GBM vs GNN per target | `kernel-v14-p1m/out/v14_blend_report.csv` | blend beats both arms on 7/7 targets |

---

## 16. What the campaign learned (the 6 findings)

1. **Retrieval adds little once a strong GNN exists** (v7, exp4).
2. **GBM + MT-GNN blending works** — +0.025 at v13 (Hypothesis 1 ✅).
3. **Full-scale pretraining works** — +0.006 at P14 (Hypothesis 2 ✅).
4. **Correlation reduction was NOT the mechanism** — corr rose 0.9513→0.9552, blend still gained.
5. **EPS/NC carried most of P14's gain** — eps +0.026, nc +0.024 of the +0.006 total.
6. **Loss reweighting for EPS/NC failed** (v15) — both targets got worse (✘ Hypothesis 3).

Plus the hard-won post-P14 lesson: **every cross-target/sibling blending OOF gain failed
to transfer to the LB** (v16 0.874, v17 0.862, v18 0.864, v19 0.874, v20 gate fail) —
the sibling family leaked through val folds, and the small multi-labeled subset is not
representative of the public test distribution.

## 17. Final standing & remaining path

- **Final submission: P14 (public LB 0.883), frozen.** Rule-compliant (pure kernel
  artifact, GroupKFold on smiles, arms fold-safe, no external files).
- **Gap:** ≈0.003 from top-20 (0.886), ≈0.015 from top-10 (0.898).
- **Untried, orthogonal candidates** (do NOT reuse the v16/v19 sibling mode):
  1. ModernBERT / pairwise-pretrained SMILES representation arm (NeurIPS 2025 winner technique).
  2. GRIN 3-unit repeat encoder as a 3rd GNN arm (only if OOF corr with P14 < 0.88).
  3. TransPolymer frozen embeddings → GBM (proven on eps/nc benchmarks).

---

## Verdict tally

- ✅ **Accepted hypotheses:** 2 (blend, full-scale pretrain).
- ✘ **Rejected hypotheses:** 1 (EPS/NC loss focus) + 4 failed branches (v7, v8, v12, v16/v17/v18/v19, v20).
- **LB floor improved:** 0.828 → 0.883 (+0.055 net over the campaign).

#experiment #log #pipeline #kaggle #final
