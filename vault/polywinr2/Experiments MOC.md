# 🧪 PolyWin R2 — Experiments MOC

> **Map of Content (MOC)** — All experiment nodes branch from here. Parent = the node it
> forked from; each experiment is one pre-registered change to its parent with a
> stopping gate. Follow the chain: `v4 → v6 → v11 → v13 → P14 → v15 → v16`.
>
> Experimental backbone: 5-fold GroupKFold on canonical [[SMILES]], per-target Ridge
> blend over `ALPHA_GRID`, unweighted mean R² of the 7 [[Target Properties]].

---

## 📈 Lineage (submitted waves, public LB)

| Node | Change vs parent | public LB | Δ | Verdict |
|------|------------------|-----------|-----|---------|
| [[v4 Baseline Stack]] | start (GBM trio + FFN) | 0.828 | — | start |
| [[v6 Honest OOF Stack]] | leak-safe OOF stack, no NNs | 0.847 | +0.019 | accepted → superseded |
| [[v7 Retrieval Augmented GBM]] | retrieval `FULL` features | *not submitted* | — | FAIL (OOF) |
| [[v8 PI1M Pseudo Labelling]] | PI1M pseudo-labels | 0.828 | −0.019 | FAIL (dropped) |
| [[v10 Pretrained GNN]] | pretrained MT-GNN | 0.830 | — | superseded |
| [[v11 Fold-Safe Blend]] | fold-safe blend weights | 0.852 | +0.022 | best standing |
| [[v12 Chemistry Bucket-MoE]] | chemistry bucket-MoE | 0.849 | −0.003 | FAIL (OOF ≠ LB) |
| [[v13 GBM + MT-GNN Blend]] | GBM + MT-GNN Ridge blend | **0.877** | +0.025 | Hypothesis 1 ✔ |
| [[P14 Full-PI1M Pretrain]] | full 995k PI1M pretrain (10 ep) | **0.883** | +0.006 | Hypothesis 2 ✔ **FINAL** |
| [[v15 EPS/NC Focus]] | EPS/NC ×2 loss weight | OOF −0.0051 | — | Hypothesis 3 ✘ |
| [[v16 Cross-Target Decoder]] | physics + learned decoder arms | 0.874 | −0.009 | FAIL (gate PASS'd, LB not) |

## 🔬 Hypotheses (pre-registered)

1. **GBM + MT-GNN blend beats GBM alone** → [[v13 GBM + MT-GNN Blend]] (accepted, +0.025)
2. **Full-scale pretraining improves the model** → [[P14 Full-PI1M Pretrain]] (accepted, +0.006)
3. **EPS/NC loss emphasis improves EPS/NC** → [[v15 EPSGradient Focus]] (rejected)

## 🧭 Findings (see `docs/lab-postmortem-2026-08-08.md`)

1. Retrieval adds little once a strong GNN exists → [[v7 Retrieval Augmented GBM]]
2. GBM + GNN blending works → [[v13 GBM + MT-GNN Blend]]
3. Full-scale pretraining works → [[P14 Full-PI1M Pretrain]]
4. Correlation reduction was **NOT** the mechanism (corr rose 0.9513→0.9552, blend still gained)
5. EPS/NC carried most of P14's +0.006 gain → [[EPS]] [[Nc]]
6. Loss reweighting for EPS/NC failed → [[v15 EPSGradient Focus]]

## 🗂 Supporting

- [[AISEHack 2.0 - Round 2 - MOC]] — rules & comp hub
- [[Target Properties]] — the 7 targets ([[Eea]], [[Egb]], [[Egc]], [[Ei]], [[EPS]], [[Nc]], [[Tg]])
- [[Dataset Files]], [[PI1M.csv]], [[Baseline Model]], [[Leaderboard and Evaluation]]

#hub #experiment #pipeline #kaggle