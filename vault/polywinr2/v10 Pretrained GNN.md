# v10 Pretrained GNN

First deep branch: the multi-task graph neural net (MT-GNN) entry. Its encoder is
self-supervised pretrained on unlabeled [[PI1M.csv]] chemistry *inside the notebook*.

- Model: GINE trunk (masked atom/bond reconstruction) → per-target heads
- Public LB: **0.830** (submission `55246047`?) — weaker alone than the fold-safe stack of [[v6 Honest OOF Stack]]
- Design ancestor: `docs/superpowers/specs/2026-08-03-polywin-v5-electronic-foundation-design.md`

**Why it matters (Finding 4):** the GNN alone underperformed the [[v6 Honest OOF Stack]];
its value only appears when **blended** with the GBM trio → [[v11 Fold-Safe Blend]] →
[[v13 GBM + MT-GNN Blend]].

**Parents:** [[v6 Honest OOF Stack]] (data flow) · [[PI1M.csv]] (pretrain source)
**Next:** fold-safe combination of GBM + GNN OOF → [[v11 Fold-Safe Blend]]

#experiment #gnn #pretraining #kaggle