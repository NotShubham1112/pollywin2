# v13 GBM + MT-GNN Blend

The blend that finally hooked: the GBM trio of [[v6 Honest OOF Stack]] Ridge-blended
with the **5-seed MT-GNN OOF** (seeds 42/999/2025), per-target alphas on `ALPHA_GRID`.

- Public LB: **0.877** (submission `55264007`), Δ **+0.025** vs [[v11 Fold-Safe Blend]]
- Hypothesis 1 **accepted**: the 4-arm Ridge blend beat GBM-alone and GNN-alone
- Correlation between the arms rose (0.9513 → 0.9552) yet the blend still gained —
  the only mechanism was adding a genuinely complementary model family

**Why it matters:** first line to break the 0.870 ceiling and the shape (per-target
Ridge over arms) that every later run reuses.

**Parents:** [[v6 Honest OOF Stack]] · [[v11 Fold-Safe Blend]] · 5-seed [[v10 Pretrained GNN]]
**Next:** pretrained at full scale (995k rows, 10–25 ep) → [[P14 Full-PI1M Pretrain]]

#experiment #blend #gnn #kaggle