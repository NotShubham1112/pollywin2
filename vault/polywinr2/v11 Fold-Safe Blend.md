# v11 Fold-Safe Blend

First regime shift to a real blend: GBM-trio OOF + MT-GNN OOF combined with **fold-safe**
weights (weights tuned on OOF inside the same GroupKFold that produced them).

- Public LB: **0.852** (submission `55286407`), Δ **+0.022** over [[v6 Honest OOF Stack]]
- This is when the dual-model hypothesis (#1) took off; next wave hardens the GNN itself.

**Parents:** [[v6 Honest OOF Stack]] · [[v10 Pretrained GNN]]
**Next:** a richer GNN base + tuned ridge → [[v13 GBM + MT-GNN Blend]]

#experiment #blend #oof #kaggle