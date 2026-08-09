# v12 Chemistry Bucket-MoE

Expert routing attempt: split the GBM training set by chemistry buckets and gate rows
through bucket-specialist GBM arms + a router.

- Public LB: **0.849** (submission `55255012`), Δ **−0.003** vs [[v11 Fold-Safe Blend]] — dropped
- OOF looked slightly better than the fold-safe blend; the public LB did not follow

**Lesson:** chemistry-bucket gating overfit the OOF partitioning. Same failure shape
as [[v7 Retrieval Augmented GBM]] (subset-logic signals that do not transfer). The
routing idea is shelved; the field does not need per-bucket experts once the GNN
arms exist.

**Parents:** [[v11 Fold-Safe Blend]]
**Next:** strengthen the GNN base instead → harmless 5-seed MT-GNN (42/999/2025) +
alphas → [[v13 GBM + MT-GNN Blend]]

#experiment #moe #failed #kaggle