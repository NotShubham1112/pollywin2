# v6 Honest OOF Stack

Second wave. Fixed the leakage in [[v4 Baseline Stack]] with an honest OOF meta-stack — no neural nets yet.

- Model: level-1 GBM trio → level-2 meta-regressor, all OOF-packed via GroupKFold
- No NNs; pure tabular stack
- Public LB: **0.847** (submission `55216423`), Δ **+0.019**

**Lesson:** leak-safe OOF packing alone buys +0.019. This is the template every later
blend reuses (fold-safe weights from [[v11 Fold-Safe Blend]] onward).

**Next:** the first deep learner branches in → [[v10 Pretrained GNN]]

#experiment #stack #oof #kaggle