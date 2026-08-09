# v8 PI1M Pseudo Labelling

Failure: PI1M pseudo-labelling trained on out-of-fold predictions of [[PI1M.csv]].

- Idea: use the large unlabeled [[PI1M.csv]] corpus with OOF pseudo-labels as extra training
- Public LB: **0.828** (submission `55246041`), Δ **−0.019** vs [[v6 Honest OOF Stack]] — regression, dropped
- Design: `docs/superpowers/specs/2026-08-03-pi1m-pseudo-labelling-design.md`

**Lesson:** pseudo-labels amplified the model's existing errors — a clean OOF +0.017
translated to a public-LB −0.019. Closed; [[PI1M.csv]] is reused later only as a
**feature/pretraining** source (see [[P14 Full-PI1M Pretrain]]), not as labels.

**Next:** abandoned → the pretrain branch asks a different question → [[v10 Pretrained GNN]]

#experiment #pseudo-labelling #failed #kaggle