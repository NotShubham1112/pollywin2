# v7 Retrieval Augmented GBM

Experimental dead-end: retrieval-augmented features on top of the GBM stack.

- Idea: hand-built molecular-similarity retrieval features (from [[Dataset Files]] neighbors)
- OOF: failed (no gain over [[v6 Honest OOF Stack]]); never submitted
- Design: `docs/superpowers/specs/2026-08-03-retrieval-augmented-gbm-design.md`

**Lesson (Finding 1):** once a strong encoder exists, retrieval adds little — the GNN
already captures molecular similarity implicitly. See [[v13 GBM + MT-GNN Blend]].

**Next:** abandoned in favor of the pretrained GNN branch → [[v10 Pretrained GNN]]

#experiment #retrieval #failed #kaggle