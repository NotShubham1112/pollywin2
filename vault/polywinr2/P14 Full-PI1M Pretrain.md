# P14 Full-PI1M Pretrain

Production model. Full-scale pretrain of the MT-GNN encoder on the entire ~995k-row
[[PI1M.csv]] unlabeled corpus (10 epochs of masked atom/bond reconstruction) before
fine-tuning on the competition targets.

- Public LB: **0.883** (best submission), Δ **+0.006** vs [[v13 GBM + MT-GNN Blend]]
- **FINAL production anchor — locked. Survived [[v15 EPS/NC Focus]] and [[v16 Cross-Target Decoder]].**
- Hypothesis 2 **accepted**: full-scale pretraining is the lesson of [[v8 PI1M Pseudo Labelling]]
  done right (features/pretrain, never labels).

**Why it matters:** the best experiment of the competition. Everything after this was a
hunting mission against it — and lost. See the running tally in the [[Experiments MOC]].

**Parents:** [[v13 GBM + MT-GNN Blend]] · [[PI1M.csv]]
**Next:** EPS/NC-two-loss rebalance → [[v15 EPS/NC Focus]] (rejected)

#experiment #pretraining #gnn #final #kaggle