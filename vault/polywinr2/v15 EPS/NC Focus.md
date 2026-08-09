# v15 EPS/NC Focus

Loss rebalance arm: tried to lean the GNN loss harder on [[EPS]] and [[Nc]] (×2 loss
weight) — the two targets that carried most of [[P14 Full-PI1M Pretrain]]'s +0.006 gain.

- OOF: **−0.0051** weighted score vs P14 — worse, never submitted
- Hypothesis 3 (**EPS/NC loss emphasis improves EPS/NC**) **rejected**

**Why it matters:** kills the "if EPS/NC already drive gains, weighting them helps"
chain of reasoning. Also closed branch #2 by probing what a **re-centred** version of the
P14-channel might do — degenerate.

**Parents:** [[P14 Full-PI1M Pretrain]]
**Next:** further target reweighting → [[v16 Cross-Target Decoder]] (closed)

#experiment #loss #rejected #kaggle