# v16 Cross-Target Decoder

Final experiment — a clean falsification of target-specialist reweighting.

Extends the [[P14 Full-PI1M Pretrain]] GNN into a **cross-target decoder**: a physics-prior
plus a learned decoder arm aiming to exploit the multi-target structure of the
1297-row covered subset.

- Public LB: **0.874** (submission `55289004`), Δ **−0.009** vs **P14 (0.883)** — **rejected**
- Offline scoring on the 1297-row covered multi-label subset looked excellent:
  weighted gain **+0.0118**, worst target **−0.0016**, **7/7 gates passed**
- The 1297-row subset was **NOT representative** of the public test distribution

**The lesson (falsified branch):** an offline gain of +0.0118 — larger than the entire
historical improvement range of the project — should have raised skepticism, not lowered
it. Improvements measured on the covered subset did not transfer to the full evaluation
distribution.

**⬛ Branch closed.** Together with the oracle-reweight analysis (no headroom) and the
[[v15 EPS/NC Focus]] failure, this branch of the search tree is exhausted.

**Parents:** [[P14 Full-PI1M Pretrain]]
**Next:** no further submissions on blend/target weighting — protect **P14**. Only genuinely
new signal (new descriptor/model family, new representation, new pretraining) earns a slot.

#experiment #reweighting #failed #postmortem #kaggle