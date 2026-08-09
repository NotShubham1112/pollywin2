# v4 Baseline Stack

The starting floor. First submitted wave of [[AISEHack 2.0 - Round 2 - MOC|AISEHack 2.0 Round 2]].

- Model: GBM trio (LGBM/CatBoost/XGB, HistGradientBoosting) + a small FFN
- Features: RDKit descriptors + Morgan/MACCS fingerprints (22 + 2215 cols)
- Public LB: **0.828** (submission `55194181`)

**Why it mattered:** the reference point against which every later [[Experiments MOC|experiment]]
is measured. Metric floor, not a contender.

**Next:** leak-safe stack without NNs → [[v6 Honest OOF Stack]]

#experiment #baseline #pipeline #kaggle