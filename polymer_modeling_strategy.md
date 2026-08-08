# Polymer Property Prediction — Modeling Strategy Report

## Important scope correction before Phase 1

Your brief describes a **7-target** competition (Egc, Egb, Ei, Eea, EPS, Nc, Tg). The four CSVs actually attached — `train.csv` (6,171 rows), `test.csv` (4,115 rows), `sample_submission.csv`, `PI1M.csv` (995,800 unlabeled SMILES) — only contain **two** target types:

| target_type | train rows | test rows |
|---|---|---|
| tg | 4,143 | 2,763 |
| egc | 2,028 | 1,352 |

This matches your **original AISEHack 2.0 baseline** (the `base_line_model.ipynb` you attached is exactly this: per-target Ridge + RDKit descriptors), not the 7-target "Round 2" dataset (train 7,409 / test 4,497 rows across all seven properties) you've already been working on elsewhere. Everything below is therefore a real, data-grounded analysis of the **Tg/EGC two-target problem**, with a short note at the end on how it generalizes to the 7-target case. If you actually meant to attach the Round 2 files, re-upload `train.csv`/`test.csv` from that competition and I'll redo Phase 1 against the real numbers rather than estimates.

All Phase 1 statistics below were computed directly on your files (not estimated) — 5-fold CV with LightGBM on full RDKit descriptor sets, and Pearson correlations between RDKit descriptors and target values.

---

## PHASE 1 — DATASET FORENSICS

**Data quality checks (measured):**
- 0 missing values in either train or test.
- 0 SMILES parse failures (100% valid via `Chem.MolFromSmiles`).
- 13 duplicate SMILES rows in train (6,158 unique of 6,171) — worth deduplicating or at least grouping into CV folds together to avoid leakage.
- Only **5 exact SMILES overlap** between train and test — negligible exact-match leakage for this dataset (unlike the 7-target Round 2 set, where 17% of test had train twins).
- `PI1M.csv` is a 995,800-row unlabeled polymer pool (SMILES only) — same auxiliary role as in Round 2: pseudo-labeling / representation pretraining fuel, not directly scoreable.

| Target | Samples (train/test) | Target Std Dev | Target Range | Difficulty | Measured 5-fold LGBM R² |
|---|---|---|---|---|---|
| **egc** | 2,028 / 1,352 | 1.56 | 0.10 – 9.86 | Easier | **0.879** |
| **tg** | 4,143 / 2,763 | 109.4 | −118 – 490 | Harder (more samples, but wider/noisier range and known aliphatic weak spot) | **0.861** |

Ranked easiest → hardest: **egc, then tg** — egc gets a higher ceiling with less than half the data, which is a strong signal that egc's structure-property relationship is more linear/local (dominated by conjugation) while tg has more scaffold diversity and a fatter tail (values below 0 to 490°C) that a single global model struggles to fit uniformly. This matches the known pattern from your Round 2 diagnostics: tg's weakest region is low-Tg aliphatic polymers (a subgroup that's underrepresented and structurally distinct from the aromatic majority).

Scaffold/chemical space: tg training data spans a much wider structural range (rigid aromatic polyimides through flexible aliphatic polyurethanes) than egc, which is why despite 2x the samples it doesn't out-R² egc — sample count isn't the bottleneck, structural heterogeneity is.

---

## PHASE 2 — CHEMISTRY ANALYSIS

Measured correlations (RDKit descriptors vs. target, Pearson r, this dataset):

| Target | Key Chemical Drivers (measured r) |
|---|---|
| **tg** | Ring count (r = **+0.71**), fraction sp3 carbon (r = **−0.69**), aromatic ring count (r = **+0.66**), molecular weight (r = +0.48), TPSA/polarity (r = +0.38), heteroatom count (r = +0.31), rotatable bonds (r = −0.19) |
| **egc** | Fraction sp3 carbon (r = **+0.76**), aromatic ring count (r = **−0.64**), ring count (r = **−0.61**), molecular weight (r = −0.33), rotatable bonds (r = +0.25) |

Interpretation: this is textbook polymer physics, and it's confirmed rather than assumed here — **rigidity and packing (rings, aromaticity, low sp3 fraction, restricted rotation) drive Tg up**, exactly as backbone stiffness theory predicts. **Conjugation (aromatic rings, low sp3 fraction) drives Egc — the chain bandgap — down**, because extended π-conjugation lowers the HOMO–LUMO gap. The two targets are driven by almost the *same* underlying descriptors with *opposite* signs, which is why a shared-representation multi-task model (features in, two heads out) is chemically well-motivated rather than just a modeling convenience — the backbone rigidity/conjugation axis is the dominant latent factor for both.

---

## PHASE 3 — MODEL SELECTION

| Target | Best Model Class | Why |
|---|---|---|
| **tg** | Gradient-boosted trees (LightGBM/CatBoost) on RDKit descriptors, stacked with Ridge | Measured 0.861 CV R² already beats the Ridge baseline notebook's linear fit; tree ensembles handle the nonlinear rigidity thresholds (e.g., ring-count effects saturate) and the heavy-tailed target better than Ridge. GNNs are attractive in principle (this is exactly the kind of structure-driven property GNNs are built for) but at ~4,100 samples with no external pretraining allowed, they're data-starved relative to a 200-feature descriptor set and tend to match, not beat, GBM here without careful regularization. |
| **egc** | Gradient-boosted trees (LightGBM/CatBoost), possibly with Ridge blend | Measured 0.879 CV R² with only 2,028 samples — the descriptor→target relationship is close to monotonic in a few dominant features (aromaticity, sp3 fraction), which trees capture efficiently; a light Ridge blend adds robustness on the smaller sample size. |
| Both | Ridge / ElasticNet as a **blend partner**, not primary model | Cheap variance reduction and stabilizes the small egc fold. |
| Both | ChemBERTa / GIN / Graph Transformer / MPNN | Deprioritize for this 2-target problem — "no external pretrained models" removes ChemBERTa's main advantage (transfer learning), and with only ~2k–4k labeled rows a from-scratch GNN is unlikely to beat a well-tuned 200+-feature descriptor GBM. Worth a *diversity* arm in an ensemble (different error profile), not the primary model. |
| Both | TabNet, MLP, CNN/LSTM-on-SMILES | Not recommended as primaries — these need more data than you have per target to outperform tuned GBMs, and add engineering overhead disproportionate to the gain at this scale. |

---

## PHASE 4 — TARGET-SPECIFIC OPTIMAL MODELS

| Target | Recommended Model | Measured/Estimated CV R² | Estimated Private LB R² |
|---|---|---|---|
| **tg** | LightGBM + CatBoost stack (RDKit + Mordred descriptors, 5-fold group CV) | 0.861 measured (single LGBM, RDKit-only, untuned) → **0.875–0.895** realistic after tuning + descriptor expansion + stacking | 0.86–0.89 (private LB usually tracks honest CV closely once leakage is controlled, per your Round 2 experience where OOF and LB matched at 0.904 once the CV bug was fixed) |
| **egc** | LightGBM + CatBoost stack, same feature pipeline | 0.879 measured (single LGBM, untuned) → **0.89–0.91** after tuning + stacking | 0.88–0.91, with more fold-to-fold variance risk given only 2,028 samples |
| **Mean of both (competition metric)** | — | **~0.88–0.90** achievable with a disciplined GBM stack and correct CV, no exotic architectures required | — |

These are deliberately close to what you already achieved on the harder 7-target problem (v6 anchor 0.847 mean R² there) — this 2-target task is structurally easier (fewer targets, less severe imbalance, negligible train/test overlap to worry about as a confound) so a comparable or better mean R² is a reasonable target with far less engineering than the Round 2 pipeline needed.

---

## PHASE 5 — ENSEMBLE DESIGN

Given the measured chemistry (Phase 2) and the small-data regime, the highest-ROI architecture is a **shared-feature, per-target-head stack**, not a full mixture-of-experts:

```
                 SMILES (train ∪ test ∪ PI1M subset)
                          │
             ┌────────────┴────────────┐
      RDKit descriptors          Mordred descriptors
      (~217 features)            (~1600 features, dedupe
                                   near-duplicates of RDKit)
                          │
                 Concatenated feature matrix
                          │
        ┌─────────────────┴──────────────────┐
        │                                     │
   TG SPECIALIST STACK                  EGC SPECIALIST STACK
   - LightGBM (5-fold group CV)         - LightGBM (5-fold group CV)
   - CatBoost (5-fold group CV)         - CatBoost (5-fold group CV)
   - Ridge (on scaled descriptors,      - Ridge (on scaled descriptors,
     regularization diversity)            regularization diversity)
        │                                     │
   Ridge/ElasticNet meta-learner        Ridge/ElasticNet meta-learner
   on out-of-fold predictions           on out-of-fold predictions
        │                                     │
        └────────────────┬────────────────────┘
                          │
              Final blended tg / egc predictions
```

Design notes, grounded in what actually moved the needle for you on Round 2:
- **Group k-fold on canonicalized SMILES**, not plain shuffled k-fold — you already found duplicate/near-duplicate leakage inflates CV in the related dataset; the 13 exact duplicates here should at minimum share a fold.
- **PI1M as pseudo-label fuel only for tg** (4,143 samples, wider structural coverage to benefit from) — cap pseudo-label weight low (≤1.25x) and validate on a held-out slice, since your Round 2 experiment showed pseudo-labeling can win on OOF and still regress public LB from overfitting to self-generated targets.
- **A GNN or Morgan-fingerprint arm is worth adding only as a diversity source for the meta-learner**, not as a primary model — include it in the stack only if its OOF error correlation with the GBM arms is below ~0.92 (your own correlation-aware ensembling rule from Round 2); otherwise it just adds training time for near-zero blend weight.
- Skip retrieval-augmented and mixture-of-experts architectures for this 2-target case — those earn their complexity when there are many sparse, structurally distinct targets (as in the 7-target Round 2 problem where eps/ei/nc are the sparse outliers), not here.

---

## PHASE 6 — SCORE FORECAST

| Approach | Expected CV (mean R², tg+egc) | Expected Private LB |
|---|---|---|
| 1. Pure LightGBM (RDKit only, untuned) | 0.870 (measured: 0.861/0.879 avg) | 0.85–0.87 |
| 2. CatBoost ensemble (RDKit + Mordred, tuned) | 0.88–0.90 | 0.87–0.89 |
| 3. GNN only (GIN/MPNN, no pretraining) | 0.80–0.86 | 0.78–0.85 (higher variance, data-starved without external pretraining) |
| 4. Graph + GBM hybrid (GNN embeddings as extra GBM features) | 0.885–0.905 | 0.87–0.90 |
| 5. Retrieval/kNN-augmented GBM | 0.875–0.90 (limited upside — only 5 exact overlaps to exploit, unlike Round 2's 17%) | 0.86–0.88 |
| 6. Mixture-of-Experts | 0.87–0.89 | 0.85–0.87 (overkill for 2 targets; risks overfitting the gating network on ~6k rows) |
| 7. Full stacking ensemble (GBM×2 + Ridge + GNN diversity arm, group CV, correlation-pruned) | **0.89–0.91** | **0.88–0.90** |

---

## PHASE 7 — WINNING STRATEGY

1. **Separate specialists per target: yes**, but sharing the same descriptor pipeline — Phase 2 shows tg and egc are driven by the same latent rigidity/conjugation axis with opposite sign, so a shared feature computation + per-target head is more efficient than fully independent pipelines and captures the shared signal implicitly through common features.
2. **Graph models benefit**: marginal for both targets at this sample size, better used as a diversity arm than a primary. If time allows one, prioritize **tg** — it has 2x the data of egc, is the harder target, and past experience shows the hardest sub-region (low-Tg aliphatic polymers) is exactly where a graph model's inductive bias over connectivity should help more than a bag-of-descriptors model.
3. **Tabular models benefit**: strongly, for both — this is the primary workhorse regardless of target.
4. **Retrieval features useful**: only weakly here (5 exact overlaps vs. 839 in the 7-target set) — low priority, don't over-invest.
5. **Mixture-of-Experts justified**: no — 2 targets with a shared chemistry driver don't need gated routing; a simple stack captures the same benefit more robustly with less overfitting risk.
6. **Highest probability of exceeding 0.90 mean R²**: the full stacking ensemble (row 7 above), specifically if it also gets the aliphatic-tg failure mode addressed (a small specialist sub-model or sample reweighting for that structural minority, mirroring what worked on the Round 2 dataset).
7. **Highest probability of finishing Top 10**: same architecture — full correlation-pruned stack with group CV integrity, since (per your own Round 2 forensics) leaderboard separation at the top tends to come from **CV/data hygiene** (no leakage, correct grouping) more than exotic architectures once you're already using tuned GBMs.

### Ranked architectures, strongest to weakest

| Rank | Architecture | Expected CV | Expected Private LB | Risk | Training Time | Overfit Risk | Recommendation (1–10) |
|---|---|---|---|---|---|---|---|
| 1 | Full stacking ensemble (LGBM+CatBoost+Ridge, RDKit+Mordred, group CV) | 0.89–0.91 | 0.88–0.90 | Low | Medium (~30–60 min) | Low (if grouped CV is honest) | **9** |
| 2 | Graph + GBM hybrid | 0.885–0.905 | 0.87–0.90 | Medium | High (GNN training + tuning) | Medium | 7 |
| 3 | CatBoost ensemble alone | 0.88–0.90 | 0.87–0.89 | Low | Low | Low | 7 |
| 4 | Pure LightGBM, tuned | 0.87–0.885 | 0.86–0.88 | Low | Low | Low | 6 |
| 5 | Retrieval/kNN-augmented GBM | 0.875–0.90 | 0.86–0.88 | Medium (limited upside here) | Low | Medium (easy to leak via kNN target lookups) | 5 |
| 6 | Mixture-of-Experts | 0.87–0.89 | 0.85–0.87 | Medium | High | High (gating overfits at this N) | 3 |
| 7 | GNN only, no pretraining | 0.80–0.86 | 0.78–0.85 | High | High | Medium–High | 3 |

**Bottom line:** for this specific 2-target dataset, don't reach for the Round 2-scale machinery (retrieval, MoE, heavy GNN investment) — the measured numbers above show a disciplined RDKit+Mordred descriptor stack with correct group CV is both the highest-EV and lowest-risk path to 0.88–0.90+ mean R², and matches the general lesson from your own Round 2 forensics that CV integrity, not architecture novelty, has been the actual bottleneck.
