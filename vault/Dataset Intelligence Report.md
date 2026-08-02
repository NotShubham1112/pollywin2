# AISEHack 2.0 — Polymer Property Prediction: Dataset Intelligence & Winning Architecture Report

**Scope:** Scientific forensics of `official_dataset` (train 7,409 rows, test 4,940 rows, PI1M ~996k polymers).
**Evidence:** figures and CSVs in `vault/figures/` (Phase 1–5 scripts reproducible).

---

## Executive Summary

- The task is **7 independent low-to-medium data regressions** sharing one chemistry space. 56% of labels are `Tg`; the five "small" targets (`egb, eps, nc, ei, eea`) each have only ~220–340 samples — this is the real difficulty.
- **No meaningful train/test covariate shift** (KS p>0.2 on MW for all targets) → plain shuffled CV is safe; no domain adaptation needed.
- **839 test SMILES already exist in train** (17% of test) → nearest-neighbour / retention leakage is available and legal; a kNN-style fallback is a free insurance policy.
- Chemistry is decisive: **Tg ↔ ring rigidity (ρ=0.78), Egc ↔ conjugation (ρ=−0.78), Nc/EPS ↔ sulfur & aromaticity (ρ≈0.7)**, Ei ↔ aromaticity (−0.67). Features that encode these will dominate.
- Empirically, **gradient boosting (XGBoost/LightGBM/HistGB) beats Ridge by 2×** on Tg and Egc (RMSE 35.2 vs 73.1; 0.50 vs 0.92). Ridge (the baseline) is a weak floor — even a modest pipeline doubles performance.
- **Multi-task sharing is strongly justified by physics:** on 418 molecules labeled with >1 property, Egc↔Egb ρ=0.94, EPS↔Nc ρ=0.92, Egc↔Eea ρ=−0.83. A shared representation for the five electronic targets (egb, eps, nc, ei, eea) is the single highest-leverage idea.
- **PI1M is exploitable but not trivial:** only 171/299 train molecules appear in it. Best ROI = pseudo-labelling + nearest-neighbour propagation + (optional) contrastive pretraining of fingerprints.

---

## Phase 1 — Dataset Forensics

### 1.1 Dataset size & imbalance

| target | count | % of train |
|---|---|---|
| tg  | 4143 | 55.9 |
| egc | 2028 | 27.4 |
| egb |  337 |  4.6 |
| eps |  229 |  3.1 |
| nc  |  229 |  3.1 |
| ei  |  222 |  3.0 |
| eea |  221 |  3.0 |

Test proportions mirror train (e.g., tg 2763/4940 = 55.9%). **Imbalance is 19:1 (tg vs eea).** RMSE is reported per-target (see baseline notebook: `root_mean_squared_error` per target), so **there is no global weighted metric — each target's RMSE contributes independently**. Do not let tg dominate; the small five must be handled with care.

### 1.2 Duplicates & leakage

| metric | value |
|---|---|
| unique SMILES (train) | 6565 |
| duplicate SMILES (train) | 844 |
| rows in duplicate groups | 1262 (17%) |
| SMILES shared across target types | 415 |
| conflicting labels (same SMILES+type, different target) | 3 |
| **test SMILES present in train** | **839 / 4940 (17%)** |
| test covered by PI1M | 175 / 4940 |

**Interpretation:**
- Only 3 conflicting labels — the dataset is well curated.
- **Leakage opportunity:** 17% of test rows have a near-exact train twin. A **nearest-neighbour prediction on Morgan fingerprint distance** for those rows is a guaranteed-quality fallback and can stabilise GBM predictions. This is **legal** (it uses train labels only).
- **Group-based CV is not strictly required** because labels are per-polymer, but with 17% duplicate polymers you must **deduplicate before any holdout that you use to measure generalization**; otherwise your CV RMSE is optimistic (same polymer in train and val folds). Use canonicalised SMILES (strip `*`/`[*]`) as the group key.

### 1.3 Target distributions & transforms

| target | mean | median | std | skew | kurtosis | min | max | log10-skew |
|---|---|---|---|---|---|---|---|---|
| tg  | 143.5 | 136.4 | 109.1 | 0.09 | −0.71 | **−109.8** | 495.0 | −7.7 |
| egc |  4.53 |  4.61 |  1.57 | −0.10 | −0.64 | 0.02 | 9.86 | −11.0 |
| egb |  4.28 |  4.05 |  1.98 | 0.44 | −0.50 | 0.51 | 10.1 | −9.1 |
| eps |  4.58 |  4.32 |  1.09 | 1.21 | 1.67 | 2.61 | 9.09 | −9.8 |
| nc  |  1.93 |  1.90 |  0.24 | 0.88 | 0.78 | 1.56 | 2.76 | −5.5 |
| ei  |  6.35 |  6.17 |  1.05 | 0.78 | 0.52 | 4.03 | 9.84 | −10.8 |
| eea |  2.28 |  2.27 |  1.11 | 0.22 | −0.79 | 0.39 | 5.14 | −6.4 |

**Transform guidance:**
- **Tg has negative values (−109.8)** → log transform invalid. It is mildly heavy-tailed and bimodal; use **robust/rank preprocessing** or raw + winsorisation. Do NOT log1p-shift blindly.
- **egc/egb/eea** are near-symmetric (|skew|<0.5) → no transform needed; GBMs handle outliers natively.
- **eps, nc, ei** are right-skewed (skew 0.8–1.2) → mild `log` or `Yeohson-Johnson` can help a linear/Ridge head, but **XGBoost/LightGBM are skew-agnostic** — keep raw.
- **Rank transform / quantile transform** is the safest universal option for GBM inputs and slightly improves small-target stability; test it per target.
- **Outliers:** tg min −109.8 (likely experimental artefacts), eps 2.61–9.09. Winsorise at 1–99th percentile **per target** only for linear models.

### 1.4 Test distribution shift

KS test on molecular weight (train vs test, per target): **all p > 0.2**, mean MW within 1% (e.g., tg 480.2 vs 469.7). No shift in heavy-atom counts. **Conclusion: the split is i.i.d. — no domain adaptation, no special weighting needed.** Use the entire train set (plus dedup) without reweighting.

---

## Phase 2 — Polymer Chemistry Analysis

### 2.1 Polymer family taxonomy (train)

| family | % |
|---|---|
| conjugated (≥1 aromatic ring) | 72.7 |
| heteroatom-rich (hetero ratio >0.15) | 77.0 |
| aromatic (aromatic ratio >0.3) | 68.4 |
| ring-rich (≥3 rings, ≥15 heavy) | 41.6 |
| sulfur-containing | 22.9 |
| linear / flexible | 21.6 |
| fluorinated | 11.3 |
| silicon-containing | 2.8 |
| rigid (≤2 rotatable, ≥2 rings) | 3.5 |

**The dataset is dominated by aromatic/conjugated/heteroaromatic polymers** — consistent with organic-photonics / electronic-materials target set (bandgap, EPS, Nc, Ei, Eea). The `*` atom is the chain-attachment dummy; treat SMILES as polymer repeat units.

### 2.2 Chemistry per target

| target | %aromatic | %conjugated | %S | %F | %linear | avg rings |
|---|---|---|---|---|---|---|
| tg  | 76.6 | 81.9 | 13.8 | 12.4 | 18.0 | 3.99 |
| egc | 53.9 | 58.3 | 19.7 |  7.1 | 35.6 | 1.84 |
| egb | 52.2 | 54.0 | 44.8 | 13.6 | 20.8 | 1.01 |
| eps | 69.4 | 69.9 | 63.8 | 13.1 |  7.0 | 1.17 |
| nc  | 68.6 | 69.9 | 61.6 | 15.7 |  5.7 | 1.16 |
| ei  | 68.0 | 68.5 | 64.9 | 16.2 |  5.9 | 1.16 |
| eea | 71.0 | 71.9 | 66.1 | 13.1 |  8.1 | 1.19 |

**Big insight:** the five "small" electronic targets (`eps, nc, ei, eea`, and to a degree `egb`) live in an almost identical chemistry regime — **sulfur-rich, aromatic, non-linear, low-ring-count** polymers. They differ mainly in the *outcome*, not the input distribution. This is exactly the situation where a shared multi-task backbone wins.

### 2.3 Structure–property drivers (empirical Spearman ρ)

| target | top-3 drivers (ρ) |
|---|---|
| tg  | ring_ratio 0.78, ring_count 0.75, frac_sp3 −0.73 |
| egc | n_conj_rings −0.78, frac_sp3 0.77, aromatic_ratio −0.76 |
| egb | frac_sp3 0.82, nS −0.71, n_conj_rings −0.68 |
| eps | frac_sp3 −0.70, nS 0.67, acceptors 0.45 |
| nc  | frac_sp3 −0.78, nS 0.73, n_conj_rings 0.62 |
| ei  | ring_ratio −0.68, aromatic_ratio −0.67, ring_count −0.61 |
| eea | nS 0.56, frac_sp3 −0.44, MW 0.26 |

**Scientific interpretation:**
- **Tg** ↑ with ring rigidity (more rings, higher ring-to-atom ratio, less sp3 flexibility) → glass transition is dominated by chain stiffness & packing; H-bonding acceptors matter secondarily. ✓ matches theory.
- **Egc (chain bandgap)** ↓ with conjugation (aromatic rings) and ↑ with sp3 content — the classic optical-gap relation: longer effective conjugation length narrows the gap. ✓
- **Egb (bulk bandgap)** — same physics as Egc but bulk/packing effects; high sp3 widens gap, sulfur narrows it (polythiophenes). ✓
- **Ei (ionisation energy)** ↓ with aromaticity/rings — delocalised π systems have lower IP (HOMO up). ✓
- **Eea (electron affinity)** ↑ with sulfur — thiophene/S-rich systems are electron-accepting. ✓
- **EPS & Nc** both ↑ with sulfur and polarisation and ↓ with sp3 — consistent with the **Clausius–Mossotti relation (ε≈Nc²)**, explaining the measured **EPS↔Nc ρ=0.92**. ✓

These give **physics-grounded features**: ring_density, conjugation_length, sulfur_ratio, hetero_ratio, HOMO/LUMO proxies, polarisability.

---

## Phase 3 — Feature Engineering Audit

| Feature family | Predictive power | Computational cost | Explainability | Verdict |
|---|---|---|---|---|
| **RDKit 2D descriptors** | High (baseline uses them) | Very cheap | High | Use all ~200; winsorise & drop constant |
| **Morgan fingerprints (r=2, 2048)** | **Highest** (substructure ↔ property) | Cheap | Medium | **Core feature**; add r=1 and count variant |
| **MACCS keys (167)** | Medium–high, interpretable substructure bits | Cheap | High | Use as complement (my bench added them) |
| **Topological / graph (rings, rot bonds, sp3)** | High for Tg/Egc | Cheap | High | Wrap into polymer features |
| **Polymer-specific** (aromatic ratio, ring density, hetero ratio, sulfur ratio, flexibility index, conjugation score, donor/acceptor counts) | **Highest ROI** — directly encode the Phase-2 drivers | Trivial | High | **Add all**; they beat raw descriptors |
| Physicochemical (logP, MR/polarisability, TPSA, HBD/HBA) | High for EPS/Nc/Ei | Cheap | High | Add |
| **Fingerprint-pair distances to train (nearest-neighbour)** | Medium (leakage-aided) | Medium | Low | Add kNN features for test rows with near twins |
| Transformer/LLM embeddings (ChemBERTa etc.) | Potentially high | High | Low | Optional Phase-2 upgrade |

**Feature ranking by expected contribution (for a GBM):**
1. Morgan (r2) + MACCS → chemistry identity
2. Polymer-specific ratios → physics drivers
3. RDKit 2D descriptors → breadth
4. kNN / pair-distance features → leakage exploitation
5. Pretrained embeddings → final squeeze

---

## Phase 4 — Model Strategy Analysis (empirical 5-fold CV, all features)

Measured on RDKit desc + Morgan2048 + MACCS + polymer features (see `model_benchmark.csv`):

| target | Ridge | RandomForest | HistGB | LightGBM | XGBoost |
|---|---|---|---|---|---|
| tg  | 73.06 | 39.05 | 35.92 | 35.29 | **35.18** |
| egc |  0.92 |  0.56 | 0.53 | 0.52 | **0.50** |
| small5 (merged) | 2.48 | **2.37** | 2.55 | 2.57 | 2.45 |

**Reads:**
- **XGBoost wins on tg & egc** (35.2, 0.50); LightGBM ≈ XGBoost but 2–3× faster.
- On the small targets, **RandomForest is best** (2.37) — tree averaging regularises well with 1238 samples.
- **Ridge is ~2× worse** everywhere → the baseline is very beatable.

### Scoring table (1–10)

| Model | LB perf | Interpretability | Speed | Robustness |
|---|---|---|---|---|
| Ridge | 2 | 9 | 10 | 3 |
| ElasticNet | 3 | 8 | 9 | 4 |
| Random Forest | 6 | 7 | 6 | 7 |
| XGBoost | 8 | 6 | 7 | 8 |
| LightGBM | 8 | 6 | 9 | 7 |
| CatBoost | 8 | 6 | 6 | 8 |
| HistGB | 7 | 6 | 8 | 7 |
| MLP (on desc+fp) | 6 | 4 | 7 | 5 |
| GNN | 7 | 3 | 5 | 6 |
| Transformer | 7 | 3 | 4 | 5 |
| Multi-task MLP | 8 | 4 | 7 | 6 |
| Multi-task GNN | 9 | 3 | 5 | 7 |

**Practical ranking for this competition:** XGBoost ≈ LightGBM > CatBoost > HistGB > RF > (multi-task GNN/MLP) > linear. **For the small five, prefer RF or a regularised GBM with strong shrinkage, or the multi-task shared model (below).**

---

## Phase 5 — Multi-Task Learning Investigation

**Measured cross-target correlations** (418 molecules labelled with >1 property, 1475 pairs):

| | tg | egc | egb | eps | nc | ei | eea |
|---|---|---|---|---|---|---|---|
| egc | 1.0* | 1 | 0.94 | −0.72 | −0.76 | 0.68 | −0.83 |
| egb | – | 0.94 | 1 | −0.70 | −0.82 | 0.67 | −0.71 |
| eps | – | −0.72 | −0.70 | 1 | 0.92 | −0.37 | 0.61 |
| nc  | – | −0.76 | −0.82 | 0.92 | 1 | −0.61 | 0.45 |
| ei  | – | 0.68 | 0.67 | −0.37 | −0.61 | 1 | 0.12 |
| eea | – | −0.83 | −0.71 | 0.61 | 0.45 | 0.12 | 1 |

(*only 2 tg cross pairs — unreliable)

**Conclusions:**
1. The five electronic targets are **physically coupled** (Eg+Ei−Eea≈2×IP−EA, EPS≈Nc²). This is textbook multi-task structure.
2. **tg is decoupled** from the rest (chemistry differs: 4-ring rigid polymers vs 1-ring sulfur polymers) → keep **tg separate**.
3. **Recommendation: Option D — Hybrid system.**
   - **tg:** dedicated single-target model (GBM stack). Its data is large and its chemistry unique.
   - **egc:** dedicated model OR shared with the electronic group — test both; egc has 2028 samples so it can stand alone, but sharing with egb/eps/nc/ei/eea adds 1238 samples of related signal.
   - **egb, eps, nc, ei, eea:** **one multi-task architecture** with a shared trunk (e.g., multi-head MLP on concatenated fingerprints+descriptors, or a shared GBM + per-target heads / or a LightGBM with `target_type` as a feature — my "small5 merged" benchmark shows a single global GBM reaches RMSE≈2.4–2.6 on this group).
4. A **single global model with `target_type` as feature** (Option B) is a good simple baseline for all 7 (one model, one CV), but **underperforms the hybrid** because tg's chemistry dominates the shared tree splits.

---

## Phase 6 — PI1M Strategy (995,799 unlabeled polymers)

| method | likely gain | compute cost | feasibility | verdict |
|---|---|---|---|---|
| **kNN / pseudo-label propagation** (PI1M → train nearest neighbours) | Medium (+3–8%) | Low | Easy | **Do first.** Only 299 train / 175 test rows are in PI1M, but unlabeled neighbours still regularise fingerprints. |
| **Pseudo-labelling with high-confidence predictions** | Medium | Low | Easy | Do for the small five (adds samples). |
| Contrastive pretraining of Morgan/fingerprint embeddings on PI1M | Low–Medium | Medium | Moderate | Optional; gains small vs. GBMs |
| Masked-SMILES / SMILES-BERT self-supervised pretraining | Medium | High | **Hard (no GPU guarantee, 996k × tokenisation)** | Only if you have a GPU; otherwise skip. |
| Graph neural net pretraining on PI1M | Medium–High | High | Hard | Only if using GNN as final model. |

**Recommendation:** pursue **pseudo-labelling + kNN propagation** first (cheap, safe). **Skip deep pretraining** unless a GPU is available — the 17% test overlap already gives a bigger, cheaper edge.

---

## Phase 7 — Cross-Validation Design

| strategy | leakage risk | verdict |
|---|---|---|
| Random KFold | Medium (17% duplicate polymers spill across folds) | OK only after dedup |
| **Stratified by target_type** | Low | **Required** — train per target, so stratify within each target's model |
| **Group KFold by polymer (canonical SMILES)** | **Lowest** — honest generalisation | **Recommended primary** |
| Scaffold Split | Low–Medium (some near-duplicates leak) | Good for robustness check |
| Repeated KFold (5×5) | Low | Recommended for final model selection (small targets) |

**Final design:**
- Canonicalise SMILES (strip `*`/`[*]`) → group key.
- **Group 10-fold CV** (groups = canonical polymer) for honest error.
- For the small five (n≈220–340), use **5×5 repeated Group CV** and report mean±std.
- **Do NOT use plain KFold on raw rows** — it will overstate accuracy because identical polymers appear in both train/val.

---

## Phase 8 — Ensemble Design

**Bronze (safe, ~beats baseline 2×):**
- Features: RDKit desc + Morgan2048 + MACCS + polymer-specific features.
- Models: LightGBM + XGBoost + HistGB per target (dedicated for tg/egc; merged `target_type` model for the small five).
- Blend: rank-average or weight by OOF RMSE (e.g., 0.4·XGB + 0.35·LGB + 0.25·HistGB).
- CV: Group 10-fold. Expect: tg ≈35, egc ≈0.50.

**Silver (strong):**
- Add kNN/near-neighbour features for the 839 overlapping test rows + pseudo-labels from PI1M.
- Add CatBoost and a RandomForest on the small five.
- Per-target target-encoding + winsorisation for linear blend head.
- Expect: tg ≈33–34, egc ≈0.47–0.48.

**Gold (contender):**
- Hybrid multi-task: shared backbone for {egb, eps, nc, ei, eea} (multi-head MLP on desc+fp+polymer feats, or shared-LGBM), separate tg & egc stacks.
- Stacking meta-learner (Ridge on OOF preds of all base models).
- 5×5 repeated Group CV, seed-blending, per-target optimal transforms.
- Expect: tg ≈31–33, egc ≈0.45, small-five RMSE −15–20%.

**Top-10 (if time/GPU permits):**
- Add pretrained embeddings (ChemBERTa / mol2vec) as extra GBM features + GNN head on the multi-task trunk.
- 10-fold × 5 seeds full ensemble; post-process with per-target physics bounds (e.g., ε≥1, Eg≥0, Nc in [1,3]).
- Use the 839 known test twins as anchor points to correct predictions.

---

## Phase 9 — Final Recommendation (the battle plan)

1. **Best feature set:** Morgan r2 2048-bit (bit + count) + MACCS + all RDKit 2D descriptors (winsorised, constant-dropped) + **polymer-specific physics features** (aromatic ratio, ring density, sulfur/hetero ratio, conjugation length, flexibility index, donor/acceptor counts, polarisability, logP) + **kNN/pair-distance features** for the 839 overlapping test rows.
2. **Best model architecture:** **Hybrid multi-task**: dedicated LightGBM+XGBoost stack for **tg**; dedicated stack for **egc**; **shared multi-head trunk for {egb, eps, nc, ei, eea}** (multi-task MLP or shared-LGBM with target_type feature); optional GNN head at Gold level.
3. **Best CV strategy:** **Group 10-fold** on canonicalised SMILES (polymer group); **5×5 repeated Group CV** for the small five; stratify by target_type within each model.
4. **Best ensemble strategy:** Stack base GBM OOF predictions with a Ridge meta-learner; blend models by OOF RMSE; seed-average 5×.
5. **Most important chemistry insights:** (a) Tg is a rigidity story (rings ↑, sp3 ↓); (b) Egc/Egb are conjugation stories (aromatic rings ↓ gap, sp3 ↑ gap); (c) EPS & Nc track each other (Clausius–Mossotti) and rise with sulfur/polarisability; (d) the five electronic targets share one chemistry regime → multi-task is the free lunch; (e) sulfur is the sleeper atom for Eea.
6. **Biggest leaderboard opportunities:** (1) multi-task sharing across the 5 small targets; (2) exploiting the 839-test-twin leakage with kNN; (3) PI1M pseudo-labelling; (4) simply switching Ridge→GBM (2× gain on tg/egc).
7. **Biggest risks:** (a) naive KFold optimism from duplicate polymers → inflated LB drop; (b) overfitting the small five (n≈220) — use strong shrinkage + repeated CV; (c) assuming log-transforms where invalid (tg has negatives); (d) treating all targets in one model and letting tg dominate.
8. **Fastest path to top performance:** (1) build the GBM stack on polymer-aware features with Group CV (this alone ≈ silver); (2) add the multi-task trunk for the electronic targets (largest remaining gain); (3) add kNN leakage features; (4) pseudo-label from PI1M; (5) stack + seed-average.

---

*Reproduction:* all numbers come from scripts in `vault/figures/` (`phase1_forensics.py`, `phase2_chemistry.py`, `phase3_benchmark2.py`, `phase5_multitask.py`) run against `official_dataset/`. Figures referenced: `target_frequency.png`, `target_histograms.png`, `transform_skew.png`, `test_shift_mw.png`, `polymer_families.png`, `chemistry_heatmap.png`, `model_benchmark.png`, `cross_target_corr.png`.
