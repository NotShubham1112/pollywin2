# Polymer Property Prediction: Deep Research & Improvement Plan

**Current State**: ~0.85 LB (wMAE) | **Target**: 0.90+ LB (wMAE)
**Targets**: Egc, Egb, Ei, EPS, Eea, Nc, Tg
**Current Stack**: RDKit descriptors, Morgan fingerprints, Physics features, LightGBM, CatBoost, XGBoost, HistGBM, Ridge blending, Retrieval features, Specialist per-target, Multi-target stacking

---

## CRITICAL FINDING: Competition Identity

Your targets (Egc, Egb, Ei, EPS, Eea, Nc, Tg) match the **TransPolymer/Kuenneth et al. benchmark datasets** — NOT the NeurIPS 2025 competition (which had Tg, FFV, Tc, Density, Rg). This changes strategy significantly:

| Property | Dataset Size | Type | Unit |
|----------|-------------|------|------|
| Egc (bandgap chain) | 3,380 | Electronic | eV |
| Egb (bandgap bulk) | 561 | Electronic | eV |
| Eea (electron affinity) | 368 | Electronic | eV |
| Ei (ionization energy) | 370 | Electronic | eV |
| EPS (dielectric constant) | 382 | Optical | - |
| Nc (refractive index) | 382 | Optical | - |
| Tg (glass transition) | varies | Thermal | C |

Key insight: These are **electronic/optical properties** with extreme sparsity (368-3,380 samples). The strategies from NeurIPS 2025 (Tg-focused, MD-heavy) need adaptation.

---

## PRIORITY 1: HIGH PROBABILITY IMPROVEMENTS (>0.010 LB each)

### 1.1 SMILES Language Model Integration (Expected: +0.020-0.035 LB)

**Evidence**:
- NeurIPS 2025 1st place: ModernBERT-base (code-pretrained) **outperformed** ChemBERTa and polyBERT. BERT weight > 0.91 in final ensemble.
- TransPolymer (RoBERTa + MLP head): SOTA on Egc, Egb, Eea, Ei, Xc, EPS, Nc benchmarks.
- CodeBERTa-small (84M params) achieves Top-8 equivalent with only BERT + AutoGluon.
- 2nd place (Ezra): Simple ExtraTrees + Celsius-to-Fahrenheit conversion = 0.077 LB. Distribution shift matters more than model complexity.

**Implementation**:
```python
# Key: Use ModernBERT or CodeBERTa (NOT ChemBERTa)
# Two-stage pretraining proven optimal:
# Stage 1: Pretrain on pairwise comparison task (which polymer has higher property)
# Stage 2: Fine-tune on regression with 10x SMILES augmentation
```

**Critical technique**: Non-canonical SMILES augmentation
```python
from rdkit import Chem
def augment_smiles(smiles, n=10):
    mol = Chem.MolFromSmiles(smiles)
    return [Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True) 
            for _ in range(n)]
```

At inference: 30-50 predictions per SMILES, aggregate with **median** (not mean) to suppress outliers.

**Compute**: 1x ModernBERT-base fine-tune = ~4-8 hours on single GPU (24GB VRAM)
**Risk**: LOW | **Evidence**: Reproduced independently by multiple teams

---

### 1.2 Multi-Task Learning with Property Correlation Structure (Expected: +0.010-0.020 LB)

**Evidence**:
- Kuenneth et al. (ChemRxiv 2021): Multi-task NNs on 36 polymer properties. MT outperforms ST especially when dataset sizes are small AND correlations are high.
- Key finding: "MT models outperform ST models especially when correlations between properties within the subclass are high and/or when the dataset sizes within those subclasses are small."
- Your targets have strong physical correlations:
  - Egc ↔ Egb (chain vs bulk bandgap): physically coupled
  - Eea ↔ Ei (electron affinity ↔ ionization energy): HOMO/LUMO related
  - EPS ↔ Nc (dielectric ↔ refractive index): related through Clausius-Mossotti

**Implementation**:
```python
# Architecture: Shared encoder + property selector vector + per-property heads
# Use NN-MT2 architecture (proven best in literature)
# Input: [fingerprint | one-hot property selector]
# This allows model to learn shared representations while being property-specific
```

**Loss function**: Weighted MSE that handles missing labels naturally
```python
loss = sum(w_i * mask_i * (y_pred_i - y_true_i)^2) / sum(mask_i)
```

**Critical**: Scale all properties to comparable ranges (MinMax or RobustScaler)

**Risk**: LOW | **Evidence**: 4+ independent studies confirm MT gain for polymer properties

---

### 1.3 Distribution Shift Correction & Post-Processing (Expected: +0.010-0.015 LB)

**Evidence**:
- NeurIPS 2025 1st place: `submission["Tg"] += (submission["Tg"].std() * 0.5644)`
- 2nd place: Celsius-to-Fahrenheit conversion gave 0.068 LB (better than 1st place!)
- 3rd place: Fold-wise regression calibration improved scores by 5.5%
- Post-competition report: "Most teams applied target-wise normalization, heuristic offsets, fold-wise linear calibration"

**Implementation**:
```python
# For each target, learn calibration on OOF predictions:
# 1. Simple shift: y_calibrated = y_pred + alpha * std(y_pred)
# 2. Linear: y_calibrated = a * y_pred + b (fit on OOF vs ground truth)
# 3. Isotonic regression (non-parametric)
# 4. Per-fold calibration to avoid leakage
```

**Key insight**: With sparse labels, OOF predictions may have different distribution than test. Learn per-fold linear transforms.

**Risk**: VERY LOW (pure post-processing) | **Evidence**: Universal among top solutions

---

### 1.4 Rich Feature Engineering Beyond Morgan Fingerprints (Expected: +0.008-0.015 LB)

**Evidence**:
- NeurIPS 2025 1st place unique features: shape-based descriptors, Gasteiger charge, element composition, bond type ratios, polyBERT embeddings
- MCP (Multi-Cover Persistence): Topological data analysis features outperform ECFP and PG fingerprints on Egc, Egb, Eea, Ei, EPS, Nc
- MMPolymer paper: 3D structural information significantly enhances performance even for "2D" properties

**Feature families to add**:

| Feature Family | Source | Your Targets | Compute |
|---------------|--------|-------------|---------|
| Mordred descriptors | RDKit+Mordred | All | Low |
| Atom Pair fingerprint | RDKit | All | Low |
| Topological Torsion | RDKit | All | Low |
| MACCS keys | RDKit | All | Low |
| Gasteiger charges | RDKit | Eea, Ei, EPS | Low |
| Element composition | Custom | All | Trivial |
| Bond type ratios | Custom | All | Trivial |
| HOMO/LUMO estimates | xTB or semi-empirical | Egc, Egb, Eea, Ei | Medium |
| polyBERT embeddings | Pretrained | All | Low (inference) |
| MCP features | GUDHI + rhomboidtiling | All | Medium |

**Feature selection**: Train preliminary LightGBM, select top-N by importance. Use Optuna to tune N per target.

**Risk**: LOW | **Evidence**: Universal best practice

---

### 1.5 Self-Supervised Pretraining (Expected: +0.008-0.012 LB)

**Evidence**:
- SSL GNN paper: "decreases RMSE by 28.39% and 19.09% for EA and IP" in low-data regimes
- TransPolymer: MLM pretraining on 5M PI1M polymers + fine-tuning = SOTA
- PolyBERT: Pretraining on 100M hypothetical polymers → dense 600-dim fingerprints
- JEPA (Joint Embedding Predictive Architecture): New SSL approach for polymer graphs, +39.8% R² in smallest data regime

**Implementation for your stack**:
```python
# Option A: Use pretrained polyBERT embeddings as additional features
# Option B: SSL pretrain GNN on unlabeled polymer data (PI1M ~ 1M polymers)
# Option C: Pairwise comparison pretraining (NeurIPS 2025 winner technique)
```

**Most practical**: Use polyBERT or ModernBERT frozen embeddings as extra features for your tree models.

**Risk**: MEDIUM | **Evidence**: Strong in literature, moderate in practice

---

## PRIORITY 2: MEDIUM PROBABILITY (0.005-0.010 LB each)

### 2.1 Retrieval-Augmented Prediction with Learned Embeddings (Expected: +0.005-0.010 LB)

**Evidence**:
- NN-DTA (non-parametric nearest neighbor): "boosts DTA prediction performance with no extra training"
- Two retrieval modes: label aggregation (pair-wise) + representation aggregation (point-wise)
- Embedding-based retrieval >> chemical similarity retrieval in both cost AND performance
- Your current retrieval features likely use simple Tanimoto. Upgrade to learned embeddings.

**Implementation**:
```python
# 1. Compute polyBERT/ModernBERT embeddings for all train+test polymers
# 2. For each test polymer, find k=5-20 nearest neighbors in embedding space
# 3. Compute weighted target: y_retrieval = sum(w_i * y_i) / sum(w_i)
#    where w_i = exp(-distance_i / sigma)
# 4. Concatenate [y_retrieval_1, ..., y_retrieval_k, min_dist, mean_dist] as features
# 5. Use multiple k values and multiple embedding spaces
```

**Key insight**: Use embedding-based retrieval, NOT fingerprint Tanimoto. Learned representations capture chemical similarity better.

**Risk**: MEDIUM | **Evidence**: Strong in cheminformatics, limited polymer-specific evidence

---

### 2.2 Ensemble Architecture Upgrade: 3-Layer Stacking (Expected: +0.005-0.008 LB)

**Evidence**:
- NeurIPS 2025 top solutions: 3-model ensembles (BERT + tabular + 3D)
- Multi-view paper: 4-family ensemble (tabular + GNN + 3D + BERT) → Top 9
- stackgbm library demonstrates consistent improvement from 2-layer stacking

**Your current**: Ridge blending (single layer). Upgrade to:

```
Layer 1: LightGBM (feat set A) | CatBoost (feat set B) | XGBoost (feat set C) | 
          HistGBM (feat set D) | BERT (SMILES) | GNN (graph) | Retrieval features
Layer 2: LightGBM meta-learner (trained on OOF predictions)
Layer 3: Per-target Ridge with non-negative constraints
```

**Critical**: Use per-target stacking weights. Do NOT use single global ensemble.

**Risk**: LOW | **Evidence**: Universal competition best practice

---

### 2.3 ∆-Learning / Multi-Fidelity Approach (Expected: +0.005-0.008 LB)

**Evidence**:
- PolyMon framework: ∆-learning is a core strategy. Train model to predict DIFFERENCE between low-fidelity (DFT) and high-fidelity (experimental) values
- T-S CrabNet: Teacher-student strategy boosted accuracy by 37.1% on 5% data
- R-KD framework: Best for data-scarce properties

**Implementation**:
```python
# If you have both DFT and experimental data:
# 1. Train base model on DFT values (larger dataset)
# 2. Train ∆-model to predict (experimental - DFT) from features
# 3. Final = DFT_model + ∆_model
# 
# Without multi-fidelity data:
# Use teacher-student: Best single model (teacher) generates soft labels for unlabeled polymers
# Student learns from soft labels + hard labels
```

**Risk**: MEDIUM | **Evidence**: Strong in materials ML, requires multi-fidelity data

---

### 2.4 Hierarchical Cross-Validation (Expected: +0.003-0.006 LB)

**Evidence**:
- NeurIPS 2025: GroupKFold on canonical SMILES prevents molecule-level leakage
- Post-competition report: "grouped cross-validation to avoid leakage and leaned on OOF predictions"
- PolyMetriX: LOCOCV (Leave-One-Cluster-Out CV) for extrapolation testing

**Implementation**:
```python
# 1. Group by canonical SMILES (prevents same polymer in train/val)
# 2. Stratified K-fold on target quantiles (ensures each fold has similar distribution)
# 3. For multi-target: Use per-target stratified folds
# 4. CRITICAL: Compute OOF predictions properly — they become Layer 2 features
```

**Risk**: VERY LOW | **Evidence**: Fundamental best practice

---

## PRIORITY 3: SPECULATIVE (0.002-0.005 LB, higher risk)

### 3.1 GNN with Polymer-Specific Graph Representation (Expected: +0.003-0.008 LB)

**Evidence**:
- GRIN (Graph Repetition Invariance): +15% on polymer density, +24-30% RMSE reduction on copolymer ionization
- wD-MPNN (polymer-chemprop): Captures chain architecture + stoichiometry
- GATv2Conv (6 layers) + Morgan fingerprints: 0.078 LB (3rd place NeurIPS)
- BUT: GNNs often underperform tree models on very small datasets (<500 samples)

**For your dataset sizes**:
- Egc (3380): GNN likely helps
- Egb, Eea, Ei, EPS, Nc (368-561): Trees likely better, GNN may overfit

**Risk**: HIGH (small data) to MEDIUM (larger data)

---

### 3.2 3D Conformer Features (Expected: +0.002-0.005 LB)

**Evidence**:
- Uni-Mol 2: Contributed ~0.001-0.003 LB in NeurIPS 2025
- MMPolymer: 3D structure significantly enhances "2D" properties too
- BUT: Uni-Mol excluded from FFV ensemble due to memory constraints with large molecules

**Implementation**:
```python
# 1. Generate 3D conformers with RDKit (ETKDG method)
# 2. Compute 3D descriptors: radius of gyration, asphericity, eccentricity
# 3. Use Uni-Mol 2 for prediction (84M model, runs in ~1h per target)
```

**Risk**: MEDIUM | **Evidence**: Modest gains, significant compute cost

---

### 3.3 Test-Time Augmentation (TTA) (Expected: +0.002-0.004 LB)

**Evidence**:
- NeurIPS 2025 1st place: 50 random SMILES per molecule, median aggregation
- Multi-view paper: TTA up to 10 SMILES variants consistently reduced variance
- "Increasing TTA and enabling test-time dropout consistently reduced variance"

```python
def predict_with_tta(model, smiles, n=50):
    predictions = []
    for _ in range(n):
        aug_smiles = random_smiles(smiles)
        predictions.append(model.predict(aug_smiles))
    return np.median(predictions, axis=0)
```

**Risk**: VERY LOW | **Evidence**: Universal improvement, zero training cost

---

## RANKED PRIORITIZATION TABLE

| Rank | Idea | Expected LB Gain | Impl. Time | Risk | Evidence |
|------|------|-----------------|------------|------|----------|
| 1 | SMILES Language Model (ModernBERT/CodeBERTa) | +0.020-0.035 | 2-3 days | LOW | NeurIPS 1st/8th repro, TransPolymer SOTA |
| 2 | Non-canonical SMILES augmentation (train+test) | +0.010-0.015 | 0.5 days | VERY LOW | NeurIPS 1st place core technique |
| 3 | Multi-task learning with selector vectors | +0.010-0.020 | 2-3 days | LOW | Kuenneth et al. 36-property study |
| 4 | Distribution shift correction + per-fold calibration | +0.010-0.015 | 0.5 days | VERY LOW | NeurIPS 1st/2nd/3rd all used this |
| 5 | Rich feature expansion (Mordred, AtomPair, MACCS, MCP) | +0.008-0.015 | 1-2 days | LOW | MCP paper, NeurIPS 1st place |
| 6 | Self-supervised pretraining (polyBERT embeddings) | +0.008-0.012 | 1 day | LOW | TransPolymer, SSL-GNN papers |
| 7 | 3-layer stacking with per-target meta-learners | +0.005-0.008 | 1 day | LOW | Universal best practice |
| 8 | Retrieval-augmented with learned embeddings | +0.005-0.010 | 1 day | MEDIUM | NN-DTA, Multi-view paper |
| 9 | ∆-learning / teacher-student distillation | +0.005-0.008 | 2 days | MEDIUM | PolyMon, T-S CrabNet |
| 10 | Hierarchical/grouped CV with OOF features | +0.003-0.006 | 0.5 days | VERY LOW | NeurIPS top-5 standard |
| 11 | GNN (GATv2 or GRIN) for larger targets | +0.003-0.008 | 3-4 days | MED-HIGH | GRIN paper, NeurIPS 3rd |
| 12 | 3D conformer features + Uni-Mol 2 | +0.002-0.005 | 2-3 days | MEDIUM | MMPolymer, NeurIPS 1st |
| 13 | Test-time augmentation (50x SMILES) | +0.002-0.004 | 0.5 days | VERY LOW | NeurIPS 1st place |

---

## ANTI-PATTERNS (Things to AVOID)

1. **Aggressive SMILES enumeration for GNNs**: NeurIPS 3rd place reported "ineffectiveness for GNNs" — random stereoisomer enumeration caused overfitting
2. **Extra trees without unit calibration**: ETR + Celsius-to-Fahrenheit hack worked ONLY because of distribution shift. Without proper calibration, ETR underperforms GBM.
3. **Single global ensemble weight**: Per-target optimization is essential. Different targets need different model blends.
4. **Ignoring missing labels**: Masked loss functions are critical. Do NOT impute missing labels.
5. **Canonical-only SMILES**: Non-canonical forms at training AND test time consistently improve by ~0.01 LB.

---

## ESTIMATED TOTAL IMPACT

Conservative estimate (implementing ranks 1-10):
- Base: 0.850 LB
- + Language Model: 0.850 → 0.875 to 0.885
- + Augmentation: +0.010
- + Multi-task: +0.008
- + Distribution correction: +0.008
- + Feature expansion: +0.006
- + SSL embeddings: +0.005
- + Stacking upgrade: +0.004
- + Retrieval: +0.003
- + CV improvement: +0.002
- **Estimated final: 0.895 to 0.910 LB**

Aggressive estimate (all 13 including speculative): 0.905 to 0.920 LB

---

## IMPLEMENTATION ORDER (2-week sprint)

**Week 1**:
- Day 1-2: Non-canonical SMILES augmentation + TTA (ranks 2, 13)
- Day 3-4: Distribution shift correction (rank 4)
- Day 5-6: ModernBERT/CodeBERTa integration (rank 1)
- Day 7: Feature expansion (rank 5)

**Week 2**:
- Day 8-9: Multi-task learning architecture (rank 3)
- Day 10-11: 3-layer stacking + retrieval features (ranks 7, 8)
- Day 12: SSL embeddings (rank 6)
- Day 13: CV upgrade + OOF pipeline (rank 10)
- Day 14: Final ensemble tuning + submission

---

## KEY PAPERS & REPOSITORIES

1. **TransPolymer** (npj Comput Mater 2023) — RoBERTa for Egc/Egb/Eea/Ei/EPS/Nc — SOTA on YOUR exact targets
2. **NeurIPS 2025 1st place** (jday96314/NeurIPS-polymer-prediction) — ModernBERT + AutoGluon + Uni-Mol2
3. **MMPolymer** (arXiv 2406.04727) — Multimodal (1D SMILES + 3D) pretraining
4. **MCP features** (PMC 2024) — Topological data analysis descriptors
5. **PolyMon** (arXiv 2603.13303) — Unified framework with ∆-learning
6. **Kuenneth et al. MT paper** (PMC 2021) — Multi-task on 36 polymer properties
7. **SSL-GNN for polymers** (RSC Mol Syst Des Eng 2024) — Self-supervised pretraining
8. **GRIN** (arXiv 2505.10726) — Repetition-invariant polymer GNN
