# Design Specification: v22 BERT Arm (SMILES Transformer)

**Status:** Approved  
**Author:** Manus AI (AI Research Director)  
**Date:** 2026-08-11  
**Iteration:** v22  
**Target:** 0.90+ Public LB in `ppp-round-2`  

---

## 1. Objective
To outperform the P14 baseline (0.883 LB) by introducing a high-capacity, self-supervised SMILES transformer arm. This iteration addresses the representation saturation observed in v20 by scaling the model parameters, vocabulary granularity, and pretraining data volume.

---

## 2. Architecture and Components

### 2.1 Tokenizer (`tokenizer.py`)
*   **Method:** Byte Pair Encoding (BPE) learned from a stratified 150k subset of `PI1M.csv`.
*   **Vocabulary Size:** ~4,000 tokens.
*   **Protected Tokens:** `*`, `(`, `)` are excluded from masking to preserve polymer topology.
*   **Properties:** Deterministic and reproducible from a fixed seed.

### 2.2 Encoder (`encoder.py`)
*   **Architecture:** `nn.TransformerEncoder`.
*   **Hyperparameters:** $d_{model} = 384$, 6 layers, 8 attention heads, learned positional embeddings.
*   **Parameters:** ~13 Million.
*   **Pretraining:** Masked Language Modeling (MLM) on the full 995k `PI1M.csv` corpus for ~1 epoch.
*   **Pooling:** Mean-pooling of final-layer hidden states (excluding `[PAD]` tokens).

### 2.3 Cross-Validation Arm (`arm_cv.py`)
*   **Logic:** `compute_bert_arm` utilizes frozen transformer embeddings.
*   **Heads:** Per-target fold-safe Ridge regression heads.
*   **Validation:** 5-fold GroupKFold stratified by canonical SMILES to prevent sibling leakage.

### 2.4 Blending Tier (`blend.py`)
*   **Strategy:** 4-arm Ridge blend: `[gbm, mt, bert]`.
*   **Optimization:** Per-target alpha sweep identical to the P14/v21 framework.

---

## 3. Execution and Gates

### 3.1 Runtime Budget (P100)
| Task | Estimated Time |
| :--- | :--- |
| P14 Pipeline Baseline | 1.5 Hours |
| BPE Tokenizer Training | 5 Minutes |
| MLM Pretraining (995k rows) | 30–45 Minutes |
| Ridge Heads & Blending | 5 Minutes |
| **Total** | **~2.5 Hours** |

### 3.2 Performance Gates
*   **Gate 0 (Diagnostic):** `bert_only_r2` reporting.
*   **Gate 1 (Audit):** Leak audit must return 0.
*   **Gate 2 (Soft Pass):** $\Delta \ge +0.0015$ improvement over P14 mean $R^2$.
*   **Gate 2 (Strong Pass):** $\Delta \ge +0.003$ improvement (Confidence Tier).
*   **Gate 3 (Floor):** Every individual target $R^2$ must be $\ge -0.003$ relative to P14.

---

## 4. Deliverables
1.  `v22_encoder.py`, `v22_tokenizer.py`, `v22_arm_cv.py`, `v22_blend.py`, `v22_gate_report.py`.
2.  `run_v22_gate.py` (Local harness).
3.  `build_v22_kaggle_nb.py` (Notebook generator).
4.  `PolyWin_R2_v22_bert_arm.ipynb` (Final Kaggle submission).
5.  `vault/pipeline_out_v22/v22_gate_report.csv` (Audit trail).

---

## 5. Fail Path
If the implementation fails any gate (specifically Gate 1, Gate 2-soft, or Gate 3), the verdict is recorded, no submission file is written, and the P14 (0.883) model remains the final submission.
