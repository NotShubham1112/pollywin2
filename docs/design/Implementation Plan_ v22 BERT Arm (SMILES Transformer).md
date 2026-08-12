# Implementation Plan: v22 BERT Arm (SMILES Transformer)

**Author:** Manus AI (AI Research Director)  
**Date:** 2026-08-11  
**Iteration:** v22  
**Goal:** 0.90+ LB via high-capacity self-supervised SMILES representation.

---

## 1. Phase 1: Modular Component Development (Local Sandbox)

### 1.1 `v22_tokenizer.py` (BPE Engine)
1.  Implement `learn_bpe` using a frequency-based merge strategy on a 150k stratified slice of `PI1M.csv`.
2.  Define `PROTECTED_TOKENS = ['*', '(', ')', '[', ']', '=', '#']` to ensure chemical topology is not merged into ambiguous tokens.
3.  Implement `encode` (SMILES $\rightarrow$ Token IDs) and `decode` (Token IDs $\rightarrow$ SMILES) with `[CLS]`, `[SEP]`, `[MASK]`, and `[PAD]` support.
4.  **Unit Test:** Round-trip verification (Original $\rightarrow$ Encoded $\rightarrow$ Decoded) on 100 complex SMILES.

### 1.2 `v22_encoder.py` (Transformer Core)
1.  Implement `SMILESBertEncoder` class using `nn.TransformerEncoder` with $d=384, L=6, H=8$.
2.  Implement `MLMTrainer` class:
    *   Masking: 15% random masking (excluding protected tokens).
    *   Optimizer: AdamW with Weight Decay.
    *   Scheduler: Linear warmup with Cosine Decay.
3.  Implement `extract_embeddings` function: Mean-pooling of the final hidden layer across non-padding tokens.
4.  **Unit Test:** Forward pass check for shape consistency `[Batch, 384]`.

### 1.3 `v22_arm_cv.py` (Ridge Cross-Validation)
1.  Implement `compute_bert_arm`:
    *   Load train/test SMILES.
    *   Extract frozen transformer embeddings.
    *   Run 5-fold `GroupKFold` (grouped by canonical SMILES).
    *   Fit per-target `Ridge` heads on embeddings.
2.  **Unit Test:** Leak audit (ensure no overlap between fold train-indices and validation-indices).

---

## 2. Phase 2: Integration and Gating

### 2.1 `v22_blend.py` (The 4-Arm Ensemble)
1.  Re-run the P14 baseline in the sandbox to generate reference OOFs.
2.  Integrate the `bert_oof` and `bert_test` arrays into the existing `[gbm, mt]` stack.
3.  Perform a per-target alpha grid search for the 4-arm blend.
4.  **Assertion:** `np.corrcoef(bert_oof, train_target)` must be within expected bounds to ensure row alignment.

### 2.2 `v22_gate_report.py` (The Verdict)
1.  Calculate $\Delta$ mean $R^2$ vs. P14.
2.  Verify Gate 1 (Leakage = 0) and Gate 3 (Floor $\ge -0.003$).
3.  Check Gate 2-soft ($\Delta \ge +0.0015$).
4.  **Output:** Write `v22_gate_report.csv` to `vault/`.

---

## 3. Phase 3: Notebook Build and Deployment

### 3.1 `build_v22_kaggle_nb.py`
1.  Aggregate all modular `.py` files into a single `.ipynb` structure.
2.  Add a "Smoke Test" flag to collapse PI1M pretraining to 100 rows for rapid environment verification.
3.  Final check for Kaggle P100 compatibility (memory usage and total runtime).

---

## 4. Timeline (Local Sandbox)
*   **Phase 1:** 45 Minutes
*   **Phase 2:** 20 Minutes
*   **Phase 3:** 10 Minutes
*   **Total Sandbox Time:** ~75 Minutes
