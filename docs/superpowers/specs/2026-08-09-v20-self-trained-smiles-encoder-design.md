# v20 Design: Self-Trained SMILES Encoder Arm (Arm A), 3-arm Ridge

Date: 2026-08-09
Status: Approved (design + fine-tune format confirmed by user) -> spec in review

## 1. Goal

Beat P14 (LB 0.883, honest OOF mean-R2 0.8641) with a **leak-free, self-contained,
third representation arm** that is fused exactly the way P14 fuses arms, then
shipped as one Kaggle notebook. This follows the closure of all cross-target
sibling/physics blends (v16-v19: OOF gains failed to transfer; root cause = OOF
sibling leakage). The new arm contributes new representation, never a label/label
sibling feature.

## 2. Binding constraints (vault submissions + rules)

- Notebook/code-only competition. Every submission must run end-to-end inside one
  pinned Kaggle notebook, shared with the 5 hosts, linked in the description.
- **All weights/artifacts produced during the notebook run.** Forbidden: uploading
  or downloading pre-trained weights/checkpoints, external vectors, feature files,
  cached tensors, processed datasets.
- Use only official data (train.csv, test.csv, PI1M.csv, sample_submission.csv).
- OSI-licensed, commercial-use-ok libraries/code only.
- Leak-safe by construction: GroupKFold on `smiles`, no sibling/cross-target
  features in any arm.

Consequence: the SMILES encoder is **self-trained from scratch on PI1M inside the
notebook** (masked-token pretraining + per-target folds). No pre-trained download.

## 3. Pipeline (all inside the notebook)

1. Load train.csv (7409), test.csv (4940), PI1M.csv (995,799 SMILES strings,
   mean len ~47). Features for the trf arm come from the raw SMILES string only.
2. **Tokenizer**: fit a small SMILES tokenizer on PI1M strings with the
   `tokenizers` library (Apache-2.0; ByteLevel/BPE or regex-based tokenizer).
   Max length ~128. No external vocab.
3. **Masked-LM pretraining**: compact RoBERTa-style encoder, ~10-40M params
   (4-6 layers, hidden 256-384, ~8 heads), trained on a PI1M subsample with a
   masked-token objective. Gate run local: 5k-20k strings, 1-2 epochs,
   ~1-2h on the local RTX 3050 (6GB). Shipped run: 50k-200k strings on Kaggle GPU.
4. **Per-target fine-tune** (fold-safe, mirrors mt_gnn_v2):
   - One shared GroupKFold(5) across all 7409 rows, group = canonical `smiles`.
   - Per fold: train a small regression head per target on the *frozen* encoder
     pooling of the fold's train rows; predict that fold's val rows -> oof_trf.
     Test rows: bag of the 5 fold heads.
   - Heads are cheap per-target Ridge/linear heads on the pool (7 targets x
     5 folds = 35 small heads).
5. **Blend**: P14 object extended exactly: per-target Ridge over
   `[oof_gbm, oof_mt, oof_trf]`, alpha grid
   [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0], GroupKFold on smiles, per-target
   alphas. Apply to `[test_gbm, test_mt, test_trf]`.
6. **Output**: reuse the exact P14 submission writer (id order already
   validated; n=4940) -> `submission.csv`.

## 4. Pre-registered gates (same contract as v19, now leak-free)

Select the v20 blend as the new submission ONLY if all hold on the fold-safe OOF:

1. equal-weight mean R2 (7 targets) improves on P14 by **>= +0.003**
2. **no** single target regresses by more than **-0.003**
3. every per-target Ridge alpha lands in **[0, 0.30]** (no over-trust in trf)
4. notebook reproduces; OOF/test stable across 2 seeds

If any gate fails -> freeze P14, spend no submission slot.

## 5. Files / artifacts

- Python modules (local, OSI libs only):
  - encoder/tokenizer MLM train script
  - per-target fine-tune + blend + gate script
- `vault/pipeline_out_embed/` (arm vectors, cached locally only, never ingested
  to the Kaggle notebook)
- Repo notebook builder, mirroring `build_v14_kaggle_nb.py` (v20).

## 6. Risks & controls

- Engineering risk (encoder fit/CTS): moderate; sized to local 6GB GPU;
  CPU fallback acceptable for a small gate run.
- Transfer risk: an OOF gate passing is necessary, not sufficient. Mitigated by
  the small per-target Ridge alpha (arm is lightly weighted when uninformative;
  P14 remains dominant).
- Redundant-arm risk: if trf OOF correlates ~0.99 with existing arms the Ridge
  learns a near-zero weight; the gate inherently rejects it.
- 24h GPU / 3-day budget: first prove the arm locally, then one Kaggle GPU run
  if the gate passes.

## 7. Out of scope

- GRIN/long-range encoder arm (only if v20 gates and is orthogonal).
- Pairwise-pretrain ModernBERT (needs >16GB, exceeds budget).
- Any pre-trained weights in the shipped notebook (forbidden).
- Any sibling/physics cross-target feature (leak class, forbidden).

## 8. Timeline

1. Today: tokenizer + MLM trainer + per-target head + blend/gate modules; smoke
   on a few hundreds rows.
2. Gate run: 5k-20k PI1M, ~1-2h local.
3. If gate passes: build v20 notebook, smoke + bit-check, push, one-shot Kaggle.
4. If gate fails in budget -> P14 stays final; v20 is informational.