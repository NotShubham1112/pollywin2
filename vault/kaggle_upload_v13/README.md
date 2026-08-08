# PolyWin R2 — v13 Kernel (GBM trio stack + MT-GNN per-target Ridge blend)

## File to upload
`PolyWin_R2_v13_gbm_gnn_blend.ipynb`

## What it does (single, self-contained notebook)
1. **Data** — reads only the competition's current-round CSVs via `find_input(INP, ...)`:
   `train.csv`, `test.csv`, `PI1M.csv`. No private artifacts, no previous-round
   `official_dataset/archive`, no external pre-trained weights.
2. **Features** — RDKit canonicalization + 22 descriptors + Morgan/MACCS 2215-bit
   fingerprints, built in-kernel (2215 feature columns).
3. **Graphs** — molecular graphs built in-kernel (atom/bond one-hots, GINE).
4. **PI1M pretraining** — the shared GINE trunk is self-supervised pretrained
   *inside* the notebook on `PI1M.csv` (masked atom/bond reconstruction). No
   persisted weights bundled.
5. **Level-0 predictions (leak-safe)** — per-target LGBM twins, MT-GNN fold OOF
   (GroupKFold on canonical SMILES; a molecule's rows never span folds), and a
   GBM trio stack floor.
6. **v13 blend** — per-target `Ridge(alpha=1.0)` on `[GBM, GNN]`; then writes
   `submission_v13.csv` (id, target) and `v13_blend_report.csv`.

## Local validation (SMOKE unset, exact config)
- GBM stack OOF = 0.8435, MT-GNN OOF = 0.8398
- **v13 blend = 0.8638** (corr(GBM,GNN) 0.92–0.97)
- Submission: 4940 rows, 0 NaNs, output range sane.

## On Kaggle
Set `INP`/`WORK` automatically (cell 1 detects `/kaggle`); disable internet only
if needed — RDKit/PyG/LGBM/XGB/CatBoost are pre-installed and auto-verified.

## Time
Runs in roughly 6–12 min on a Kaggle T4 GPU (5 folds, 120 GNN epochs max,
20k pretrain SMILES cap; early-stopping keeps it bounded).