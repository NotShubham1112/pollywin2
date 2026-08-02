# PI1M.csv

Unlabeled dataset of ~1 million polymer [[SMILES]] strings provided for advanced algorithms.

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Listed in: [[Dataset Files]]

---

## Details

| Column | Description |
|--------|-------------|
| `SMILES` | Polymer SMILES strings (no labels) |

## Potential Uses

- **Self-supervised pretraining** — masked atom/token prediction on SMILES or molecular graphs
- **Semi-supervised learning** — pseudo-labeling with a model trained on `train.csv`
- **Representation learning** — train embeddings on PI1M, fine-tune on the 7 [[Target Properties]]
- **Data augmentation** — canonicalization/enumeration strategies

Helps generalization → the [[Leaderboard and Evaluation|Private Leaderboard]] decides winners, so robust models beat public-LB overfitting.

---

#dataset #pretraining
