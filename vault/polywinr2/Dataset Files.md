# Dataset Files

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Context: [[Competition Overview]]

---

## train.csv

Training dataset — **7,409 polymer property measurements** spanning the 7 [[Target Properties]].

| Column | Description |
|--------|-------------|
| `smiles` | [[SMILES]] representation of the polymer structure |
| `target` | Experimental value of one of the seven polymer properties |
| `target_type` | Property category corresponding to the target value ([[Egc]], [[Egb]], [[Ei]], [[Eea]], [[EPS]], [[Nc]], [[Tg]]) |

## test.csv

Test dataset used for prediction — **4,497 data points**.

| Column | Description |
|--------|-------------|
| `id` | Unique sample identifier |
| `smiles` | [[SMILES]] representation of the polymer structure |
| `target_type` | Property to be predicted |

Predictions on this file feed the [[Leaderboard and Evaluation|Public and Private Leaderboards]].

## PI1M.csv

Additional polymer [[SMILES]] dataset (~1M structures) for **advanced algorithms** — e.g. self-supervised pretraining, semi-supervised learning, grap

h/transformer pretraining. It has **no labels**.

| Column | Description |
|--------|-------------|
| `SMILES` | Polymer SMILES strings |

## sample_submission.csv

Example file showing the required prediction format — see [[Submission Rules]].

## baseline_model.ipynb

Provided starter notebook — see [[Baseline Model]].

---

#dataset #data #kaggle
