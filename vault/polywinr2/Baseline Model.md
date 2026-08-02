# Baseline Model

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Provided file: `baseline_model.ipynb` (see [[Dataset Files]])

---

## What the Baseline Does

End-to-end ML workflow provided by organizers:

1. **Featurization** — molecular descriptors generated from polymer [[SMILES]] using **RDKit**
2. **Preprocessing** — basic feature engineering and preprocessing
3. **Model** — **Ridge Regression**
4. **Output** — generates predictions and a valid `submission.csv` (format: see [[Submission Rules]])

## Ideas to Beat the Baseline

- Richer features: Morgan fingerprints, polymer-specific descriptors, graph features
- Stronger models: gradient boosting (XGBoost/LightGBM/CatBoost), GNNs, pretrained molecular transformers
- Leverage [[PI1M.csv]] for self-supervised / semi-supervised pretraining
- Multi-task learning across the 7 [[Target Properties]] (they share SMILES input)
- Per-[[Target Properties|target_type]] models

---

#baseline #model #machine-learning
