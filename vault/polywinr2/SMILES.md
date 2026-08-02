# SMILES

**Simplified Molecular Input Line Entry System** — a text notation for representing molecular structures.

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Used in: [[Dataset Files]] · Featurized by: [[Baseline Model]] (RDKit)

---

## Role in this Competition

- Polymers are given as SMILES strings in both `train.csv` and `test.csv`.
- Polymer SMILES typically include `*` wildcards marking **repeat-unit connection points**.
- All 7 [[Target Properties]] must be predicted **purely from the SMILES structure**.

## Featurization Options

- **RDKit molecular descriptors** (baseline approach → [[Baseline Model]])
- **Morgan/circular fingerprints** (ECFP-style)
- **Graph representation** → GNN models
- **String/transformer embeddings** → pretrained chemical language models
- Augment unlabeled structures from [[PI1M.csv]] for pretraining

---

#concept #chemistry #features
