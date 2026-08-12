# PolyWin R2 – AISEHack 2.0 Polymer Property Prediction Round 2

**Kaggle Competition**: [PPP Round 2 Rules](https://www.kaggle.com/competitions/ppp-round-2/rules)  
**Targets**: EEA, EGB, EGC, EI, EPS, NC, Tg (7 polymer properties)  
**Metric**: Mean R² across all targets  
**Best LB Score**: 0.883 (P14 baseline submitted 2026-08-10)

## Repository Structure

```
┌─────────────────────────────────────────────────────┐
│  README.md          .gitignore                  │
├─────────────────────────────────────────────────────┤
│  competition/       rules/  data/                │
│  ├── rules/        └── raw/  archive/          │
│  │   ├── AISEHack rules documentation           │
│  │   ├── raw/      └── train.csv  test.csv    │
│  ├── src/                                   │
│  │   ├── core/      └── mt_gnn_v2.py           │
│  │   ├── v22_bert_arm/   └── v22_encoder.py   │
│  │   ├── v22_arm_cv.py   └── v22_blend.py     │
│  │   ├── v22_tokenizer.py                        │
│  │   ├── v22_arm_cv.py                           │
│  │   ├── v22_blend.py                            │
│  │   ├── v22_gate_report.py                      │
│  │   ├── build_v22_kaggle_nb.py                  │
│  │   ├── run_v22_gate.py                         │
│  │   ├── tests/                                 │
│  │   └── v20_smiles_encoder/                    │
│  │       ├── v20_encoder.py                      │
│  │       ├── v20_arm_cv.py                       │
│  │       ├── v20_blend.py                        │
│  │       ├── v20_gate_report.py                   │
│  │       └── run_v20_gate.py                     │
│  ├── notebooks/                               │
│  │   ├── v14_p14_baseline/                       │
│  │   ├── v22_bert_arm/                          │
│  │   ├── v21_sibling_arm/                       │
│  │   ├── v20_smiles_encoder/                     │
│  │   ├── v16_cross_target/                       │
│  │   ├── v15_epsnc_focus/                        │
│  │   ├── v14_p1m_pretrain/                       │
│  │   ├── v13_blend/                              │
│  │   ├── v12_bucket_moe/                         │
│  │   ├── v11_reproduce/                          │
│  │   ├── v10_pretrain/                           │
│  │   ├── v9_gnn/                                 │
│  │   ├── v13_specialist/                         │
│  │   ├── v2_validation/                          │
│  │   ├── experiments/                            │
│  │   ├── pipeline/                               │
│  │   └── notebooks/                              │
│  ├── tests/                                     │
│  ├── vault/                                     │
│  └── submission_v17_final.csv                    │
└── src/                                          │
    ├── core/            # Shared core modules
    ├── mt_gnn_v2.py     # P14 main source (MT-GNN with GBM stack)
    ├── decoder_v16.py   # Cross-target decoder (v16)
    ├── v22_bert_arm/    # Latest experiment: BERT SMILES encoder
    ├── v22_tokenizer.py # Tokenizer for BERT encoder
    ├── v22_arm_cv.py    # ARM CV experiments
    ├── v22_blend.py     # Blending strategies
    ├── v22_gate_report.py # Gate report generation
    ├── build_v22_kaggle_nb.py # Kaggle notebook builder v22
    ├── run_v22_gate.py   # P14 gate evaluation
    ├── tests/            # Unit tests for v22
    └── v20_smiles_encoder/ # Self-trained SMILES encoder (v20)
```

## Quick Start

1. **Clone/Checkout** the repository
2. **Install dependencies** (if needed)
3. **Run the v22 BERT encoder** to reproduce the winning solution:
   ```bash
   python src/v22_bert_arm/run_v22_gate.py
   ```
4. **Submit** your final model to the Kaggle competition.

## Key Components

- **Core Architecture** (`src/core/`): Shared modules for GNN models, embeddings, and pipelines
- **BERT Encoder** (`src/v22_bert_arm/`): State-of-the-art SMILES encoder trained on P14 dataset
- **Experiment Tracking** (`src/v22_*/`): All experiments organized by version (v10→v22)
- **Notebooks** (`notebooks/`): Ready-to-run Jupyter notebooks for reproducibility
- **Competition Data** (`competition/`): Raw data, train/test splits, and submission files

## Best Result

- **P14 Baseline**: 0.883 Mean R² (submitted 2026-08-10)
- **Final Submission**: `submission_v17_final.csv`

## License

MIT License - see LICENSE file for details.