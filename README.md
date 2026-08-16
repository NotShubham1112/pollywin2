# PolyWin R2 — AISEHack 2.0 Polymer Property Prediction (Round 2)

> **Team Cosmic** · **Shubham N Kambli** · AISEHack 2.0 Polymer Property Prediction: Round 2 (`ppp-round-2` on Kaggle)
> Final submission: **P14 — public LB 0.883** (frozen production).

A disciplined, evidence-driven campaign to predict seven polymer properties (`eea, egb, egc, ei, eps, nc, tg`) from SMILES. Engineered as a per-target **GBM + Graph Neural Network Ridge blend** over a 5-fold GroupKFold on canonical SMILES, with a self-supervised MT-GNN encoder pretrained on the full ~995k-molecule PI1M archive. Branched into fifteen pre-registered experiments; two hypotheses accepted, the rest cleanly falsified.

> **⛳ Canonical submission — read this first.** The **only** method submitted in Round 2 is **P14** (public LB **0.883**, submission `55346358`). Its canonical repo artifact is **`notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb`**, generated from source-of-truth builder **`src/notebook_builders/build_v14_kaggle_nb.py`**. Everything else under `notebooks/` (v9–v22, sibling/BERT arms, failed experiments) is the **research record only — none of it is the submitted method**.

---

## 🏁 Final Submission

| Field | Value |
|---|---|
| **Competition** | [AISEHack 2.0 — Polymer Property Prediction: Round 2](https://www.kaggle.com/competitions/ppp-round-2) |
| **Metric** | Unweighted mean R² over 7 targets |
| **Primary notebook** | [PolyWin R2 v14 — P1M Pretrain (P14)](https://www.kaggle.com/code/shubhamkambli11/polywin-r2-v14-p1m-pretrain) |
| **Canonical repo artifact** | `notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb` |
| **Additional model notebook** | [PolyWin R2 v21 — Sibling Arm](https://www.kaggle.com/code/shubhamkambli11/polywin-r2-v21-sibling-arm) (research only — not submitted) |
| **Public LB** | **0.883** (submission `55346358`) |
| **Code repository** | [github.com/NotShubham1112/pollywin2](https://github.com/NotShubham1112/pollywin2) |
| **Team name** | Cosmic |
| **Participant** | Shubham N Kambli |
| **Final state** | P14 frozen as production; no further submission slots spent after v16 FAIL. |

---

## 📑 Table of Contents

1. [Problem & Data](#-problem--data)
2. [Approach](#-approach)
3. [Repository Layout](#-repository-layout)
4. [Pipeline at a Glance](#-pipeline-at-a-glance)
5. [Experiment Log (LB Progression)](#-experiment-log-lb-progression)
6. [Key Findings](#-key-findings)
7. [Reproducing P14](#-reproducing-p14)
8. [Tech Stack](#-tech-stack)
9. [What Didnt Work](#-what-didnt-work)
10. [Citation & License](#-citation--license)

---

## 🧪 Problem & Data

| Target | Description | Train rows | Unit |
|---|---|---:|---|
| `eea` | Electron Affinity | 368 | eV |
| `egb` | Bandgap (bulk) | 561 | eV |
| `egc` | Bandgap (chain) | 3,380 | eV |
| `ei`  | Ionization Energy | 370 | eV |
| `eps` | Dielectric Constant | 382 | — |
| `nc`  | Refractive Index | 382 | — |
| `tg`  | Glass Transition Temperature | varies | °C |

- **Train:** 7,409 rows · **Test:** 4,940 rows.
- **Competition-provided auxiliary corpus:** `PI1M.csv` (~995k unlabeled polymers) — the exact file the competition serves at `/kaggle/input/ppp-round-2/PI1M.csv`; used **only for self-supervised pretraining**, never as labels.
  - **Provenance:** shipped in this repo at `competition/data/raw/PI1M.csv` (47.5 MB, git-tracked). It is the exact file the competition serves at `/kaggle/input/ppp-round-2/PI1M.csv`; if you pull a fresh Kaggle copy, drop it at the same path to reproduce the pretrain step.
- **Public LB** is a 37% slice of test; **Private LB** is the remaining 63% and decides final standing.

---

## 🧠 Approach

A two-arm **per-target Ridge blend** of gradient-boosted descriptors and a multi-task graph neural network, with fold-safe OOF packing. The blend beat either arm alone on **all 7 targets**.

```
PI1M (995k) ──► self-supervised MT-GNN pretrain (GINE, 10 epochs)
                               │
train.csv ──► RDKit + Morgan   │
                  │             │
                  ▼             ▼
              GBM trio ◄───► MT-GNN (5-seed bag)
              (LGB/Cat/       │
               XGB/HGB)       │
                  │             │
                  └────► Per-target Ridge
                              │ alpha tuned per target
                              ▼
                       submission.csv
```

**Core design choices**

- **GroupKFold on canonical SMILES** so all 7 target rows of one polymer land in the same fold — no sibling leakage.
- **Per-target Ridge blend** with α tuned on OOF; never a single global ridge.
- **Self-supervised GINE pretraining** on full PI1M (~995k molecules, 10 epochs) for stronger shared representations.
- **5-seed bagging** (42 / 999 / 2025) for the MT-GNN to stabilize variance.
- **No external test labels**, no Kaggle-internal data, no public-API leaks.

---

## 📁 Repository Layout

```
ploywin-r2/
├── README.md                          # this file
├── requirements.txt                   # pinned local env (torch==2.5.1, matches Kaggle kernel)
├── competition/
│   ├── data/raw/                      # train.csv, test.csv, PI1M.csv (cached)
│   └── rules/                         # official Kaggle rules + AISEHack eligibility clause
├── notebooks/
│   ├── exploration/                   # AISEHack_Round2_Pipeline.ipynb
│   ├── v9_gnn/                        # GNN arm exploration (v9)
│   ├── v10_pretrain/                  # PI1M pretrain experiments (v10)
│   ├── v11_reproduce/                 # v11 reproduction (compliant floor)
│   ├── v12_bucket_moe/                # chemistry bucket-MoE (v12)
│   ├── v13_blend/                     # GBM + MT-GNN blend (v13)
│   ├── v13_specialist/                # small-five multi-task specialist (v13)
│   ├── v14_p14_baseline/              # P14 full-PI1M pretrain (FINAL)
│   ├── v15_epsnc_focus/               # rejected hypothesis
│   ├── v16_cross_target/              # cross-target decoder (gate PASS, LB FAIL)
│   ├── v21_sibling_arm/               # sibling Ridge arm
│   └── v22_bert_arm/                  # self-trained BERT/RoBERTa SMILES encoder
├── scripts/
│   └── legacy/                        # phase-1 experiment scripts (exp2/3/4, r2_phase1, reorg)
├── src/
│   ├── core/                          # mt_gnn_v2.py, decoder_v16.py
│   ├── experiments/                   # per-experiment scripts
│   ├── notebook_builders/             # *.py -> .ipynb generators
│   ├── pipeline/                      # smoke / patch / verify helpers
│   ├── v20_smiles_encoder/            # RoBERTa-style SMILES encoder arm
│   └── v22_bert_arm/                  # BERT arm + gates + smoke tests
├── tests/                             # pytest harness for v20 / v22 gates
├── vault/
│   ├── pipeline_out*/                 # cached OOF predictions, gate reports
│   └── polywinr2/                     # Obsidian lab notebooks (research MOC)
├── docs/
│   ├── lab-postmortem-2026-08-08.md   # full post-mortem of the campaign
│   └── session.md                     # working session journal
└── hackathon_rules/                   # local copy of rules
```

---

## ⚙️ Pipeline at a Glance

1. **Pretrain** GINE encoder on the deduplicated PI1M corpus (~995k polymers, 10 epochs). Saved artifact: `mt_gnn_v2_pretrained.pt`.
2. **Engineer features** per fold: RDKit descriptors (~22), Morgan / MACCS fingerprints (~2,215), atom-pair deltas.
3. **Train two arms per fold** — GBM trio (LGBM / CatBoost / XGBoost / HistGBM) and the pretrained MT-GNN bagged over 5 seeds.
4. **OOF-pack** predictions strictly inside each fold (no test label leakage, no group leakage).
5. **Per-target Ridge blend** over `[GBM, MT-GNN]` with α tuned on OOF.
6. **Sanity gates** — pre-registered: small-five weighted gain ≥ +0.003, no target regresses > −0.003, all arms fold-safe.
7. **Export** `submission.csv` (4,940 rows, 0 NaN, physics bounds enforced: egc/egb/ei ≥ 0, eps ≥ 1, nc ∈ [1, 3]).

---

## 📈 Experiment Log (LB Progression)

| Wave | Config | Public LB | Δ vs parent | Verdict | Submission |
|---|---|---:|---:|---|---|
| v4 Baseline | GBM trio + FFN, RDKit feats | 0.828 | — | start | `55194181` |
| v6 Honest OOF Stack | leak-safe meta-stack, no NNs | 0.847 | +0.019 | accepted → superseded | `55216423` |
| v7 Retrieval | retrieval `FULL` features | *not submitted* | — | FAIL (OOF) | — |
| v8 PI1M Pseudo-Label | PI1M pseudo-labels from OOF | 0.828 | −0.019 | FAIL → dropped | `55246041` |
| v10 Pretrained GNN | self-supervised MT-GNN (GINE) | 0.830 | — | superseded | `55246047` |
| v11 Fold-Safe Blend | fold-safe per-target weights | 0.852 | +0.022 | best standing | `55286407` |
| v12 Chemistry Bucket-MoE | KMeans chemistry routing + per-cluster GBMs | 0.849 | −0.003 | FAIL — OOF did not transfer | `55305403` |
| **v13 GBM + MT-GNN Blend** | **4-arm Ridge, 5-seed MT-GNN** | **0.877** | **+0.025** | ✅ **Hypothesis 1** | **`55342412`** |
| **P14 Full-PI1M Pretrain** | **+ 995k PI1M pretrain (10 ep), Ridge [GBM, MT]** | **0.883** | **+0.006** | ✅ **Hypothesis 2 — FINAL** | **`55346358`** |
| v15 EPS/NC Focus | EPS/NC ×2 loss weight | OOF −0.0051 | — | ❌ Hypothesis 3, never submitted | — |
| v16 Cross-Target Decoder | physics + learned decoder arms | 0.874 | −0.009 | FAIL — gate passed, LB regressed | `55289004` |
| v17 Sib-Ridge + physics eps | sibling-target Ridge arm | 0.862 | −0.021 | FAIL — dropped | — |
| v18 Hard-gate sib(halved)+phys | half-weight sibling gate | 0.864 | −0.019 | FAIL — dropped | — |
| v19 CT-PGCN | sib+phys blend (α ≤ 0.30) | 0.874 | −0.009 | FAIL — OOF leakage artifact | — |
| v20 Self-Trained SMILES Encoder | RoBERTa-style masked SMILES encoder | *not submitted* | — | FAIL — local gate (OOF −0.0004) | — |
| v22 BERT Arm | fold-safe BERT/RoBERTa arm | in-progress | — | gated, sandbox | — |

**Best public LB: 0.883 (P14).** Submission floor improved 0.828 → 0.883 over the campaign (**+0.055 net**).

---

## 🔑 Key Findings

1. **Retrieval adds little once a strong GNN exists.** v7 (retrieval-augmented GBM) was worse than baseline on all 7 targets. Once the encoder captured molecular similarity, hand-built retrieval features were redundant.
2. **GBM + MT-GNN blending works** *(Hypothesis 1, accepted)*. The 4-arm Ridge blend gained +0.025 over either arm alone. The gains were additive, not a fold fluke.
3. **Full-scale pretraining works** *(Hypothesis 2, accepted)*. Scaling the pretrain corpus from 20k to the full 995k PI1M archive gained +0.006 (0.877 → 0.883). Bigger, better shared representations transferred directly downstream.
4. **Correlation reduction was NOT the mechanism.** The OOF correlation between GBM and GNN actually rose (0.9513 → 0.9552), yet the blend still gained. The improvement came from a stronger encoder, not from decorrelation.
5. **EPS/NC carried most of P14s gain.** eps +0.026, nc +0.024 out of the +0.006 total — the two weakest targets were the highest-leverage rows.
6. **Loss reweighting for EPS/NC failed** *(Hypothesis 3, rejected)*. Doubling the sample weight for eps/nc made both targets *worse* (−0.0053, −0.0234). The gradient overfit the tiny eps/nc folds while stealing signal from the dominant targets.
7. **Cross-target / sibling blending OOF gains never transferred to the LB.** Every sib-family submission (v16, v17, v18, v19) passed the offline gate but regressed on the public LB. Root cause: the sibling pivot is built from full train and used unchanged inside GroupKFold val folds → a sibling-leakage artifact that does not survive to the public test. Pre-registration prevented further slot-spending.

---

## 🚀 Reproducing P14

The canonical production notebook is **`notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb`** (also published on Kaggle as the primary submission).

**Local reproduction** (smoke-tested where possible):

```bash
# 0. Create the pinned environment (Python 3.10+; torch==2.5.1 matches every submitted kernel)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 1. (Optional) regenerate the OOF features
python src/pipeline/make_smoke.py

# 2. Pretrain the GINE encoder on PI1M
python src/core/mt_gnn_v2.py --pretrain --epochs 10 --data competition/data/raw/PI1M.csv

# 3. Build the Kaggle notebook from the source-of-truth builder
python src/notebook_builders/build_v14_kaggle_nb.py

# 4. Run the executable notebook
jupyter nbconvert --to notebook --execute notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb
```

**Kaggle reproduction** (recommended):

1. Open the published notebook: [PolyWin R2 v14 — P1M Pretrain](https://www.kaggle.com/code/shubhamkambli11/polywin-r2-v14-p1m-pretrain).
2. Fork → "Save & Run All" (GPU recommended for the GNN arm).
3. Submit the produced `submission.csv`.

---

## 🛠️ Tech Stack

- **Models:** LightGBM, CatBoost, XGBoost, HistGradientBoosting, PyTorch Geometric (GINE), custom RoBERTa-style SMILES encoder
- **Featurization:** RDKit (descriptors, Morgan / MACCS / AtomPair fingerprints), PyTorch
- **Validation:** scikit-learn GroupKFold (groups = canonical SMILES), per-target Ridge blend
- **Notebook build:** programmatic `.py -> .ipynb` via `nbformat` (each `build_*_kaggle_nb.py` is the source of truth)
- **Smoke testing:** `pytest` for v20 / v22 gates, `nbclient` for in-kernel smoke execution
- **Visualization / reporting:** matplotlib, pandas, custom gate reports (`v*_gate_report.csv`)

---

## ❌ What Didnt Work

Honest accounting of the hypotheses that **failed** — and why the campaign is still a success.

| Branch | What was tried | Why it failed | Lesson |
|---|---|---|---|
| **v7** Retrieval augmentation | kNN columns from train neighbors | Worse than baseline on all 7 targets | Strong encoder subsumes retrieval |
| **v8** PI1M pseudo-labelling | reuse OOF preds as labels on PI1M | OOF +0.017 → LB −0.019 | Pseudo-labels amplify OOF noise |
| **v12** Chemistry Bucket-MoE | KMeans chemistry routing + per-cluster GBMs | OOF gain did not transfer | Small cluster → overfit |
| **v15** EPS/NC loss focus (×2 weight) | re-weight eps/nc rows | Both targets regressed (−0.005, −0.023) | Overfits tiny eps/nc folds |
| **v16** Cross-Target Decoder | physics + 6-arm sibling Ridge | Gate PASS, LB −0.009 | Sibling-leakage artifact in OOF |
| **v17 / v18 / v19** | sib-Ridge / half-gate / CT-PGCN | All FAIL at the LB | Same root cause as v16 |
| **v20** Self-trained SMILES encoder | RoBERTa-style masked SMILES encoder | OOF −0.0004 (gate fail) | 3rd arm near-collinear with existing |

The campaign success metric is **not "every idea worked"** — it is **"no submission slot was wasted on a hypothesis that did not clear the pre-registered gate."** Two confirmed wins, one clean lose, and four branches killed at the gate.

---

## 📜 Citation & License

If you reference this work, please cite:

> Shubham N Kambli (Team Cosmic). *PolyWin R2 — AISEHack 2.0 Polymer Property Prediction: Round 2 (P14, public LB 0.883).* 2026. https://github.com/NotShubham1112/pollywin2

**License:** MIT (code) · CC-BY-4.0 (documentation / lab postmortems).

---

<p align="center">
  <em>Disciplined experimentation beats clever architecture. Stop the experiment when the gate fails.</em>
</p>
