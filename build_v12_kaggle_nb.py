#!/usr/bin/env python
"""Build PolyWin_R2_v12_bucket_moe.ipynb - v12 Chemistry Bucket MoE (end-to-end Kaggle kernel).

One self-contained notebook: rebuild GBM stack + pretrain GNN on PI1M + chemistry
bucket MoE + submission. Run:  python build_v12_kaggle_nb.py
"""
import nbformat as nbf

OUT = "PolyWin_R2_v12_bucket_moe.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
P = lambda s: C.append(nbf.v4.new_code_cell(s))

# =====================================================================
M("""# PolyWin R2 — v12: Chemistry Bucket MoE (Kaggle kernel)

## Question
Does routing each molecule to a **per-chemistry-cluster** blend weight `w` beat the
v11 **per-target scalar** blend on honest OOF? Some chemistry families may prefer the
GBM stack, others the pretrained GNN; a per-bucket `w` should exploit that structure
with near-zero overfitting risk (fixed clustering + coarse weight grid, no trained gate).

## Protocol (honest)
* **Self-contained**: rebuilds the GBM stack (4 GBMs + L1.5 Ridge + L2 meta), pretrains
  the GINE encoder on the unlabeled PI1M archive, fine-tunes fold-safely, then blends.
* Buckets = **KMeans on 13 curated chemistry features** (K in {2,3,4}); weights tuned
  **fold-safely inside each cluster**; K chosen by the bucket blend's own fold-safe OOF.
* Test rows assign to the nearest cluster centroid and use the fold-averaged weight.
* **Fallback**: if bucket-MoE does not beat the v11 blend on mean OOF, the notebook
  still emits the v11 blend submission, so a valid submission always exists.
* Success criteria (pre-registered): Strong = beats v11 on >=4 targets OOF, no target
  regresses > -0.003; Moderate = 2-3 targets; Failure = <=1 target -> ship v11 blend.

Only OSI-approved libs: PyTorch, PyTorch Geometric, RDKit, scikit-learn, LightGBM, CatBoost, XGBoost.
""")

# =====================================================================
P("""import os, sys, time, gc, warnings, random
import subprocess, importlib.util

def ensure_pkg(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print("installing", pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--disable-pip-version-check", pkg])

for _p, _n in [("rdkit", "rdkit"), ("torch_geometric", "torch_geometric"),
               ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("xgboost", "xgboost")]:
    ensure_pkg(_p, _n)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
np.random.seed(42); random.seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import r2_score, root_mean_squared_error as rmse_metric
from sklearn.model_selection import GroupKFold

def get_torch_device():
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda"); t = t + 1; torch.cuda.synchronize(); del t
            return torch.device("cuda")
        except Exception as e:
            print("CUDA probe failed -> CPU:", str(e)[:120])
    return torch.device("cpu")

device = get_torch_device()
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

ON_KAGGLE = os.path.exists("/kaggle")
SMOKE = os.environ.get("POLYWIN_SMOKE", "0") == "1"
GLOBAL_FOLDS = 3 if SMOKE else 10
MINI_EPOCHS = 3 if SMOKE else int(os.environ.get("MINI_EPOCHS", "40"))
PRETRAIN_EPOCHS = 1 if SMOKE else int(os.environ.get("PRETRAIN_EPOCHS", "5"))
PRETRAIN_SAMPLE = 2000 if SMOKE else int(os.environ.get("PRETRAIN_SAMPLE", "20000"))
BUCKET_FIT_CAP = 3000 if SMOKE else None

if ON_KAGGLE:
    WORK = "/kaggle/working"; INP = "/kaggle/input"
else:
    WORK = os.path.join("vault", "pipeline_out_v12_smoke" if SMOKE else "pipeline_out_v12")
    INP = "official_dataset"
os.makedirs(WORK, exist_ok=True)
FIG = os.path.join(WORK, "figures"); os.makedirs(FIG, exist_ok=True)

def find_input(base, name):
    for p in [os.path.join(base, name), os.path.join(base, "ppp-round-2", name),
              os.path.join(base, "competitions", "ppp-round-2", name),
              os.path.join(base, "aisehack-2-0", name)]:
        if os.path.exists(p):
            return p
    return None

print("device:", device, "| SMOKE:", SMOKE, "| folds:", GLOBAL_FOLDS,
      "| MINI_EPOCHS:", MINI_EPOCHS, "| PRETRAIN_EPOCHS:", PRETRAIN_EPOCHS,
      "| PRETRAIN_SAMPLE:", PRETRAIN_SAMPLE, "| BUCKET_FIT_CAP:", BUCKET_FIT_CAP)
""")

# =====================================================================
M("## 1. Data — canonicalize, dedupe, GroupKFold folds")

# =====================================================================
P("""TARGETS = ["tg","egc","egb","eps","nc","ei","eea"]
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

def canon_key(smiles):
    return smiles.replace("*", "").replace("[*]", "")

def parse_mol(smiles):
    m = MolFromSmiles(smiles.replace("*", "[*]"))
    if m is None:
        m = MolFromSmiles(smiles.replace("*", "C"))
    return m

train_path = find_input(INP, "train.csv")
test_path  = find_input(INP, "test.csv")
pl_path    = find_input(INP, "PI1M.csv")
assert train_path and test_path, "train.csv / test.csv not found"
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

train["canon"] = train["smiles"].map(canon_key)
test["canon"] = test["smiles"].map(canon_key)
dedup = train.groupby(["canon", "target_type"], as_index=False)["target"].median()
_smi = train.drop_duplicates(["canon", "target_type"])[["canon", "target_type", "smiles"]]
dedup = dedup.merge(_smi, on=["canon", "target_type"], how="left")
print("rows before dedupe:", len(train), "-> after:", len(dedup))
print("unique polymers (train):", dedup["canon"].nunique())

FOLDS_CSV = os.path.join(WORK, "folds.csv")
if os.path.exists(FOLDS_CSV):
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy()
    assert len(folds) == len(dedup), f"folds.csv length mismatch: {len(folds)} != {len(dedup)}"
    GLOBAL_FOLDS = int(folds.max()) + 1
else:
    gkf = GroupKFold(n_splits=GLOBAL_FOLDS)
    folds = np.zeros(len(dedup), dtype=int)
    for i, (_, va) in enumerate(gkf.split(dedup, groups=dedup["canon"])):
        folds[va] = i
    pd.DataFrame({"canon": dedup["canon"].values, "target_type": dedup["target_type"].values,
                  "fold": folds}).to_csv(FOLDS_CSV, index=False)
dedup["fold"] = folds
print(dedup.groupby(["target_type","fold"]).size().unstack(fill_value=0).to_string())
""")

nb.cells = C
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(C), "cells")
