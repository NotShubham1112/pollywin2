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

# =====================================================================
M("## 2. Feature factory (v8 Layer 2 — RDKit descriptors + fingerprints + polymer-physics + fragments)")

# =====================================================================
P("""DESC_NAMES = [d[0] for d in Descriptors.descList]

def rdkit_desc(mol):
    try:
        return list(Descriptors.CalcMolDescriptors(mol).values())
    except Exception:
        return [np.nan] * len(DESC_NAMES)

_GEN_M2 = GetMorganGenerator(radius=2, fpSize=2048)
_GEN_M1 = GetMorganGenerator(radius=1, fpSize=1024)

def _fps(mol):
    m2 = np.array(_GEN_M2.GetFingerprint(mol), dtype=np.float32)
    m1 = np.array(_GEN_M1.GetFingerprint(mol), dtype=np.float32)
    mc = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    return np.concatenate([m1, m2, mc])

def polymer_physics(mol):
    if mol is None:
        return np.zeros(15)
    arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    na = mol.GetNumAtoms()
    nC = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="C")
    nS = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="S")
    nF = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="F")
    nSi = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="Si")
    nCl = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="Cl")
    nBr = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="Br")
    nN = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="N")
    nO = sum(1 for a in mol.GetAtoms() if a.GetSymbol()=="O")
    nHal = nF + nCl + nBr
    conj = arom + sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble()==2.0)
    return np.array([
        arom / max(heavy,1),              # aromatic ratio
        rings / max(heavy,1),             # ring density
        rings,                             # ring count
        1.0 - rot / max(heavy,1),          # rigidity score
        rot / max(heavy,1),                # flexibility score
        nHal / max(heavy,1),               # halogen density
        nS / max(heavy,1),                 # sulfur density
        nN / max(heavy,1),                 # nitrogen density
        nO / max(heavy,1),                 # oxygen density
        (nN + nO) / max(heavy,1),          # hetero density
        conj / max(heavy,1),               # conjugation score
        rdMolDescriptors.CalcNumHBD(mol)/max(heavy,1),   # H-bond donor density
        rdMolDescriptors.CalcNumHBA(mol)/max(heavy,1),   # H-bond acceptor density
        Descriptors.MolLogP(mol),          # logP
        Descriptors.MolMR(mol)/max(heavy,1)              # molar refractivity density
    ], dtype=np.float32)

POLY_NAMES = ["arom_ratio","ring_density","ring_count","rigidity","flexibility","halogen_density",
              "sulfur_density","nitrogen_density","oxygen_density","hetero_density",
              "conjugation","hbd_density","hba_density","logp","mr_density"]

FRAGMENTS = ["C(=O)O", "C(=O)N", "C(=O)NC(=O)", "C-O-C", "c1ccccc1", "c1csc", "F", "C#N",
             "S(=O)(=O)", "C=O", "C=C", "c1ccncc1", "N=C=O", "OC(=O)", "NC(=O)", "c1ccc2", "CC(C)C"]
FRAG_NAMES = ["ester","amide","imide","ether","benzene","thiophene","fluoro","nitrile",
              "sulfone","carbonyl","alkene","pyridine","isocyanate","carboxyl","amid_link","fused_ring","isopropyl"]
def fragment_vec(mol):
    if mol is None:
        return np.zeros(len(FRAGMENTS), dtype=np.float32)
    s = Chem.MolToSmiles(mol)
    return np.array([1.0 if f in s else 0.0 for f in FRAGMENTS], dtype=np.float32)

def build_features(smiles_list, canon_list=None):
    rows_d, rows_f, rows_p, rows_r = [], [], [], []
    ok = []
    for smi in smiles_list:
        m = parse_mol(smi)
        if m is None:
            rows_d.append(np.zeros(len(DESC_NAMES))); rows_f.append(np.zeros(1024+2048+167))
            rows_p.append(np.zeros(15)); rows_r.append(np.zeros(len(FRAGMENTS))); ok.append(False)
            continue
        rows_d.append(rdkit_desc(m)); rows_f.append(_fps(m))
        rows_p.append(polymer_physics(m)); rows_r.append(fragment_vec(m)); ok.append(True)
    D = pd.DataFrame(np.array(rows_d, dtype=np.float64), columns=DESC_NAMES)
    F = pd.DataFrame(np.array(rows_f, dtype=np.float32),
                     columns=[f"fp_{i}" for i in range(np.array(rows_f).shape[1])])
    P_ = pd.DataFrame(np.array(rows_p, dtype=np.float32), columns=POLY_NAMES)
    R_ = pd.DataFrame(np.array(rows_r, dtype=np.float32), columns=[f"frag_{n}" for n in FRAG_NAMES])
    X = pd.concat([D, F, P_, R_], axis=1)
    return X, np.array(ok, dtype=bool)

print("Building features on train...")
t0 = time.time()
Xtr, ok_tr = build_features(dedup["smiles"].tolist())
print(f"train features {Xtr.shape} in {time.time()-t0:.0f}s, parse-ok {ok_tr.mean():.1%}")

print("Building features on test...")
t0 = time.time()
Xte, ok_te = build_features(test["smiles"].tolist())
print(f"test features {Xte.shape} in {time.time()-t0:.0f}s, parse-ok {ok_te.mean():.1%}")
""")

P("""# ---- cleaning: winsorize, drop constant, impute median ----
X_all = pd.concat([Xtr, Xte], axis=0).reset_index(drop=True)
X_all = X_all.replace([np.inf, -np.inf], np.nan)
const_cols = [c for c in X_all.columns if X_all[c].nunique() <= 1]
X_all = X_all.drop(columns=const_cols)
for c in X_all.columns:
    lo, hi = X_all[c].quantile(0.001), X_all[c].quantile(0.999)
    X_all[c] = X_all[c].clip(lo, hi)
med = X_all.median()
X_all = X_all.fillna(med).replace([np.inf, -np.inf], 0.0)
Xtr = X_all.iloc[:len(dedup)].reset_index(drop=True)
Xte = X_all.iloc[len(dedup):].reset_index(drop=True)
print("after cleaning:", Xtr.shape, Xte.shape, "| dropped const cols:", len(const_cols))
Xtr.to_pickle(os.path.join(WORK, "Xtr.pkl"))
Xte.to_pickle(os.path.join(WORK, "Xte.pkl"))
dedup.to_pickle(os.path.join(WORK, "dedup.pkl"))
test.to_pickle(os.path.join(WORK, "test.pkl"))
""")

# =====================================================================
M("## 3. Validation harness (GroupKFold per target, OOF RMSE) + Layer 4 GBM experts")

# =====================================================================
P("""Y = dedup["target"].values
oof_store = {}      # (model, target) -> oof preds
test_store = {}     # (model, target) -> test preds

def get_splits(tt):
    m = (dedup["target_type"] == tt).values
    idx = np.where(m)[0]
    splits = []
    for f in range(folds.max() + 1):
        fold_mask = (folds[m] == f)
        va = idx[fold_mask]
        tr = idx[~fold_mask]
        if len(va) > 0 and len(tr) > 0:
            splits.append((tr, va))
    return m, idx, splits

def record(name, tt, oof, te_pred):
    oof_store[(name, tt)] = oof
    test_store[(name, tt)] = te_pred
    return rmse_metric(Y[dedup["target_type"].values == tt], oof)

MODEL_COLS = list(Xtr.columns)
print("v12 stack uses all feature cols:", len(MODEL_COLS))
""")

P("""import lightgbm as lgbm
import xgboost as xgbm
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def gbm_fit_predict(tt, make_model, Xtr_full, Xte_full, use_folds=True):
    m, idx, splits = get_splits(tt)
    oof = np.zeros(m.sum())
    te_pred = np.zeros(len(Xte_full))
    feats = list(Xtr_full.columns)
    for tr, va in splits:
        mdl = make_model()
        mdl.fit(Xtr_full.iloc[tr], Y[tr])
        oof[np.where(m)[0].searchsorted(va)] = mdl.predict(Xtr_full.iloc[va])
        te_pred += mdl.predict(Xte_full) / len(splits)
    return oof, te_pred

def make_lgb():
    return lgbm.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
                              reg_alpha=0.3, reg_lambda=1.0, min_child_samples=10,
                              random_state=42, verbose=-1, n_jobs=-1)
def make_cat():
    return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                             random_seed=42, verbose=0, allow_writing_files=False)
def make_xgb():
    return xgbm.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                             subsample=0.85, colsample_bytree=0.7, reg_alpha=0.3,
                             reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=-1)
def make_hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42,
                                         l2_regularization=1.0)

GBM_CHK = os.path.join(WORK, "moe_gbm_chk.parquet")
if ON_KAGGLE:
    GBM_CACHE_CHK = None
else:
    GBM_CACHE_CHK = os.path.join("vault", "pipeline_out", "gnn_arm", "moe_gbm_chk.parquet")
USE_GBM_CACHE = (not ON_KAGGLE) and SMOKE and os.path.exists(GBM_CACHE_CHK)

LEADERBOARD = {}
model_oof = {n: {} for n in ("lgb", "cat", "xgb", "hgb")}
model_te = {n: {} for n in ("lgb", "cat", "xgb", "hgb")}
if USE_GBM_CACHE:
    chk = pd.read_parquet(GBM_CACHE_CHK)
    for _, r in chk.iterrows():
        oof_store[(r["key"], r["target"])] = r["oof"]
        test_store[(r["key"], r["target"])] = r["test_pred"]
    print(f"SMOKE: loaded {len(chk)} GBM checkpoints from {GBM_CACHE_CHK}")
else:
    print("Training GBM experts (v12 stack)...")
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        leader = {}
        for name, mk in [("lgb", make_lgb), ("cat", make_cat), ("xgb", make_xgb), ("hgb", make_hgb)]:
            t0 = time.time()
            oof, tep = gbm_fit_predict(tt, mk, Xtr[MODEL_COLS], Xte[MODEL_COLS])
            r = record(name + "_" + tt, tt, oof, tep)
            leader[name] = r
            model_oof[name][tt] = oof; model_te[name][tt] = tep
            print(f"  {tt} {name}: RMSE={r:.4f} ({time.time()-t0:.0f}s)")
        LEADERBOARD[tt] = leader
    pd.DataFrame(LEADERBOARD).round(4).to_csv(os.path.join(WORK, "leaderboard_gbm.csv"))
    chk_rows = [pd.DataFrame({"key": k[0], "target": k[1], "oof": [oof_store[k]],
                              "test_pred": [test_store[k]]}) for k in oof_store]
    pd.concat(chk_rows, ignore_index=True).to_parquet(GBM_CHK, index=False)
    print("saved moe_gbm_chk.parquet")
""")

nb.cells = C
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(C), "cells")
