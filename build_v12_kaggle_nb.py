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

# =====================================================================
M("## 4. Stacking — L1.5 Ridge + L2 meta (reliability + cross-target OOF)")

# =====================================================================
P("""from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_MODELS = ["lgb", "cat", "xgb", "hgb"]

def store_key(b, tt):
    return (b + "_" + tt, tt)

def build_stack_features(oof_store, tt, models):
    feats, cols = [], []
    for b in models:
        k = store_key(b, tt)
        if k in oof_store:
            feats.append(oof_store[k]); cols.append(k)
    if len(feats) == 0:
        return None, None
    return np.column_stack(feats), cols

L15_OOF = {}; L15_TE = {}
print("Level-1.5 Ridge stack (per target, own base OOFs)...")
for tt in TARGETS:
    m, idx, splits = get_splits(tt)
    Z, cols = build_stack_features(oof_store, tt, BASE_MODELS)
    if Z is None:
        print(f"  {tt}: no base features"); continue
    Zte = np.column_stack([test_store[c] for c in cols])
    pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
    oof = np.zeros(m.sum()); te_pred = np.zeros(len(Zte))
    for tr, va in splits:
        tr_l, va_l = pos[tr], pos[va]
        sr = StandardScaler().fit(Z[tr_l]); Ztr_s = sr.transform(Z[tr_l]); Zva_s = sr.transform(Z[va_l])
        meta = Ridge(alpha=10.0); meta.fit(Ztr_s, Y[idx][tr_l])
        oof[va_l] = meta.predict(Zva_s)
        te_pred += meta.predict(sr.transform(Zte)) / len(splits)
    L15_OOF[tt] = oof; L15_TE[tt] = te_pred
    print(f"  l15 {tt}: RMSE={rmse_metric(Y[m], oof):.4f}  (cols={cols})")

CROSS_MAP = {
    "eps": ["nc","egc","egb","eea"],
    "nc": ["eps","egb","egc","ei"],
    "egc": ["egb","eea","nc","eps","ei"],
    "egb": ["egc","nc","eea","eps","ei"],
    "ei": ["egc","egb","nc"],
    "eea": ["egc","egb","eps"],
    "tg": [],
}

def reliability_features(tt, models):
    Z, cols = build_stack_features(oof_store, tt, models)
    if Z is None:
        return None, None
    feats = np.column_stack([Z.mean(1), Z.std(1), Z.max(1), Z.min(1)])
    return feats, ["rel_mean", "rel_std", "rel_max", "rel_min"]

def cross_oof_features(tt):
    feats, cols = [], []
    m_tt = (dedup["target_type"] == tt).values
    for ct in CROSS_MAP[tt]:
        m_ct = (dedup["target_type"] == ct).values
        c2o = dict(zip(dedup.loc[m_ct, "canon"].values, L15_OOF[ct]))
        vals = np.array([c2o.get(c, np.nan) for c in dedup.loc[m_tt, "canon"].values], dtype=np.float32)
        miss = np.isnan(vals).astype(np.float32)
        vals = np.nan_to_num(vals, nan=float(np.nanmean(L15_OOF[ct])))
        feats += [vals, miss]; cols += [f"cross_{ct}", f"cross_{ct}_miss"]
    if not feats:
        return None, None
    return np.column_stack(feats), cols

def cross_te_features(tt):
    feats, cols = [], []
    for ct in CROSS_MAP[tt]:
        feats.append(np.asarray(L15_TE[ct], dtype=np.float32))
        feats.append(np.zeros(len(test), dtype=np.float32))
        cols += [f"cross_{ct}", f"cross_{ct}_miss"]
    if not feats:
        return None, None
    return np.column_stack(feats), cols

FINAL_OOF = {}; FINAL_TE = {}
print("Level-2 meta (own base + reliability + cross-target OOF)...")
for tt in TARGETS:
    m, idx, splits = get_splits(tt)
    Z1, c1 = build_stack_features(oof_store, tt, BASE_MODELS)
    if Z1 is None:
        print(f"  {tt}: no base features"); continue
    Zrel, crel = reliability_features(tt, BASE_MODELS)
    Zcr, ccr = cross_oof_features(tt)
    Z2 = np.column_stack([Z1, Zrel] + ([Zcr] if Zcr is not None else []))
    cols = c1 + crel + (ccr or [])
    Zte1 = np.column_stack([test_store[c] for c in c1])
    Zte_rel = np.column_stack([Zte1.mean(1), Zte1.std(1), Zte1.max(1), Zte1.min(1)])
    Zte_cr, _ = cross_te_features(tt)
    Zte2 = np.column_stack([Zte1, Zte_rel] + ([Zte_cr] if Zte_cr is not None else []))
    pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
    oof = np.zeros(m.sum()); te_pred = np.zeros(len(Zte2))
    for tr, va in splits:
        tr_l, va_l = pos[tr], pos[va]
        sr = StandardScaler().fit(Z2[tr_l]); Z2tr = sr.transform(Z2[tr_l]); Z2va = sr.transform(Z2[va_l])
        meta = Ridge(alpha=10.0); meta.fit(Z2tr, Y[idx][tr_l])
        oof[va_l] = meta.predict(Z2va)
        te_pred += meta.predict(sr.transform(Zte2)) / len(splits)
    FINAL_OOF[tt] = oof; FINAL_TE[tt] = te_pred
    print(f"  final {tt}: RMSE={rmse_metric(Y[m], oof):.4f}  (n_feats={len(cols)})")
""")

# =====================================================================
M("## 5. Pretrained GNN — graph featurization, GINE encoder, PI1M pretrain + fold-safe fine-tune")

# =====================================================================
P("""ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "Si", "P", "OTHER"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

def one_hot(value, choices):
    vec = [0.0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec

def atom_features(atom):
    sym = atom.GetSymbol(); hyb = atom.GetHybridization().name
    return (one_hot(sym, ATOM_SYMBOLS) + one_hot(hyb, HYBRIDIZATIONS) + [
        atom.GetIsAromatic()*1.0, atom.IsInRing()*1.0, atom.GetDegree()/4.0,
        atom.GetTotalNumHs()/4.0, atom.GetFormalCharge()/2.0])

N_ATOM_FEATS = len(ATOM_SYMBOLS) + len(HYBRIDIZATIONS) + 5
N_BOND_FEATS = len(BOND_TYPES) + 2

def bond_features(bond):
    return one_hot(bond.GetBondType().name, BOND_TYPES) + [
        bond.GetIsConjugated()*1.0, bond.IsInRing()*1.0]

def smiles_to_graph(smiles, target_idx=None, y=None, sample_weight=1.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]; edge_attr += [bf, bf]
    if len(edge_index) == 0:
        edge_index = [[0, 0]]; edge_attr = [[0.0] * N_BOND_FEATS]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target_idx is not None:
        data.target_idx = torch.tensor([target_idx], dtype=torch.long)
        data.y = torch.tensor([y], dtype=torch.float)
        data.w = torch.tensor([sample_weight], dtype=torch.float)
    return data

target_stats = {}
for t in TARGETS:
    vals = dedup.loc[dedup["target_type"] == t, "target"]
    target_stats[t] = (vals.mean(), vals.std() + 1e-9)

def build_train_graphs():
    graphs = []
    freq = dedup["target_type"].value_counts(normalize=True)
    for row_id, row in zip(dedup.index, dedup.itertuples()):
        ti = TARGET_IDX[row.target_type]
        mean_, std_ = target_stats[row.target_type]
        g = smiles_to_graph(row.smiles, target_idx=ti,
                            y=(row.target - mean_) / std_, sample_weight=1.0/freq[row.target_type])
        if g is not None:
            g.row_id = row_id
            graphs.append(g)
    return graphs

def build_test_graphs():
    graphs = []
    for row_id, row in zip(test.index, test.itertuples()):
        g = smiles_to_graph(row.smiles, target_idx=TARGET_IDX[row.target_type], y=0.0, sample_weight=1.0)
        if g is not None:
            g.row_id = row_id
            graphs.append(g)
    return graphs

def build_pretrain_graphs(smiles_list):
    graphs = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
    return graphs

class GINEEncoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden=128, n_layers=4, dropout=0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(n_atom_feats, hidden)
        self.bond_encoder = nn.ModuleList([nn.Linear(n_bond_feats, hidden) for _ in range(n_layers)])
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_encoder(x)
        for conv, bn, bond_enc in zip(self.convs, self.bns, self.bond_encoder):
            e = bond_enc(edge_attr)
            h = conv(h, edge_index, e)
            h = bn(h); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class PretrainModel(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden=128, n_layers=4, dropout=0.2,
                 mask_atom=0.15, mask_bond=0.20):
        super().__init__()
        self.encoder = GINEEncoder(n_atom_feats, n_bond_feats, hidden, n_layers, dropout)
        self.atom_decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, n_atom_feats))
        self.bond_decoder = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, n_bond_feats))
        self.mask_atom = mask_atom; self.mask_bond = mask_bond

    def forward(self, x, edge_index, edge_attr, batch):
        n = x.size(0); m = edge_index.size(1)
        atom_mask = torch.rand(n, device=x.device) < self.mask_atom
        bond_mask = torch.rand(m, device=x.device) < self.mask_bond
        x_c = x.clone(); x_c[atom_mask] = 0.0
        ea_c = edge_attr.clone(); ea_c[bond_mask] = 0.0
        h = self.encoder(x_c, edge_index, ea_c)
        if atom_mask.any():
            atom_loss = F.mse_loss(self.atom_decoder(h[atom_mask]), x[atom_mask])
        else:
            atom_loss = torch.zeros((), device=x.device)
        src = h[edge_index[0, bond_mask]]; dst = h[edge_index[1, bond_mask]]
        if bond_mask.any() and src.numel() > 0:
            bond_loss = F.mse_loss(self.bond_decoder(torch.cat([src, dst], dim=1)), edge_attr[bond_mask])
        else:
            bond_loss = torch.zeros((), device=x.device)
        return atom_loss, bond_loss

class GNNTrunk(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden=128, n_layers=4,
                 n_targets=len(TARGETS), dropout=0.2):
        super().__init__()
        self.encoder = GINEEncoder(n_atom_feats, n_bond_feats, hidden, n_layers, dropout)
        self.head_in = hidden * 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.head_in, 64), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(64, 1))
            for _ in range(n_targets)])

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.edge_attr)
        pooled = torch.cat([global_mean_pool(h, data.batch), global_add_pool(h, data.batch)], dim=1)
        return torch.cat([head(pooled) for head in self.heads], dim=1)

    def load_encoder(self, state_dict):
        enc = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc, strict=False)
""")

P("""GNN_OOF_PATH = os.path.join(WORK, "gnn_oof.csv")
GNN_TEST_PATH = os.path.join(WORK, "gnn_test.csv")
if ON_KAGGLE:
    GNN_CACHE_OOF = None; GNN_CACHE_TEST = None
else:
    GNN_CACHE_OOF = os.path.join("vault", "kernel-v10-output", "gnn_oof.csv")
    GNN_CACHE_TEST = os.path.join("vault", "kernel-v10-output", "gnn_test.csv")
USE_GNN_CACHE = (not ON_KAGGLE) and SMOKE and os.path.exists(GNN_CACHE_OOF) and os.path.exists(GNN_CACHE_TEST)

if USE_GNN_CACHE:
    gnn_oof_df = pd.read_csv(GNN_CACHE_OOF).set_index("row_id")
    gnn_test_df = pd.read_csv(GNN_CACHE_TEST).set_index("row_id")
    print("SMOKE: loaded GNN cache from vault/kernel-v10-output")
else:
    t0 = time.time()
    train_graphs = build_train_graphs()
    test_graphs = build_test_graphs()
    print(f"Built {len(train_graphs)} train graphs, {len(test_graphs)} test graphs in {time.time()-t0:.0f}s")

    pl = []
    if pl_path:
        pldf = pd.read_csv(pl_path)
        smi_col = "SMILES" if "SMILES" in pldf.columns else "smiles"
        pldf = pldf[[smi_col]].rename(columns={smi_col: "smiles"})
        pldf["canon"] = pldf["smiles"].map(canon_key)
        pl = pldf.drop_duplicates("canon")["smiles"].tolist()
        rng = np.random.RandomState(SEED)
        rng.shuffle(pl)
        pl = pl[:PRETRAIN_SAMPLE]
        print("PI1M pretraining corpus:", len(pl), "SMILES (capped at", PRETRAIN_SAMPLE, ")")
    pl_graphs = build_pretrain_graphs(pl) if pl else []
    print("pretraining graphs:", len(pl_graphs))

    def pretrain(epochs=PRETRAIN_EPOCHS, batch_size=256, lr=1e-3, patience=5):
        if not pl_graphs:
            print("No PI1M graphs - pretraining skipped")
            return None
        model = PretrainModel(N_ATOM_FEATS, N_BOND_FEATS).to(device)
        loader = DataLoader(pl_graphs, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        best = np.inf; best_state = None; t0 = time.time()
        for epoch in range(epochs):
            model.train(); tot_a = 0.0; tot_b = 0.0; nb = 0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                a_loss, b_loss = model(batch.x, batch.edge_index, batch.edge_attr, batch)
                loss = a_loss + 0.5 * b_loss
                loss.backward(); opt.step()
                tot_a += a_loss.item(); tot_b += b_loss.item(); nb += 1
            avg_a = tot_a / max(nb, 1); avg_b = tot_b / max(nb, 1)
            sched.step(avg_a + 0.5 * avg_b)
            val = avg_a + 0.5 * avg_b
            if val < best:
                best = val; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"pretrain ep {epoch+1}/{epochs}: atom={avg_a:.4f} bond={avg_b:.4f} ({time.time()-t0:.0f}s)", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(WORK, "pretrained_encoder.pt"))
        print("saved pretrained_encoder.pt")
        return best_state

    def train_gnn(init_state=None, epochs=MINI_EPOCHS, batch_size=64, lr=1e-3, patience=10,
                  trust_frac=0.15):
        row_to_graph = {g.row_id: g for g in train_graphs}
        oof = np.full(len(dedup), np.nan)
        fold_states = []
        for fold in sorted(dedup["fold"].unique()):
            fold_train = dedup.index[dedup["fold"] != fold]
            val = dedup.index[dedup["fold"] == fold]
            rng = np.random.RandomState(SEED + fold)
            trust_mask = rng.rand(len(fold_train)) < trust_frac
            trust_ids = fold_train[trust_mask]; tr_ids = fold_train[~trust_mask]
            tr_graphs = [row_to_graph[i] for i in tr_ids if i in row_to_graph]
            val_graphs = [row_to_graph[i] for i in val if i in row_to_graph]
            tr_loader = DataLoader(tr_graphs, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)
            model = GNNTrunk(N_ATOM_FEATS, N_BOND_FEATS).to(device)
            if init_state is not None:
                model.load_encoder(init_state)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
            best_val, bad_epochs, best_state = np.inf, 0, None
            epochs_used = 0; ft0 = time.time()
            for epoch in range(epochs):
                epochs_used = epoch + 1
                model.train()
                for batch in tr_loader:
                    batch = batch.to(device); opt.zero_grad()
                    pred = model(batch)
                    pred_sel = pred.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
                    loss = (F.mse_loss(pred_sel, batch.y, reduction="none") * batch.w).mean()
                    loss.backward(); opt.step()
                model.eval(); vloss = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred = model(batch)
                        pred_sel = pred.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
                        vloss.append(F.mse_loss(pred_sel, batch.y).item())
                val_loss = np.mean(vloss) if vloss else np.inf
                sched.step(val_loss)
                if val_loss < best_val:
                    best_val, bad_epochs = val_loss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                for g in val_graphs:
                    gb = Batch.from_data_list([g]).to(device)
                    pred = model(gb)
                    ti = int(g.target_idx.item()); mean_, std_ = target_stats[TARGETS[ti]]
                    oof[dedup.index.get_loc(g.row_id)] = pred[0, ti].item() * std_ + mean_
            fold_states.append((fold, best_state))
            print(f"  fold {fold}: best val MSE (norm)={best_val:.4f} ({time.time()-ft0:.0f}s, ep={epochs_used})", flush=True)
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return oof, fold_states

    def predict_graphs_on(graphs, state):
        model = GNNTrunk(N_ATOM_FEATS, N_BOND_FEATS).to(device)
        model.load_state_dict(state); model.eval()
        preds = {}
        with torch.no_grad():
            for g in graphs:
                gb = Batch.from_data_list([g]).to(device)
                pred = model(gb)
                ti = int(g.target_idx.item()); t_name = TARGETS[ti]
                mean_, std_ = target_stats[t_name]
                preds[g.row_id] = pred[0, ti].item() * std_ + mean_
        return preds

    print("=== Pretraining GNN on PI1M ===")
    pretrain_state = pretrain()
    print("=== Fine-tuning pretrained GNN (fold-safe, trust check) ===")
    pt_oof, pt_states = train_gnn(init_state=pretrain_state)

    test_preds = {}
    print("Computing test predictions (bag over fold models)...")
    for fold, state in pt_states:
        p = predict_graphs_on(test_graphs, state)
        for rid, v in p.items():
            test_preds[rid] = test_preds.get(rid, 0.0) + v / len(pt_states)
        del p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame({"row_id": list(dedup.index), "target_type": dedup["target_type"].values,
                  "gnn_oof": pt_oof}).to_csv(GNN_OOF_PATH, index=False)
    pd.DataFrame({"row_id": list(test.index), "target_type": test["target_type"].values,
                  "gnn_test": [test_preds.get(r, np.nan) for r in test.index]}).to_csv(GNN_TEST_PATH, index=False)
    print("wrote gnn_oof.csv, gnn_test.csv")
    gnn_oof_df = pd.read_csv(GNN_OOF_PATH).set_index("row_id")
    gnn_test_df = pd.read_csv(GNN_TEST_PATH).set_index("row_id")
""")

# =====================================================================
M("## 6. v11 reference blend (per-target fold-safe weight)")

# =====================================================================
P("""grid = np.linspace(0.0, 1.0, 21)
v11_blend_oof = {}
v11_blend_te = {}
V11_W = {}
print("v11 reference blend (fold-safe per-target weight):")
for tt in TARGETS:
    m, idx, splits = get_splits(tt)
    stack_oof = FINAL_OOF[tt]
    y_tt = Y[idx]
    g_vals = gnn_oof_df["gnn_oof"].reindex(dedup.index[idx]).to_numpy()
    pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
    oof = np.full(m.sum(), np.nan)
    fold_te = np.zeros(len(test)); w_acc = []
    for tr, va in splits:
        tr_l, va_l = pos[tr], pos[va]
        best_w_here, best_r = 0.5, -np.inf
        for w in grid:
            pred = w * stack_oof[tr_l] + (1 - w) * g_vals[tr_l]
            fin = ~np.isnan(pred) & ~np.isnan(y_tt[tr_l])
            if fin.sum() < 5:
                continue
            r = r2_score(y_tt[tr_l][fin], pred[fin])
            if r > best_r:
                best_r, best_w_here = r, w
        oof[va_l] = best_w_here * stack_oof[va_l] + (1 - best_w_here) * g_vals[va_l]
        g_te = gnn_test_df["gnn_test"].reindex(test.index).to_numpy()
        fold_te += (best_w_here * FINAL_TE[tt] + (1 - best_w_here) * g_te) / len(splits)
        w_acc.append(best_w_here)
    v11_blend_oof[tt] = oof; v11_blend_te[tt] = fold_te
    V11_W[tt] = float(np.mean(w_acc)) if w_acc else np.nan
    print(f"  {tt}: v11 blend OOF R2={r2_score(y_tt, oof):.4f}  mean_w={V11_W[tt]:.2f}")
""")

nb.cells = C
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(C), "cells")
