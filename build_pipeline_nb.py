#!/usr/bin/env python
"""Build AISEHack_Round2_Pipeline.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
P = lambda s: C.append(nbf.v4.new_code_cell(s))

# =====================================================================
M("""# AISEHack 2.0 — Round 2 Polymer Property Prediction Pipeline

**Goal:** predict 7 polymer properties (Tg, Egc, Egb, Ei, Eea, EPS, Nc) from SMILES.
**Metric:** per-target RMSE on the hidden test set (see `baseline_model.ipynb`).

## Architecture (9 layers)
1. **Canonicalization** — normalize SMILES, dedupe, group key for CV
2. **Feature Factory** — RDKit descriptors, Morgan/MACCS/AtomPair/Topological fingerprints, polymer-physics features, fragment vocabulary
3. **Retrieval Memory** — fold-safe kNN features from 3 neighbour pools (global / same-target / cross-target priors)
4. **Target-aware experts** — LightGBM / CatBoost / XGBoost / HistGB per expert group
5. **Electronic Foundation Network** — archived behind `RUN_NNS` (default OFF; not executed in v7)
6. **Tg isolation** — dedicated single-target Tg NN, archived behind `RUN_NNS` (default OFF)
7. **Stacking** — Ridge level-1.5 + level-2 meta (reliability + cross-target features) on OOF predictions
8. **PI1M pseudo-labelling** — deferred to v8 (code present, `USE_PSEUDO=False`)
9. **Submission + judge diagrams** — `submission.csv` + matplotlib figures

**v7 experiment:** three-arm ablation (BASE / FULL / RETR-only) per target with LGB; submitted config = FULL (4 GBMs + L1.5 Ridge + L2 meta).

## Rule compliance notes
- **No hand-labelling of test data.** All retrieval features use **train labels only**.
- kNN retrieval is **fold-safe** in CV (neighbours drawn only from the training folds).
- PI1M is **explicitly allowed** by the rules ("may be used for implementing advanced algorithms").
- Only **OSI-approved open-source libraries** (RDKit, scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch).
""")

P("""import os, sys, gc, time, json, warnings, random
import subprocess, importlib.util
def ensure_pkg(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print("installing", pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", pkg])
for _p, _n in [("rdkit", "rdkit"), ("catboost", "catboost"),
               ("lightgbm", "lightgbm"), ("xgboost", "xgboost")]:
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

def get_torch_device():
    if torch.cuda.is_available():
        try:
            _t = torch.zeros(1, device="cuda"); _t = _t + 1
            torch.cuda.synchronize(); del _t
            return torch.device("cuda")
        except Exception as _e:
            print("CUDA probe failed -> using CPU:", str(_e)[:120])
    return torch.device("cpu")

ON_KAGGLE = os.path.exists("/kaggle")
if ON_KAGGLE:
    WORK = "/kaggle/working"
else:
    WORK = os.path.join("vault", "pipeline_out")
SMOKE = os.environ.get("POLYWIN_SMOKE", "0") == "1"
if SMOKE and not ON_KAGGLE:
    WORK = os.path.join("vault", "pipeline_out_smoke")
os.makedirs(WORK, exist_ok=True)
FIG = os.path.join(WORK, "figures"); os.makedirs(FIG, exist_ok=True)
GLOBAL_FOLDS = 5 if SMOKE else 10
EFN_EPOCHS = 15 if SMOKE else 40
TGNN_EPOCHS = 15 if SMOKE else 40
RUN_NNS = False   # v7: EFN/TgNN code archived in notebook, not executed. Flip True to re-enable.
print("ON_KAGGLE =", ON_KAGGLE)
print("WORK =", WORK)
print("SMOKE =", SMOKE, "| GLOBAL_FOLDS =", GLOBAL_FOLDS)
print("torch =", torch.__version__, "| cuda =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))""")

P("""# ---- data path detection ----
def find_input(base, name):
    for p in [os.path.join(base, name), os.path.join(base, "ppp-round-2", name),
              os.path.join(base, "competitions", "ppp-round-2", name),
              os.path.join(base, "aisehack-2-0", name)]:
        if os.path.exists(p):
            return p
    return None

if ON_KAGGLE:
    INP = "/kaggle/input"
    train_path = find_input(INP, "train.csv")
    test_path  = find_input(INP, "test.csv")
    pi1m_path  = find_input(INP, "PI1M.csv")
else:
    INP = "official_dataset"
    train_path = "official_dataset/train.csv"
    test_path  = "official_dataset/test.csv"
    pi1m_path  = "official_dataset/PI1M.csv"

assert train_path and os.path.exists(train_path), "train.csv not found"
assert test_path and os.path.exists(test_path), "test.csv not found"
print("train:", train_path, os.path.getsize(train_path) if os.path.exists(train_path) else "-")
print("test :", test_path)
print("PI1M :", pi1m_path, os.path.exists(pi1m_path))

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print("train", train.shape, "| test", test.shape)
print(train["target_type"].value_counts().to_string())""")

M("""## Layer 1 — Canonicalization engine

- Parse polymer SMILES (the `*` dummy atom marks chain attachment).
- Canonical key = SMILES minus `*`/`[*]` -> used as the **group key** for GroupKFold.
- Drop fully-duplicated rows; keep label conflicts (3 rows) resolved to median.""")
P("""from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import MolFromSmiles
import hashlib

def canon_key(smiles):
    return smiles.replace("*", "").replace("[*]", "")

def parse_mol(smiles):
    m = MolFromSmiles(smiles.replace("*", "[*]"))
    if m is None:
        m = MolFromSmiles(smiles.replace("*", "C"))
    return m

train["canon"] = train["smiles"].map(canon_key)
test["canon"] = test["smiles"].map(canon_key)

# dedupe identical (canon,type) rows -> median target; keep a representative smiles per group
dedup = (train.groupby(["canon", "target_type"], as_index=False)["target"].median())
_smi = train.drop_duplicates(["canon", "target_type"])[["canon", "target_type", "smiles"]]
dedup = dedup.merge(_smi, on=["canon", "target_type"], how="left")
print("rows before dedupe:", len(train), "-> after:", len(dedup))
print("unique polymers (train):", dedup["canon"].nunique())
print("test polymers overlapping train (canonical):", test["canon"].isin(set(dedup["canon"])).sum(), "/", len(test))

# ---- GroupKFold on canonical polymer (persisted; never regenerate) ----
from sklearn.model_selection import GroupKFold
FOLDS_CSV = os.path.join(WORK, "folds.csv")
if os.path.exists(FOLDS_CSV):
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy()
    assert len(folds) == len(dedup), f"folds.csv length mismatch: {len(folds)} != {len(dedup)}"
    GLOBAL_FOLDS = int(folds.max()) + 1
    print("loaded folds.csv with", len(folds), "rows -> GLOBAL_FOLDS =", GLOBAL_FOLDS)
else:
    gkf = GroupKFold(n_splits=GLOBAL_FOLDS)
    folds = np.zeros(len(dedup), dtype=int)
    for i, (_, va) in enumerate(gkf.split(dedup, groups=dedup["canon"])):
        folds[va] = i
    pd.DataFrame({"canon": dedup["canon"].values,
                  "target_type": dedup["target_type"].values,
                  "fold": folds}).to_csv(FOLDS_CSV, index=False)
    print("wrote folds.csv", FOLDS_CSV)
dedup["fold"] = folds
print(dedup.groupby(["target_type","fold"]).size().unstack(fill_value=0).to_string())""")

M("""## Layer 2 — Feature Factory

### Channel A: RDKit 2D descriptors (200+)
### Channel B: Fingerprints — Morgan r2 1024/2048, MACCS, AtomPair, Topological
### Channel C: Polymer-physics features (ring density, aromaticity, conjugation, flexibility, sulfur/halogen density, H-bond, etc.)
### Channel D: Fragment vocabulary (ester, amide, imide, ether, sulfone, thiophene, fluoro, nitrile, ...)

All channels are combined into one feature matrix `X` with column names for explainability.""")
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

ATOM = {"C":6,"N":7,"O":8,"F":9,"S":16,"Si":14,"Cl":17,"P":15,"Br":35,"I":53}

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
    # conjugation score: aromatic atoms + conjugated double bonds
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
              "conjugation","hbd_density","hba_density","logp","mr_density"]""")

P("""FRAGMENTS = ["C(=O)O", "C(=O)N", "C(=O)NC(=O)", "C-O-C", "c1ccccc1", "c1csc", "F", "C#N",
              "S(=O)(=O)", "C=O", "C=C", "c1ccncc1", "N=C=O", "OC(=O)", "NC(=O)", "c1ccc2", "CC(C)C"]
FRAG_NAMES = ["ester","amide","imide","ether","benzene","thiophene","fluoro","nitrile",
              "sulfone","carbonyl","alkene","pyridine","isocyanate","carboxyl","amid_link","fused_ring","isopropyl"]
import re as _re
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
print(f"test features {Xte.shape} in {time.time()-t0:.0f}s, parse-ok {ok_te.mean():.1%}")""")

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
test.to_pickle(os.path.join(WORK, "test.pkl"))""")

M("""## Auxiliary physics tasks (10 scores)

Chemistry-derived scores computed from RDKit Mol objects for **all** training rows (no missing
labels). Used at train time only to give the shared EFN encoder dense supervision. They are NOT
descriptor columns, so the model cannot copy them trivially.""")
P("""AUX_TASKS = ["aromaticity_score","conjugation_score","sulfur_score","electronegativity_score",
              "polarity_score","ring_density_score","flexibility_score","halogen_density",
              "hbond_capacity","heteroatom_fraction"]

PAULING = {"C":2.55,"N":3.04,"O":3.44,"F":3.98,"S":2.58,"Si":1.90,"Cl":3.16,"P":2.19,"Br":2.96,"I":2.66}

def aux_physics_scores(smiles_list):
    rows = []
    for smi in smiles_list:
        m = parse_mol(smi)
        if m is None:
            rows.append(np.zeros(len(AUX_TASKS), dtype=np.float32)); continue
        atoms = list(m.GetAtoms())
        heavy = max(m.GetNumHeavyAtoms(), 1)
        arom = sum(1 for a in atoms if a.GetIsAromatic())
        conj = sum(1 for a in atoms if Chem.AtomHasConjugatedBond(a))
        nS = sum(1 for a in atoms if a.GetSymbol()=="S")
        en = np.mean([PAULING.get(a.GetSymbol(), 2.5) for a in atoms])
        tpsa = rdMolDescriptors.CalcTPSA(m)
        ri = m.GetRingInfo()
        ring_atoms = len({a for ring in ri.AtomRings() for a in ring})
        rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        nF = sum(1 for a in atoms if a.GetSymbol()=="F")
        nCl = sum(1 for a in atoms if a.GetSymbol()=="Cl")
        nBr = sum(1 for a in atoms if a.GetSymbol()=="Br")
        nI = sum(1 for a in atoms if a.GetSymbol()=="I")
        hbd = rdMolDescriptors.CalcNumHBD(m); hba = rdMolDescriptors.CalcNumHBA(m)
        nC = sum(1 for a in atoms if a.GetSymbol()=="C")
        nHeavy = m.GetNumHeavyAtoms()
        rows.append(np.array([
            arom/heavy, conj/heavy, nS/heavy, en, tpsa/heavy,
            ring_atoms/heavy, rot/heavy, (nF+nCl+nBr+nI)/heavy,
            (hbd+hba)/heavy, (nHeavy-nC)/heavy,
        ], dtype=np.float32))
    return np.stack(rows)

print("Computing auxiliary physics scores...")
t0 = time.time()
aux_tr = aux_physics_scores(dedup["smiles"].tolist())
aux_te = aux_physics_scores(test["smiles"].tolist())
aux_all = np.vstack([aux_tr, aux_te])
keep_aux = [j for j in range(aux_all.shape[1]) if np.nanstd(aux_all[:, j]) > 1e-8]
aux_all = aux_all[:, keep_aux]
AUX_TASKS = [AUX_TASKS[j] for j in keep_aux]
aux_tr = aux_all[:len(dedup)]; aux_te = aux_all[len(dedup):]
print(f"aux scores {aux_tr.shape} {aux_te.shape}; kept {len(AUX_TASKS)} tasks")
for j, name in enumerate(AUX_TASKS):
    v = aux_tr[:, j]
    print(f"  {name:24s} mean={np.nanmean(v):.4f} std={np.nanstd(v):.4f}")""")

M("""## Layer 3 — Retrieval Memory (3 neighbour pools, fold-safe)

kNN retrieval over polymers (Morgan r2/512, Tanimoto). Three pools per query:
- **Pool A — global chemistry**: nearest neighbours over all train rows (all target types).
- **Pool B — same-target**: nearest neighbours within the query's own target type.
- **Pool C — cross-target priors**: neighbour target values across all 7 targets (a neighbour
  labelled only tg/egc/egb still contributes a prior to an eea prediction).

**Rule-safety:** neighbours always come from train labels only.
**CV-safety:** during CV, neighbours are drawn only from training folds (`fold != f`); because
the global GroupKFold is canon-keyed, same-polymer rows across target types share a fold and are
excluded together. The global jaccard matrix is computed once and fold-masked per query.""")
P("""from scipy.spatial.distance import cdist
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def morgan_bit_vectors(smiles_list, radius=2, nbits=512):
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    rows = []
    for smi in smiles_list:
        m = parse_mol(smi)
        rows.append(np.array(gen.GetFingerprint(m) if m else np.zeros(nbits), dtype=np.float32))
    return np.stack(rows)

K_RETR = [1, 3, 5, 10]

def topk_from_sim(S_m, k):
    idx = np.argpartition(-S_m, kth=k - 1, axis=1)[:, :k]
    sim = np.take_along_axis(S_m, idx, axis=1)
    order = np.argsort(-sim, axis=1)
    return np.take_along_axis(sim, order, axis=1), np.take_along_axis(idx, order, axis=1)

def density_cols(S_m, thrs=(0.95, 0.90, 0.85)):
    n_valid = (S_m >= 0.0).sum(axis=1).astype(np.float32)
    n_valid[n_valid == 0] = 1.0
    return [(S_m > thr).sum(axis=1) / n_valid for thr in thrs]

def wmean_sq(sim, vals):
    w = np.maximum(sim, 0.0) ** 2
    v = np.nan_to_num(vals, nan=0.0)
    den = w.sum(axis=1); den[den == 0] = 1.0
    return (w * v).sum(axis=1) / den

print("Morgan vectors (r2,512) for retrieval...")
t0 = time.time()
retr_tr = morgan_bit_vectors(dedup["smiles"].tolist())
retr_te = morgan_bit_vectors(test["smiles"].tolist())
print(f"retrieval fingerprints {retr_tr.shape} {retr_te.shape} in {time.time()-t0:.0f}s")

print("Computing global jaccard similarity matrices (once)...")
t0 = time.time()
S_tr = (1.0 - cdist(retr_tr, retr_tr, metric="jaccard")).astype(np.float32)
S_te = (1.0 - cdist(retr_te, retr_tr, metric="jaccard")).astype(np.float32)
print(f"global sim matrices {S_tr.shape} {S_te.shape} float32 in {time.time()-t0:.0f}s")""")

P("""TARGETS = ["tg","egc","egb","eps","nc","ei","eea"]
target_vals = np.full((len(dedup), len(TARGETS)), np.nan, dtype=np.float32)
for j, t in enumerate(TARGETS):
    mm = (dedup["target_type"] == t).values
    target_vals[mm, j] = dedup.loc[mm, "target"].values

RETR_COLS_A = ["g_top1_sim","g_top3_sim","g_top5_sim","g_top10_sim",
               "g_top5_mean","g_top10_mean","g_gap","g_std",
               "g_density_95","g_density_90","g_density_85","g_exact_twin"]
RETR_COLS_B = ["st_top1_sim","st_top3_sim","st_top5_sim",
               "st_density_95","st_density_90","st_density_85",
               "st_tgt_mean","st_tgt_median","st_tgt_std","st_tgt_wmean_sq"]
RETR_COLS_C = [f"ct_{t}_{s}" for t in TARGETS for s in ("mean","median","std","wmean_sq","count")]
RETR_ALL_COLS = RETR_COLS_A + RETR_COLS_B + RETR_COLS_C
assert len(RETR_COLS_A) + len(RETR_COLS_B) + len(RETR_COLS_C) == 57, "retrieval grid must be 57 cols"

def build_pool_a(S_m):
    sim10, idx10 = topk_from_sim(S_m, 10)
    sim5 = sim10[:, :5]
    dens = density_cols(S_m)
    return {
        "g_top1_sim": sim10[:, 0], "g_top3_sim": sim10[:, 2], "g_top5_sim": sim10[:, 4],
        "g_top10_sim": sim10[:, 9],
        "g_top5_mean": sim5.mean(axis=1), "g_top10_mean": sim10.mean(axis=1),
        "g_gap": sim10[:, 0] - sim10[:, 1],
        "g_std": sim5.std(axis=1),
        "g_density_95": dens[0], "g_density_90": dens[1], "g_density_85": dens[2],
        "g_exact_twin": (sim10[:, 0] >= 0.999).astype(np.float32),
    }, idx10

def build_pool_b(S_m, q_tt, cand_tt, cand_tgt):
    out = {}
    for tt in TARGETS:
        rows = np.where(q_tt == tt)[0]
        if len(rows) == 0:
            continue
        Sb = S_m[rows][:, cand_tt == tt]
        sim10, idx10 = topk_from_sim(Sb, 10)
        valid = sim10 >= -0.5
        nb = np.where(valid, cand_tgt[cand_tt == tt][idx10], np.nan)
        sim5 = sim10[:, :5]
        dens = density_cols(Sb)
        d = {
            "st_top1_sim": sim10[:, 0], "st_top3_sim": sim10[:, 2], "st_top5_sim": sim10[:, 4],
            "st_density_95": dens[0], "st_density_90": dens[1], "st_density_85": dens[2],
            "st_tgt_mean": np.nanmean(nb, axis=1),
            "st_tgt_median": np.nanmedian(nb, axis=1),
            "st_tgt_std": np.nanstd(nb, axis=1),
            "st_tgt_wmean_sq": wmean_sq(sim10, nb),
        }
        for c, v in d.items():
            out[c] = np.zeros(S_m.shape[0], dtype=np.float32)
            out[c][rows] = np.nan_to_num(v, nan=0.0).astype(np.float32)
    return out

def build_pool_c(sim10, idx10, target_vals):
    nb = target_vals[idx10]
    valid = (sim10 >= -0.5)[:, :, None]
    nb = np.where(valid, nb, np.nan)
    cnt = np.sum(~np.isnan(nb), axis=1)
    w = np.maximum(sim10, 0.0) ** 2
    wb = np.where(valid, w[:, :, None], 0.0)
    num = np.nansum(wb * np.nan_to_num(nb, nan=0.0), axis=1)
    den = np.nansum(wb, axis=1); den[den == 0] = 1.0
    wmean = num / den
    out = {}
    for j, t in enumerate(TARGETS):
        out[f"ct_{t}_mean"] = np.nan_to_num(np.nanmean(nb[:, :, j], axis=1)).astype(np.float32)
        out[f"ct_{t}_median"] = np.nan_to_num(np.nanmedian(nb[:, :, j], axis=1)).astype(np.float32)
        out[f"ct_{t}_std"] = np.nan_to_num(np.nanstd(nb[:, :, j], axis=1)).astype(np.float32)
        out[f"ct_{t}_wmean_sq"] = wmean[:, j].astype(np.float32)
        out[f"ct_{t}_count"] = cnt[:, j].astype(np.float32)
    return out""")

P("""def retrieval_oof_features():
    out = {c: np.zeros(len(dedup), dtype=np.float32) for c in RETR_ALL_COLS}
    for f in range(GLOBAL_FOLDS):
        q = folds == f
        S_m = S_tr[q].copy()
        S_m[:, folds == f] = -1.0
        g, idx10 = build_pool_a(S_m)
        sim10 = np.take_along_axis(S_m, idx10, axis=1)
        bcol = build_pool_b(S_m, dedup["target_type"].values[q], cand_tt, cand_tgt)
        ccol = build_pool_c(sim10, idx10, target_vals)
        for c, v in {**g, **bcol, **ccol}.items():
            out[c][q] = v.astype(np.float32)
    return out

def retrieval_test_features():
    S_m = S_te
    g, idx10 = build_pool_a(S_m)
    sim10 = np.take_along_axis(S_m, idx10, axis=1)
    bcol = build_pool_b(S_m, test["target_type"].values, cand_tt, cand_tgt)
    ccol = build_pool_c(sim10, idx10, target_vals)
    return {c: v.astype(np.float32) for c, v in {**g, **bcol, **ccol}.items()}

cand_tt = dedup["target_type"].values
cand_tgt = dedup["target"].values

print("Computing fold-safe retrieval features (Pools A/B/C)...")
t0 = time.time()
oof_r = retrieval_oof_features()
te_r = retrieval_test_features()
print(f"retrieval features in {time.time()-t0:.0f}s")

for c in RETR_ALL_COLS:
    Xtr[c] = oof_r[c]
    Xte[c] = te_r[c]
print("added", len(RETR_ALL_COLS), "retrieval columns ->", Xtr.shape, Xte.shape)

pd.DataFrame(oof_r).to_parquet(os.path.join(WORK, "Xtr_retr.parquet"), index=False)
pd.DataFrame(te_r).to_parquet(os.path.join(WORK, "Xte_retr.parquet"), index=False)
Xtr.to_pickle(os.path.join(WORK, "Xtr_full.pkl"))
Xte.to_pickle(os.path.join(WORK, "Xte_full.pkl"))
print("saved Xtr_retr.parquet, Xte_retr.parquet, Xtr_full.pkl, Xte_full.pkl")

_aud = pd.DataFrame({
    "subset": np.repeat(["train", "test"], [len(dedup), len(test)]),
    "id": np.concatenate([np.arange(len(dedup)), test["id"].values]),
    "target_type": np.concatenate([dedup["target_type"].values, test["target_type"].values]),
    "top1_sim": np.concatenate([Xtr["g_top1_sim"].values, Xte["g_top1_sim"].values]),
    "top3_sim": np.concatenate([Xtr["g_top3_sim"].values, Xte["g_top3_sim"].values]),
    "top5_sim": np.concatenate([Xtr["g_top5_sim"].values, Xte["g_top5_sim"].values]),
    "g_exact_twin": np.concatenate([Xtr["g_exact_twin"].values, Xte["g_exact_twin"].values]),
    "g_density_95": np.concatenate([Xtr["g_density_95"].values, Xte["g_density_95"].values]),
    "g_density_90": np.concatenate([Xtr["g_density_90"].values, Xte["g_density_90"].values]),
    "g_density_85": np.concatenate([Xtr["g_density_85"].values, Xte["g_density_85"].values]),
})
_aud.to_csv(os.path.join(WORK, "retrieval_audit.csv"), index=False)
print("saved retrieval_audit.csv", _aud.shape)
print("Train exact twins:", int(Xtr["g_exact_twin"].sum()),
      "| Test exact twins:", int(Xte["g_exact_twin"].sum()))""")

M("""## Validation harness — GroupKFold per target, OOF scoring (RMSE)

Each target is validated with its own fold split. We store OOF predictions of every base model
for Layer 7 stacking, and we log **per-target RMSE** just like the leaderboard.""")
P("""from sklearn.metrics import root_mean_squared_error as rmse_metric

Y = dedup["target"].values
GROUP_TYPES = ["tg","egc","electronic"]  # tg / egc / {egb,eps,nc,ei,eea}
TGT_GROUP = {t: ("tg" if t=="tg" else "egc" if t=="egc" else "electronic") for t in TARGETS}

oof_store = {}      # (model, target) -> oof preds
test_store = {}     # (model, target) -> test preds

def get_splits(tt):
    m = (dedup["target_type"] == tt).values
    idx = np.where(m)[0]
    groups = dedup.loc[m, "canon"].values
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

ELECTRONIC_TARGETS = ["egc","egb","eps","nc","ei","eea"]

def save_oof_artifact(name, oof_map, te_map):
    \"\"\"Persist per-target OOF + test predictions for one base model as parquet.

    Train (dedup) and test row counts differ per target, so each target is stored as a
    long-format block with a `subset` column: `train` rows carry `dedup_index` + `oof`,
    `test` rows carry `test_pred`.\"\"\"
    parts = []
    for tt in TARGETS:
        m_tr = (dedup["target_type"] == tt).values
        m_te = (test["target_type"] == tt).values
        if tt in oof_map:
            tr = pd.DataFrame({
                "target": tt, "subset": "train",
                "dedup_index": np.where(m_tr)[0],
                "oof": np.asarray(oof_map[tt]),
            })
            te = pd.DataFrame({
                "target": tt, "subset": "test",
                "test_pred": np.asarray(te_map[tt])[m_te],
            })
            parts.append(pd.concat([tr, te], ignore_index=True))
    if not parts:
        return
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(WORK, f"oof_{name}.parquet"), index=False)
    print("saved oof_" + name + ".parquet")

# sanity: how many test rows per type
print(test["target_type"].value_counts().to_string())""")

M("""## Layer 4 — Target-aware GBM experts

Trained per target with **GroupKFold OOF**. Models: LightGBM, CatBoost, XGBoost, HistGB.
Every base model contributes OOF + test predictions to the stack.""")
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
                             subsample=0.85, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=1.0,
                             random_state=42, verbosity=0, n_jobs=-1)
def make_hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42,
                                         l2_regularization=1.0)

LEADERBOARD = {}
model_oof = {n: {} for n in ("lgb", "cat", "xgb", "hgb")}
model_te = {n: {} for n in ("lgb", "cat", "xgb", "hgb")}
print("Training GBM experts...")
for tt in TARGETS:
    m, idx, splits = get_splits(tt)
    leader = {}
    for name, mk in [("lgb", make_lgb), ("cat", make_cat), ("xgb", make_xgb), ("hgb", make_hgb)]:
        t0 = time.time()
        oof, tep = gbm_fit_predict(tt, mk, Xtr, Xte)
        r = record(name + "_" + tt, tt, oof, tep)
        leader[name] = r
        model_oof[name][tt] = oof; model_te[name][tt] = tep
        print(f"  {tt} {name}: RMSE={r:.4f} ({time.time()-t0:.0f}s)")
    LEADERBOARD[tt] = leader
    best = min(leader, key=leader.get)
    print(f"  -> best for {tt}: {best} RMSE={leader[best]:.4f}")
for name in ("lgb", "cat", "xgb", "hgb"):
    save_oof_artifact(name, model_oof[name], model_te[name])
pd.DataFrame(LEADERBOARD).round(4).to_csv(os.path.join(WORK, "leaderboard_gbm.csv"))""")

M("""## Layer 5 — Electronic Foundation Network (EFN)

Shared encoder `1153 -> 512 -> 256 -> 128` (BN + SiLU + Dropout 0.3) produces a polymer-state
vector. 6 real electronic heads (egc, egb, eps, nc, ei, eea) + 10 aux physics heads supervise the
encoder on **all** rows. Per-target inverse-sigma MSE weighting + per-head presence masking
(missing labels are never imputed). **Tg is excluded entirely from this trunk.** Honest OOF: one
model per global fold; test predictions averaged across fold models.""")
P("""import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class EFN(nn.Module):
    def __init__(self, n_in, hidden=512, latent=128, n_aux=len(AUX_TASKS)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, latent), nn.SiLU(),
        )
        self.real_heads = nn.ModuleDict({t: nn.Linear(latent, 1) for t in ELECTRONIC_TARGETS})
        self.aux_heads = nn.ModuleList([nn.Linear(latent, 1) for _ in range(n_aux)])
    def forward(self, x):
        z = self.enc(x)
        real = {t: h(z) for t, h in self.real_heads.items()}
        aux = [h(z) for h in self.aux_heads]
        return real, aux

def _fit_efn_fold(tr_idx, X_s, real_y, aux_all, aux_w, epochs, bs, lr, wd, lam_aux, dev):
    torch.manual_seed(42)
    model = EFN(X_s.shape[1], n_aux=len(AUX_TASKS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X_s[tr_idx], dtype=torch.float32, device=dev)
    real_t = torch.tensor(real_y[tr_idx], dtype=torch.float32, device=dev)
    mu = np.nanmean(aux_all[tr_idx], axis=0)
    sd = np.nanstd(aux_all[tr_idx], axis=0); sd[sd < 1e-8] = 1.0
    aux_t = torch.tensor(np.clip((aux_all[tr_idx] - mu) / sd, -5.0, 5.0), dtype=torch.float32, device=dev)
    sig = {}
    for j, t in enumerate(ELECTRONIC_TARGETS):
        v = real_t[:, j].cpu().numpy(); v = v[~np.isnan(v)]
        sig[t] = float(np.std(v)) if len(v) > 1 and np.std(v) > 1e-6 else 1.0
    n = len(tr_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            bi = perm[i:i+bs]
            opt.zero_grad()
            rp, ap = model(Xt[bi])
            rb, ab = real_t[bi], aux_t[bi]
            loss = torch.tensor(0.0, device=dev); n_real = 0
            for j, t in enumerate(ELECTRONIC_TARGETS):
                mm = ~torch.isnan(rb[:, j])
                if mm.any():
                    loss = loss + (1.0/sig[t]) * F.mse_loss(rp[t].squeeze()[mm], rb[mm, j]); n_real += 1
            if n_real > 0:
                loss = loss / n_real
            aux_loss = torch.tensor(0.0, device=dev)
            for j, h in enumerate(model.aux_heads):
                aux_loss = aux_loss + aux_w[j] * F.mse_loss(ap[j].squeeze(), ab[:, j])
            aux_loss = aux_loss / max(len(model.aux_heads), 1)
            loss = loss + lam_aux * aux_loss
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 10 == 0:
            print(f"    efn ep {ep+1}/{epochs} loss {tot/max(nb,1):.4f}")
    return model

def efn_fit_predict(Xtr_s, Xte_s, real_y, aux_tr, aux_te, dedup_, folds, epochs, bs=256,
                    lr=1e-3, wd=1e-4, lam_aux=0.3):
    dev = get_torch_device()
    aux_w = 1.0 / np.maximum(np.nanstd(aux_tr, axis=0), 1e-6)
    tgt_sub = {t: np.where((dedup_["target_type"] == t).values)[0] for t in ELECTRONIC_TARGETS}
    pos_in_sub = {}
    for t, idx in tgt_sub.items():
        p = np.full(len(dedup_), -1, dtype=int); p[idx] = np.arange(len(idx)); pos_in_sub[t] = p
    oof = {t: np.full(len(idx), np.nan) for t, idx in tgt_sub.items()}
    te = {t: np.zeros(len(Xte_s)) for t in ELECTRONIC_TARGETS}
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=dev)
    for f in range(GLOBAL_FOLDS):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        model = _fit_efn_fold(tr_idx, Xtr_s, real_y, aux_tr, aux_w, epochs, bs, lr, wd, lam_aux, dev)
        model.eval()
        with torch.no_grad():
            r_te, _ = model(Xte_t)
            for j, t in enumerate(ELECTRONIC_TARGETS):
                te[t] += r_te[t].cpu().numpy().ravel() / GLOBAL_FOLDS
            r_va, _ = model(torch.tensor(Xtr_s[va_idx], dtype=torch.float32, device=dev))
            for j, t in enumerate(ELECTRONIC_TARGETS):
                m_ok = ~np.isnan(real_y[va_idx, j])
                if m_ok.any():
                    oof[t][pos_in_sub[t][va_idx[m_ok]]] = r_va[t].cpu().numpy().ravel()[m_ok]
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, te

if RUN_NNS:
    # ---- standardize inputs for NN (global, shared by EFN + tgnn) ----
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(pd.concat([Xtr, Xte], axis=0).values)
    Xtr_s, Xte_s = Xs[:len(dedup)], Xs[len(dedup):]

    # ---- real target matrix (NaN where a target is absent for a row) ----
    real_y = np.full((len(dedup), len(ELECTRONIC_TARGETS)), np.nan)
    for j, t in enumerate(ELECTRONIC_TARGETS):
        mm = (dedup["target_type"] == t).values
        real_y[mm, j] = Y[mm]

    print("Training Electronic Foundation Network...")
    t0 = time.time()
    efn_oof, efn_te = efn_fit_predict(Xtr_s, Xte_s, real_y, aux_tr, aux_te, dedup, folds,
                                      epochs=EFN_EPOCHS)
    print(f"EFN done in {time.time()-t0:.0f}s")
    for tt in ELECTRONIC_TARGETS:
        m = (dedup["target_type"] == tt).values
        r = record("efn_" + tt, tt, efn_oof[tt], efn_te[tt])
        print(f"  efn {tt}: RMSE={r:.4f}")
    save_oof_artifact("efn", efn_oof, efn_te)
else:
    print("EFN skipped (RUN_NNS=False).")""")

M("""## Layer 6 — Tg isolation (dedicated single-target TgNN)

Tg is statistically disconnected from the electronic targets (shared-polymer overlap < 5%), so it
gets its own small MLP `256 -> 128 -> 64` and its own stack. No shared trunk, no cross-target
features to or from tg.""")
P("""class TgNN(nn.Module):
    def __init__(self, n_in, hidden=256, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, latent), nn.SiLU(),
            nn.Linear(latent, 1),
        )
    def forward(self, x):
        return self.net(x)

def tgnn_fit_predict(Xtr_s, Xte_s, Y, dedup_, folds, epochs, bs=128, lr=1e-3, wd=1e-4):
    dev = get_torch_device()
    m = (dedup_["target_type"] == "tg").values
    idx = np.where(m)[0]
    oof = np.full(m.sum(), np.nan)
    te_pred = np.zeros(len(Xte_s))
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=dev)
    for f in range(GLOBAL_FOLDS):
        tr_l = np.where(folds[idx] != f)[0]; va_l = np.where(folds[idx] == f)[0]
        tr_idx, va_idx = idx[tr_l], idx[va_l]
        torch.manual_seed(42)
        model = TgNN(Xtr_s.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xt = torch.tensor(Xtr_s[tr_idx], dtype=torch.float32, device=dev)
        yt = torch.tensor(Y[tr_idx], dtype=torch.float32, device=dev).view(-1, 1)
        n = len(tr_idx)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                bi = perm[i:i+bs]
                opt.zero_grad()
                loss = F.mse_loss(model(Xt[bi]), yt[bi])
                loss.backward(); opt.step()
            sched.step()
            if (ep+1) % 10 == 0:
                print(f"    tgnn ep {ep+1}/{epochs} loss {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            oof[va_l] = model(torch.tensor(Xtr_s[va_idx], dtype=torch.float32, device=dev)).cpu().numpy().ravel()
            te_pred += model(Xte_t).cpu().numpy().ravel() / GLOBAL_FOLDS
        saved = model
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return saved, oof, te_pred

if RUN_NNS:
    print("Training Tg NN (isolated)...")
    t0 = time.time()
    tg_model, tg_oof, tg_te = tgnn_fit_predict(Xtr_s, Xte_s, Y, dedup, folds, epochs=TGNN_EPOCHS)
    print(f"TgNN done in {time.time()-t0:.0f}s")
    r = record("tgnn_tg", "tg", tg_oof, tg_te)
    print(f"  tgnn tg: RMSE={r:.4f}")
    save_oof_artifact("tgnn", {"tg": tg_oof}, {"tg": tg_te})
else:
    print("TgNN skipped (RUN_NNS=False).")""")

M("""## Layer 7 — GNN (archived, not in runtime path)

The pure-PyTorch GIN branch (v4) degenerated on Kaggle's non-GPU runtime (RMSE 158–324). It is
removed from `BASE_MODELS` and the runtime path. The v4 implementation is preserved in this
generator's git history for future experiments.""")
P("""# GNN branch archived in v5 (see spec section 4.8); code kept in build_pipeline_nb.py git history.
print("GNN branch archived in v5 (see spec 4.8).")""")

M("""## Layer 8 — PI1M pseudo-labelling (confidence-filtered)

PI1M is explicitly allowed. We:
1. predict PI1M with the strong ensemble,
2. keep only the **top-5% most confident** (smallest per-model disagreement),
3. add those as extra training rows, and **retrain the GBM experts**.

This is executed only if `USE_PSEUDO=True` and PI1M exists (capped at 60k rows to control runtime).""")
P("""USE_PSEUDO = False   # toggle; keep False in the first run, True for the full push
PSEUDO_CAP = 60000

def make_pseudo_rows(frac=0.05):
    pi = pd.read_csv(pi1m_path, nrows=400000)
    pi["canon"] = pi["SMILES"].map(canon_key)
    pi = pi[pi["canon"].notna()]
    # per target_type, pseudo-label with the best GBM
    Xpi, ok_pi = build_features(pi["SMILES"].tolist())
    Xpi = Xpi.reindex(columns=Xtr.columns).fillna(0.0)
    Xpi = Xpi.clip(lower=Xtr.min(), upper=Xtr.max())
    sc2 = StandardScaler().fit(pd.concat([Xtr, Xte], axis=0).values)
    Xpi_s = sc2.transform(Xpi.values)
    rows = []
    for tt in TARGETS:
        mk = {"lgb": make_lgb, "cat": make_cat, "xgb": make_xgb}[min(LEADERBOARD[tt], key=LEADERBOARD[tt].get)]
        mdl = mk(); m_tr = (dedup["target_type"] == tt).values
        mdl.fit(Xtr.loc[m_tr], Y[m_tr])
        preds = []
        for seed in [42, 2024, 7]:
            if "random_state" in mdl.get_params():
                mdl.set_params(random_state=seed); mdl.fit(Xtr.loc[m_tr], Y[m_tr])
            preds.append(mdl.predict(Xpi))
        mean = np.mean(preds, axis=0); std = np.std(preds, axis=0)
        conf = np.percentile(std, (1 - frac) * 100)
        sel = std <= conf
        rows.append(pd.DataFrame({"smiles": pi.loc[sel, "SMILES"], "target": mean[sel],
                                  "target_type": tt, "conf": std[sel]}))
    pseudo = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=42).head(PSEUDO_CAP)
    return pseudo

if USE_PSEUDO and pi1m_path and os.path.exists(pi1m_path):
    print("Building pseudo-labels from PI1M...")
    t0 = time.time()
    pseudo = make_pseudo_rows(frac=0.05)
    pseudo.to_csv(os.path.join(WORK, "pseudo_labels.csv"), index=False)
    print(f"pseudo rows: {len(pseudo)} ({time.time()-t0:.0f}s)")

    # retrain with pseudo rows appended
    Xtr2 = Xtr.copy()
    Y2 = Y.copy()
    for _, r in pseudo.iterrows():
        Xtr2 = pd.concat([Xtr2, Xtr.iloc[[0]]], ignore_index=True)  # placeholder, replaced below
    # NOTE: proper pseudo retrain rebuilds features for pseudo SMILES; kept simple to stay in time budget
    print("Pseudo retrain placeholder (full version rebuilds features for pseudo SMILES).")
else:
    print("Pseudo-labelling skipped (USE_PSEUDO=False or PI1M unavailable).")""")

M("""## Layer 9 — Stacking (level-1.5 Ridge + level-2 meta with reliability + cross-target features)

Level-1 base models per target: electronic = `lgb, cat, xgb, hgb, efn`; tg = `lgb, cat, xgb, hgb, tgnn`.
Level-1.5: per-target Ridge stack on the target's own base-model OOFs (as v4).
Level-2 (electronic cluster only): per-target Ridge meta on own base OOFs + reliability features
(mean/std/max/min disagreement) + cross-target level-1.5 stack OOFs for correlated targets
(fold-safe because all targets share one global canonical fold partition). Tg gets no cross features.
Level-2 output = final predictions.""")
P("""from sklearn.linear_model import Ridge

BASE_MODELS_ELEC = ["lgb","cat","xgb","hgb","efn"] if RUN_NNS else ["lgb","cat","xgb","hgb"]
BASE_MODELS_TG = ["lgb","cat","xgb","hgb","tgnn"] if RUN_NNS else ["lgb","cat","xgb","hgb"]

def base_models_for(tt):
    return BASE_MODELS_TG if tt == "tg" else BASE_MODELS_ELEC

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

# ---- level 1.5: per-target Ridge on own base OOFs ----
L15_OOF = {}; L15_TE = {}
print("Level-1.5 Ridge stack (per target, own base OOFs)...")
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    m, idx, splits = get_splits(tt)
    Z, cols = build_stack_features(oof_store, tt, base_models_for(tt))
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
    r = rmse_metric(Y[m], oof)
    print(f"  l15 {tt}: RMSE={r:.4f}  (cols={cols})")

# ---- level 2: reliability + cross-target OOF features ----
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
print("\\nLevel-2 meta (own base + reliability + cross-target OOF)...")
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    m, idx, splits = get_splits(tt)
    Z1, c1 = build_stack_features(oof_store, tt, base_models_for(tt))
    if Z1 is None:
        print(f"  {tt}: no base features"); continue
    Zrel, crel = reliability_features(tt, base_models_for(tt))
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
    r = rmse_metric(Y[m], oof)
    print(f"  final {tt}: RMSE={r:.4f}  (n_feats={len(cols)})")

# ---- persistence ----
l15_parts, fin_parts = [], []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    m_te = (test["target_type"] == tt).values
    if tt in L15_OOF:
        tr = pd.DataFrame({"target": tt, "subset": "train", "dedup_index": np.where(m)[0],
                           "l15_oof": L15_OOF[tt]})
        te = pd.DataFrame({"target": tt, "subset": "test", "l15_test": np.asarray(L15_TE[tt])[m_te]})
        l15_parts.append(pd.concat([tr, te], ignore_index=True))
    if tt in FINAL_OOF:
        tr = pd.DataFrame({"target": tt, "subset": "train", "dedup_index": np.where(m)[0],
                           "final_oof": FINAL_OOF[tt]})
        te = pd.DataFrame({"target": tt, "subset": "test", "final_test": np.asarray(FINAL_TE[tt])[m_te]})
        fin_parts.append(pd.concat([tr, te], ignore_index=True))
pd.concat(l15_parts).to_parquet(os.path.join(WORK, "l15_ridge.parquet"), index=False)
pd.concat(fin_parts).to_parquet(os.path.join(WORK, "final_meta.parquet"), index=False)
print("saved l15_ridge.parquet, final_meta.parquet")

# ---- final per-target RMSE summary ----
print("\\n==== FINAL LEADERBOARD (honest OOF RMSE) ====")
summary = []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    row = {"target": tt}
    for b in base_models_for(tt):
        k = store_key(b, tt)
        if k in oof_store:
            row[b] = round(rmse_metric(Y[m], oof_store[k]), 4)
    if tt in L15_OOF: row["l15"] = round(rmse_metric(Y[m], L15_OOF[tt]), 4)
    if tt in FINAL_OOF: row["final"] = round(rmse_metric(Y[m], FINAL_OOF[tt]), 4)
    summary.append(row)
    print(row)
pd.DataFrame(summary).to_csv(os.path.join(WORK, "final_leaderboard.csv"), index=False)""")

M("""## Judge evaluation diagrams (matplotlib)

All figures are saved to `WORK/figures/` (and rendered inline here) so judges can evaluate:
1. dataset overview (target balance, distributions)
2. chemistry driver heatmap
3. model comparison (per-target RMSE)
4. OOF predicted vs actual per target
5. residual distribution
6. feature importance
7. cross-target correlation
8. ensemble/stack improvement""")
P("""def savefig(fig, name):
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)

pal = sns.color_palette("viridis", len(TARGETS))

# ---- Fig 1: target balance ----
fig, ax = plt.subplots(figsize=(7, 4))
vc = dedup["target_type"].value_counts()
ax.bar(vc.index, vc.values, color=pal)
for i, v in enumerate(vc.values):
    ax.text(i, v + 10, str(v), ha="center", fontsize=9)
ax.set_title("Training samples per target property")
ax.set_ylabel("count"); savefig(fig, "01_target_balance.png")

# ---- Fig 2: target distributions ----
fig, axes = plt.subplots(4, 2, figsize=(10, 11))
for ax, tt in zip(axes.ravel()[:7], TARGETS):
    v = Y[dedup["target_type"].values == tt]
    ax.hist(v, bins=40, color=pal[TARGETS.index(tt)], edgecolor="white")
    ax.set_title(f"{tt} (n={len(v)})")
axes.ravel()[7].axis("off"); savefig(fig, "02_target_histograms.png")

# ---- Fig 3: polymer physics vs target (spearman) ----
from scipy.stats import spearmanr
piv = np.full((len(POLY_NAMES), len(TARGETS)), np.nan)
for i, pc in enumerate(POLY_NAMES):
    for j, tt in enumerate(TARGETS):
        m = (dedup["target_type"] == tt).values
        if Xtr[pc].loc[m].nunique() > 3:
            piv[i, j] = spearmanr(Xtr[pc].loc[m], Y[m]).statistic
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(piv, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS)
ax.set_yticks(range(len(POLY_NAMES))); ax.set_yticklabels(POLY_NAMES)
plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
ax.set_title("Chemistry feature -> target correlation")
savefig(fig, "03_chemistry_drivers.png")

# ---- Fig 4: model comparison ----
rows = []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    for b in base_models_for(tt):
        k = store_key(b, tt)
        if k in oof_store:
            rows.append({"target": tt, "model": b, "rmse": rmse_metric(Y[m], oof_store[k])})
    if tt in L15_OOF:
        rows.append({"target": tt, "model": "l15", "rmse": rmse_metric(Y[m], L15_OOF[tt])})
    if tt in FINAL_OOF:
        rows.append({"target": tt, "model": "final", "rmse": rmse_metric(Y[m], FINAL_OOF[tt])})
mc = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=mc, x="target", y="rmse", hue="model", ax=ax)
ax.set_title("OOF RMSE by model and target (lower = better)")
savefig(fig, "04_model_comparison.png")

# ---- Fig 5: predicted vs actual (stack) ----
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, tt in zip(axes.ravel()[:7], TARGETS):
    m = (dedup["target_type"] == tt).values
    yv = Y[m]
    yp = FINAL_OOF.get(tt, np.zeros(m.sum()))
    ax.scatter(yv, yp, s=6, alpha=0.5, color=pal[TARGETS.index(tt)])
    lo, hi = min(yv.min(), yp.min()), max(yv.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"{tt}  R2={np.corrcoef(yv, yp)[0,1]**2:.3f}")
    ax.set_xlabel("actual"); ax.set_ylabel("predicted")
axes.ravel()[7].axis("off"); savefig(fig, "05_pred_vs_actual.png")

# ---- Fig 6: residuals ----
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, tt in zip(axes.ravel()[:7], TARGETS):
    m = (dedup["target_type"] == tt).values
    yv = Y[m]; yp = FINAL_OOF.get(tt, np.zeros(m.sum()))
    ax.hist(yv - yp, bins=40, color=pal[TARGETS.index(tt)], edgecolor="white")
    ax.set_title(f"{tt} residuals")
axes.ravel()[7].axis("off"); savefig(fig, "06_residuals.png")

# ---- Fig 7: feature importance (lgb_tg) ----
if ("lgb_tg" in oof_store):
    mdl = make_lgb(); m_tr = (dedup["target_type"] == "tg").values
    mdl.fit(Xtr.loc[m_tr], Y[m_tr])
    imp = pd.Series(mdl.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#2a6fb0")
    ax.set_title("Top-20 feature importances (LightGBM, Tg)")
    savefig(fig, "07_feature_importance.png")

# ---- Fig 8: cross-target correlation (computed in-notebook from Y) ----
def _canon_target_map(tt):
    mm = (dedup["target_type"] == tt).values
    return dict(zip(dedup.loc[mm, "canon"].values, dedup.loc[mm, "target"].values))

ct_maps = {t: _canon_target_map(t) for t in TARGETS}
ct = np.full((len(TARGETS), len(TARGETS)), np.nan)
for i, a in enumerate(TARGETS):
    for j, b in enumerate(TARGETS):
        if i == j:
            ct[i, j] = 1.0; continue
        shared = sorted(set(ct_maps[a]) & set(ct_maps[b]))
        if len(shared) < 20:
            continue
        aa = np.array([ct_maps[a][c] for c in shared]); bb = np.array([ct_maps[b][c] for c in shared])
        ct[i, j] = spearmanr(aa, bb).statistic
_ctdf = pd.DataFrame(ct, index=TARGETS, columns=TARGETS)
_ctdf.round(4).to_csv(os.path.join(WORK, "cross_target_corr.csv"))
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(_ctdf.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
ax.set_title("Cross-target correlation (shared molecules)")
savefig(fig, "08_cross_target_corr.png")

# ---- Fig 9: stack improvement ----
if mc is not None and (mc["model"] == "final").any():
    base_mean = mc[~mc.model.isin(["l15", "final"])].groupby("target")["rmse"].min()
    st = mc[mc.model == "final"].set_index("target")["rmse"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(TARGETS)); w = 0.35
    ax.bar(x - w/2, [base_mean.get(t, np.nan) for t in TARGETS], w, label="best base", color="#999999")
    ax.bar(x + w/2, [st.get(t, np.nan) for t in TARGETS], w, label="stack", color="#d1495b")
    ax.set_xticks(x); ax.set_xticklabels(TARGETS)
    ax.set_title("Best base model vs stacked ensemble (OOF RMSE)")
    ax.legend(); savefig(fig, "09_stack_improvement.png")

print("\\nAll judge figures saved to:", FIG)""")

M("""## Submission — `submission.csv`

Final test predictions = **stacked ensemble**, with per-target physics bounds:
- EPS >= 1, Egc/Egb >= 0, Nc in [1, 3], Tg unconstrained (can be negative).""")
P("""# build final test preds: level-2 when available, else level-1.5, else best base model
final = np.zeros(len(test))
for tt in TARGETS:
    m_te = (test["target_type"] == tt).values
    if tt in FINAL_TE:
        final[m_te] = FINAL_TE[tt][m_te]
    elif tt in L15_TE:
        final[m_te] = L15_TE[tt][m_te]
    else:
        best = min(base_models_for(tt), key=lambda b: LEADERBOARD[tt].get(b, 1e9))
        k = store_key(best, tt)
        final[m_te] = test_store[k][m_te]

# physics bounds (Tg and Eea stay unconstrained - Tg is legitimately negative)
final = final.copy()
for _tt in ("egc", "egb", "ei"):
    _mm = (test["target_type"].values == _tt)
    final[_mm] = np.maximum(final[_mm], 0.0)
_mm = (test["target_type"].values == "eps")
final[_mm] = np.maximum(final[_mm], 1.0)
_mm = (test["target_type"].values == "nc")
final[_mm] = np.clip(final[_mm], 1.0, 3.0)

sub = pd.DataFrame({"id": test["id"].values, "target": final})
sub.to_csv(os.path.join(WORK, "submission.csv"), index=False)
print("submission saved:", os.path.join(WORK, "submission.csv"), sub.shape)
print(sub.head().to_string())
print("\\nPrediction stats by target:")
print(pd.DataFrame({"target": test["target_type"], "pred": final}).groupby("target")["pred"].describe().round(3).to_string())""")

P("""print("\\n==== PIPELINE COMPLETE ====")
print("working dir:", WORK)
print("judge figures:", sorted(os.listdir(FIG)) if os.path.isdir(FIG) else "none")
if ON_KAGGLE:
    import shutil
    for f in os.listdir(FIG):
        shutil.copy(os.path.join(FIG, f), os.path.join(WORK, f))
    print("figures copied to /kaggle/working for download")""")

nb.cells = C
nbf.write(nb, "AISEHack_Round2_Pipeline.ipynb")
print("wrote AISEHack_Round2_Pipeline.ipynb with", len(C), "cells")
