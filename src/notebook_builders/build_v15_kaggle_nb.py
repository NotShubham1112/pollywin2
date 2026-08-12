#!/usr/bin/env python
"""Build the PolyWin R2 v15 Kaggle notebook — EPS/NC focus fine-tuning.

v15 is a SINGLE-CHANGE experiment on top of the frozen v14 (P14 full-PI1M
pretraining, LB 0.883):

    v14 (baseline, frozen):
        g.w = 1.0 / freq[target_type]                 (inverse-freq sample weight)
    v15 (this):
        g.w = 1.0 / freq[target_type] * TGT_FOCUS.get(target_type, 1.0)
        TGT_FOCUS = {"eps": 2.0, "nc": 2.0}

The extra focus factor applies ONLY inside build_graphs (fine-tune graph
construction), so it rescales the TRAINING loss for the two weakest targets
(eps / nc) and nothing else — validation loss and test prediction are untouched.

Everything else is VERBATIM from the v14 kernel (build_v14_kaggle_nb.py):
  * CORE_A / CORE_B are extracted from mt_gnn_v2.py bit-identically,
  * full-PI1M pretrain (~995k molecules, 10 epochs),
  * GNN seeds 42,999,2025, folds 5, max-epochs 120, patience 20, batch 256,
  * GBM trio stack + per-target Ridge(alpha=1.0) blend, same alpha grid,
  * same descriptors/fingerprints, same fold assignments, same seeds.

Success criteria (pre-registered, judged ONLY after the Kaggle run, vs v14):
    Pass: (eps blend OOF gain >= +0.01 OR nc blend OOF gain >= +0.01)
          AND overall (7-target mean) blend OOF >= +0.003.
    Fail:  overall gain < +0.003, OR eps/nc gain but other targets regress
           enough to offset — then STOP, keep P14 (LB 0.883) as final.

Run:  python build_v15_kaggle_nb.py          # -> PolyWin_R2_v15_epsnc_focus.ipynb
      SMOKE=1 python build_v15_kaggle_nb.py  # fast smoke validation (2 folds, 1 epoch)
"""
import nbformat as nbf
import os as _os

SMOKE = _os.environ.get("SMOKE", "0") == "1"

SRC = "src/core/mt_gnn_v2.py"
OUT_NB = "notebooks/v15_epsnc_focus/PolyWin_R2_v15_epsnc_focus.ipynb"

PLAIN_LINE = "g.w = torch.tensor([1.0 / freq[row.target_type]], dtype=torch.float)"
BOOST_LINE = ("g.w = torch.tensor([1.0 / freq[row.target_type] "
              "* TGT_FOCUS.get(row.target_type, 1.0)], dtype=torch.float)")

with open(SRC, encoding="utf-8") as f:
    L = f.read().split("\n")


def _idx(marker):
    for i, line in enumerate(L):
        if marker in line:
            return i
    raise SystemExit("marker not found: " + marker)


_A_START = _idx("# Graph featurization")
_A_END = _idx("# Twin source:")
_B_END = _idx("    stack_oof[t] = oof; stack_test[t] = te_pred") + 1
CORE_A = "\n".join(L[_A_START:_A_END])
CORE_B = "\n".join(L[_A_END:_B_END])
assert "class GINEEncoder" in CORE_A
assert "class MTGNN" in CORE_A
assert "lgb_test_te = np.zeros((len(Xte), len(TARGETS)), dtype=np.float32)" in CORE_B
assert "stack_oof[t] = oof; stack_test[t] = te_pred" in CORE_B

# --- THE single v15 change: apply the boost to the training sample weight. ---
assert CORE_A.count(PLAIN_LINE) == 1, "expected exactly one sample-weight line in CORE_A"
CORE_A = CORE_A.replace(PLAIN_LINE, BOOST_LINE)
assert BOOST_LINE in CORE_A and PLAIN_LINE not in CORE_A

REPL = {
    "@FOLDS@": "2" if SMOKE else "5",
    "@MAXEP@": "4" if SMOKE else "120",
    "@PATE@": "5" if SMOKE else "20",
    "@BS@": "64" if SMOKE else "256",
    "@PRTEP@": "1" if SMOKE else _os.environ.get("PRETRAIN_EPOCHS", "10"),
    "@PRTSMP@": "2000" if SMOKE else _os.environ.get("PRETRAIN_SAMPLE", "2000000"),
    "@SUFFIX@": "_smoke" if SMOKE else "",
    "@GNNSEEDS@": _os.environ.get("GNN_SEEDS", "1" if SMOKE else "42,999,2025"),
}

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
P = lambda s: C.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------------
intro = f"""# PolyWin R2 — v15: EPS/NC FOCUS fine-tuning (single-change experiment)

## Experiment (pre-registered)
* v14 baseline scores **0.883 LB** (full-PI1M pretraining, P14).
* **v15 changes ONLY the fine-tune sample weight:** rows belonging to the
  **eps** and **nc** targets get an extra ×2.0 weight in the MT-GNN training
  loss — `TGT_FOCUS = {{"eps": 2.0, "nc": 2.0}}` (on top of the existing
  1/freq[target] weighting). Validation loss, test prediction, GBM stack,
  ridge blend, descriptors, folds, seeds and the full-PI1M pretrain are
  **BIT-IDENTICAL** to the frozen v14 baseline.

## Success / fail criteria (judged only after the Kaggle run)
* Pass: eps blend OOF gain ≥ +0.01 (or nc ≥ +0.01) AND overall 7-target blend
  OOF gain ≥ +0.003 vs the v14 baseline.
* Fail: overall gain < +0.003 (or other targets regress enough to cancel) →
  STOP. P14 (LB 0.883) remains the final submission.

Only OSI-approved libs: PyTorch, PyG, RDKit, scikit-learn, LightGBM, CatBoost, XGBoost.
"""
M(intro)

# ------------------------------------------------------------------------
setup = r'''import os, sys, time, gc, random, warnings
import subprocess, importlib.util

def ensure_pkg(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print("installing", pkg, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--disable-pip-version-check", pkg])

for _p, _n in [("rdkit", "rdkit"), ("torch_geometric", "torch_geometric"),
               ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("xgboost", "xgboost")]:
    ensure_pkg(_p, _n)

# --- CUDA probe / repair identical to v14 (P100 sm_60) ---
_probe = ('import torch;' + 'a=torch.zeros(4,device="cuda");b=a+1;torch.cuda.synchronize();print("OK")')
def _force_cuda():
    try:
        _r = subprocess.run([sys.executable, "-c", _probe], capture_output=True,
                            text=True, timeout=600)
    except Exception:
        _r = None
    if _r is not None and _r.returncode == 0 and "OK" in (_r.stdout or ""):
        return
    print("CUDA kernel missing; installing torch 2.5.1 (cu121, supports P100 sm_60)...", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-cache-dir", "--index-url",
                               "https://download.pytorch.org/whl/cu121", "torch==2.5.1"],
                              timeout=1800)
    except Exception as _e:
        print("torch reinstall errored:", repr(_e)[:200], flush=True)
if os.path.exists("/kaggle"):
    _force_cuda()

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors, Crippen, GraphDescriptors
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SMOKE = os.environ.get("SMOKE", "0") == "1"
SEED = 42
GNN_SEEDS = os.environ.get("GNN_SEEDS", "@GNNSEEDS@")
os.environ["GNN_SEEDS"] = GNN_SEEDS
print("GNN_SEEDS =", GNN_SEEDS, flush=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def _cuda_ok():
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.zeros(4, device="cuda"); b = a + 1; torch.cuda.synchronize(); del a, b
        return True
    except Exception:
        return False
DEVICE = "cuda" if _cuda_ok() else "cpu"
print("device:", DEVICE, flush=True)
GLOBAL_FOLDS = @FOLDS@
MAX_EPOCHS = @MAXEP@
PATIENCE = @PATE@
EARLY_HOLDOUT = 0.15
BS = @BS@
LR = 1e-3
if DEVICE == "cpu":
    GLOBAL_FOLDS = min(GLOBAL_FOLDS, 2)
    MAX_EPOCHS = min(MAX_EPOCHS, 12)
    BS = 256
    PATIENCE = min(PATIENCE, 6)
PRETRAIN_EPOCHS = @PRTEP@
PRETRAIN_SAMPLE = @PRTSMP@
TGT_FOCUS = {"eps": 2.0, "nc": 2.0}

if os.path.exists("/kaggle"):
    WORK = "/kaggle/working"; INP = "/kaggle/input"
else:
    WORK = os.path.join("vault", "pipeline_out_v15@SUFFIX@")
    INP = "competition/data/raw"
os.makedirs(WORK, exist_ok=True)
PRETRAINED = os.path.join(WORK, "pretrained_encoder.pt")
OUT = WORK

print("----- v15 CONFIG -----", flush=True)
print("PRETRAIN_SAMPLE =", PRETRAIN_SAMPLE, "| PRETRAIN_EPOCHS =", PRETRAIN_EPOCHS, flush=True)
print("TGT_FOCUS =", TGT_FOCUS, "| SMOKE:", SMOKE, "| folds:", GLOBAL_FOLDS,
      "| out:", OUT, flush=True)
'''
for _k, _v in REPL.items():
    setup = setup.replace(_k, _v)
P(setup)

# ----------------------------------------------------------------------------
P(r'''from sklearn.linear_model import Ridge

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
''')

M("## 1. Data — current-round CSVs, canonicalize, compute descriptors + fingerprints")

P(r'''def find_input(base, name):
    for p in [os.path.join(base, name), os.path.join(base, "ppp-round-2", name),
              os.path.join(base, "competitions", "ppp-round-2", name)]:
        if os.path.exists(p):
            return p
    return None

def canonical(s):
    if not isinstance(s, str):
        return None, None, None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None, None
    try:
        c = Chem.MolToSmiles(m)
        ik = Chem.MolToInchiKey(m)
    except Exception:
        return Chem.MolToSmiles(m), None, None
    return c, ik, m

def canon_fast(s):
    """MolToSmiles only (identical string to canonical(s)[0]) for the PI1M
    pretrain corpus — avoids 995k MolToInchiKey computations, no result change."""
    if not isinstance(s, str):
        return None
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None

def feats(m):
    if m is None:
        return [np.nan] * 22
    return [
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
        Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
        Descriptors.RingCount(m), Descriptors.NumAromaticRings(m),
        Descriptors.NumAliphaticRings(m), Descriptors.NumSaturatedRings(m),
        Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumHeavyAtoms(m),
        Descriptors.NumHeteroatoms(m), Descriptors.FractionCSP3(m),
        Crippen.MolMR(m), rdMolDescriptors.CalcNumBridgeheadAtoms(m),
        rdMolDescriptors.CalcNumSpiroAtoms(m),
        rdMolDescriptors.CalcNumAromaticAtoms(m) if hasattr(rdMolDescriptors, "CalcNumAromaticAtoms") else Descriptors.NumAromaticRings(m),
        GraphDescriptors.BalabanJ(m), GraphDescriptors.Ipc(m),
        rdMolDescriptors.CalcNumLipinskiHBA(m), rdMolDescriptors.CalcNumLipinskiHBD(m),
        rdMolDescriptors.CalcNumAtomStereoCenters(m),
    ]

FNAMES = ["MolWt", "LogP", "TPSA", "HDon", "HAccep", "RingCnt", "AroRing", "AliRing", "SatRing",
          "RotB", "HeavyAt", "HeteroAt", "FracCSP3", "MR", "Bridge", "Spiro", "AroAt",
          "BalabanJ", "Ipc", "LipHBA", "LIHBD", "Stereo"]
assert len(FNAMES) == 22

train_path = find_input(INP, "train.csv")
test_path = find_input(INP, "test.csv")
assert train_path and test_path, "train.csv / test.csv not found in " + INP

tr = pd.read_csv(train_path)
te = pd.read_csv(test_path)
print("train:", tr.shape, "test:", te.shape, flush=True)

tcpl = tr["smiles"].map(canonical)
tr["canon"], tr["inchikey"], _ = zip(*tcpl)
tepl = te["smiles"].map(canonical)
te["canon"], te["inchikey"], _ = zip(*tepl)

tr_f = np.array(tr["smiles"].map(lambda s: feats(Chem.MolFromSmiles(s))).tolist())
te_f = np.array(te["smiles"].map(lambda s: feats(Chem.MolFromSmiles(s))).tolist())
tr[FNAMES] = tr_f
te[FNAMES] = te_f
print("descriptors done", flush=True)

trf = tr.dropna(subset=["target"]).copy()
tef = te.copy()

FEAT_COLS = [c for c in trf.columns if c not in
            ("smiles", "target", "target_type", "canon", "inchikey", "id")]
print("FEAT_COLS:", len(FEAT_COLS), flush=True)
''')

P(r"""def add_fingerprints(df):
    morgan = np.zeros((len(df), 2048), dtype=np.float32)
    maccs = np.zeros((len(df), 167), dtype=np.float32)
    for i, s in enumerate(df["smiles"]):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        morgan[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
            m, 2, nBits=2048).ToBitString().encode(), "u1") - ord("0")
        maccs[i] = np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(),
                                 "u1") - ord("0")
    return morgan, maccs

F32_MAX = np.finfo(np.float32).max

def clean_feats(df):
    D = np.clip(df[FEAT_COLS].values, -F32_MAX, F32_MAX)
    for j in range(D.shape[1]):
        col = D[:, j]
        med = np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
    return D.astype(np.float32)

D_tr = clean_feats(trf)
mor_tr, mc_tr = add_fingerprints(trf)
X = np.hstack([D_tr, mor_tr, mc_tr]).astype(np.float32)
Xs = StandardScaler().fit(X).transform(X).astype(np.float32)

D_te = clean_feats(tef)
mor_te, mc_te = add_fingerprints(tef)
Xte = np.hstack([D_te, mor_te, mc_te]).astype(np.float32)
Xtes = StandardScaler().fit(X).transform(Xte).astype(np.float32)

Y = trf["target"].values.astype(np.float32)
T = trf["target_type"].values
G = trf["canon"].values.astype(str)

idx_of_target = {t: np.where(T == t)[0] for t in TARGETS}
print("train:", X.shape, "test:", Xte.shape, "targets:", TARGETS, flush=True)
""")

M("## 2. Level-0 sources (verbatim from mt_gnn_v2.py: graph feats + GINE + MT-GNN)")

P(CORE_A)

M("## 3. Pretrain the GINE encoder on FULL PI1M (verbatim from v14 — pre-registered unchanged)")

P(r"""pl_path = find_input(INP, "PI1M.csv")
pl = []
if pl_path:
    pldf = pd.read_csv(pl_path)
    smi_col = "SMILES" if "SMILES" in pldf.columns else "smiles"
    pldf = pldf[[smi_col]].rename(columns={smi_col: "smiles"})
    t0pl = time.time()
    pldf["canon"] = pldf["smiles"].map(canon_fast)
    print(f"PI1M canonicalized in {time.time()-t0pl:.0f}s "
          f"({len(pldf)} rows, parsed {pldf['canon'].notna().sum()})", flush=True)
    pldf = pldf.dropna(subset=["canon"])
    pl = pldf.drop_duplicates("canon")["smiles"].tolist()
    print("PI1M unique canons:", len(pl), flush=True)
    rng = np.random.RandomState(SEED); rng.shuffle(pl)
    pl = pl[:PRETRAIN_SAMPLE]
    print("PI1M full-PI1M pretraining corpus:", len(pl), "SMILES", flush=True)
else:
    print("no PI1M: pretraining skipped", flush=True)

def build_pretrain_graphs_chunked(smiles_list, chunk=50000):
    graphs = []
    t0 = time.time()
    for c0 in range(0, len(smiles_list), chunk):
        chunk_g = []
        for smi in smiles_list[c0:c0+chunk]:
            g = smiles_to_graph(smi)
            if g is not None:
                chunk_g.append(g)
        graphs.extend(chunk_g)
        del chunk_g
        gc.collect()
        print(f"  graphs {len(graphs)}/{len(smiles_list)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return graphs

pl_graphs = build_pretrain_graphs_chunked(pl) if pl else []
print("pretraining graphs (full PI1M):", len(pl_graphs), flush=True)

from torch_geometric.loader import DataLoader

class PretrainedEncoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden=128, n_layers=4,
                 mask_atom=0.15, mask_bond=0.20):
        super().__init__()
        self.encoder = GINEEncoder(n_atom_feats, n_bond_feats, hidden, n_layers)
        self.atom_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, n_atom_feats))
        self.bond_proj = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
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
            atom_loss = F.mse_loss(self.atom_proj(h[atom_mask]), x[atom_mask])
        else:
            atom_loss = torch.zeros((), device=x.device)
        if bond_mask.any():
            src = h[edge_index[0, bond_mask]]; dst = h[edge_index[1, bond_mask]]
            if src.numel() > 0:
                bond_loss = F.mse_loss(self.bond_proj(torch.cat([src, dst], dim=1)), edge_attr[bond_mask])
            else:
                bond_loss = torch.zeros((), device=x.device)
        else:
            bond_loss = torch.zeros((), device=x.device)
        return atom_loss, bond_loss

def pretrain(epochs=PRETRAIN_EPOCHS, batch_size=1024, lr=1e-3):
    if len(pl_graphs) == 0:
        print("No PI1M graphs - pretraining skipped", flush=True)
        return None
    model = PretrainedEncoder(N_ATOM_FEATS, N_BOND_FEATS).to(DEVICE)
    loader = DataLoader(pl_graphs, batch_size=batch_size, shuffle=True,
                        pin_memory=(DEVICE == "cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best = np.inf; best_state = None; t0 = time.time()
    for epoch in range(epochs):
        model.train(); tot_a = 0.0; tot_b = 0.0; nbl = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            a_loss, b_loss = model(batch.x, batch.edge_index, batch.edge_attr, batch)
            loss = a_loss + 0.5 * b_loss
            loss.backward(); opt.step()
            tot_a += a_loss.item(); tot_b += b_loss.item(); nbl += 1
            del batch
        va = (tot_a + 0.5 * tot_b) / max(nbl, 1)
        if va < best:
            best = va; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"pretrain ep {epoch+1}/{epochs}: loss={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best_state:
        torch.save(best_state, PRETRAINED)
        print("saved pretrained_encoder.pt (full PI1M)", flush=True)
    return best_state

print("=== v15: Pretraining GNN on FULL PI1M (identical to v14) ===", flush=True)
pretrained_state = pretrain()
""")

M("## 4. Level-0 predictions (verbatim: leak-safe twins + MT-GNN fold OOF + GBM trio stack)")

P(CORE_B)

M("## 5. v14 blend — per-target Ridge (OOF-tuned alpha) on [GBM, MT-GNN] OOF + submission")

P(r"""ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10, 25]

oof_gbm_global = np.full(len(X), np.nan, dtype=np.float32)
oof_mt_global = np.full(len(X), np.nan, dtype=np.float32)
for t in TARGETS:
    idx = idx_of_target[t]
    oof_gbm_global[idx] = stack_oof[t]
    oof_mt_global[idx] = mt_oof[t]
assert not np.isnan(oof_gbm_global).any() and not np.isnan(oof_mt_global).any()

test_gbm_global = np.zeros(len(Xte), dtype=np.float32)
test_mt_global = np.zeros(len(Xte), dtype=np.float32)
for t in TARGETS:
    m_te = (tef["target_type"] == t).values
    test_gbm_global[m_te] = stack_test[t][m_te]
    test_mt_global[m_te] = mt_test[m_te]

print("\n=== v15: corr(GBM OOF, GNN OOF) per target (diagnostic) ===", flush=True)
for t in TARGETS:
    idx = idx_of_target[t]
    c = np.corrcoef(oof_gbm_global[idx], oof_mt_global[idx])[0, 1]
    print(f"  {t:<4} corr={c:.4f}", flush=True)

print("\n=== v15 Criterion: tuning per-target Ridge alpha ===", flush=True)
rows = []
coefs = {t: [] for t in TARGETS}
best_a = {}
final_te = np.zeros(len(tef))
for t in TARGETS:
    idx = idx_of_target[t]
    yt = Y[idx].astype(np.float64)
    Mx = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx]])
    Mte = np.column_stack([test_gbm_global, test_mt_global])
    cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Mx, yt, G[idx]))
    oof_r2 = {}
    for a in ALPHA_GRID:
        o = np.zeros(len(idx))
        for trk, vk in cv:
            o[vk] = Ridge(alpha=a).fit(Mx[trk], yt[trk]).predict(Mx[vk])
        oof_r2[a] = r2_score(yt, o)
    a_best = max(oof_r2, key=oof_r2.get)
    best_a[t] = a_best
    oof = np.zeros(len(idx)); te_pred = np.zeros(len(tef))
    for trk, vk in cv:
        lr = Ridge(alpha=a_best); lr.fit(Mx[trk], yt[trk])
        oof[vk] = lr.predict(Mx[vk])
        te_pred += lr.predict(Mte) / GLOBAL_FOLDS
        coefs[t].append(lr.coef_.tolist())
    m_te = (tef["target_type"] == t).values
    final_te[m_te] = te_pred[m_te]
    r_blend = r2_score(yt, oof); r_g = r2_score(yt, oof_gbm_global[idx]); r_m = r2_score(yt, oof_mt_global[idx])
    cb = np.mean(coefs[t], axis=0)
    rows.append(dict(target=t, alpha=float(a_best), blend=r_blend, GBM=r_g, GNN=r_m,
                     w_GBM=cb[0], w_GNN=cb[1]))
    print(f"  {t:<4} alpha={a_best:<6} blend={r_blend:.4f} GBM={r_g:.4f} GNN={r_m:.4f} "
          f"w_GBM={cb[0]:.3f} w_GNN={cb[1]:.3f}", flush=True)

np.savez(os.path.join(OUT, "blend_oof_test.npz"),
         oof_gbm=oof_gbm_global, oof_mt=oof_mt_global,
         test_gbm=test_gbm_global, test_mt=test_mt_global,
         y_all=Y.astype(np.float64), g_all=G.astype(str), t_all=T.astype(str))
print("wrote blend_oof_test.npz", flush=True)

df = pd.DataFrame(rows).set_index("target")
print("\n=== v15 summary ===", flush=True)
print("  mean blend=%.4f | GBM=%.4f | GNN=%.4f | delta-vs-GBM %+.4f" % (
    df["blend"].mean(), df["GBM"].mean(), df["GNN"].mean(),
    df["blend"].mean() - df["GBM"].mean()), flush=True)

print("\n=== v15 Criterion vs v14 ===", flush=True)
print("summarked in report csv", flush=True)

sub = pd.DataFrame({"id": tef["id"].values, "target": final_te})
sub_path = os.path.join(OUT, "submission_v15.csv")
sub.to_csv(sub_path, index=False)
print("\nwrote", sub_path, flush=True)
print("  rows", len(sub), "| NaN", sub["target"].isna().sum(), flush=True)
df.round(4).to_csv(os.path.join(OUT, "v15_blend_report.csv"), index=True)
print("wrote", os.path.join(OUT, "v15_blend_report.csv"), flush=True)
print("v15 DONE", flush=True)
""")

nb["cells"] = C
with open(OUT_NB, "w", encoding="utf-8") as f:
    nbf.write(nb, f)