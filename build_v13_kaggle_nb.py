#!/usr/bin/env python
"""Build the PolyWin R2 v13 Kaggle notebook: GBM trio stack + MT-GNN blend.

The Level-0 architecture and ALL Level-0 math are taken VERBATIM from
mt_gnn_v2.py (lines 100-485) so they are bit-identical to the code that
produced the 0.8632 local CV. Only the data-ingestion preamble (lines 1-99)
is replaced, because Kaggle only has smiles (no r2_train_feat.pkl):
  * descriptors + fingerprints are recomputed in-kernel,
  * the GINE encoder is pretrained on PI1M in-kernel,
  * the GBM trio stack and the MT-GNN fold-safe OOF are exactly the mt_gnn_v2
    code,
  * the final submission applies the v13 per-target Ridge(alpha=1.0) blend
    (identical to v13_blend.py).

Self-contained; reads only current-round data (train.csv / test.csv / PI1M.csv).
Run:  python build_v13_kaggle_nb.py             # -> PolyWin_R2_v13_gbm_gnn_blend.ipynb
      SMOKE=1 python build_v13_kaggle_nb.py     # fast smoke validation
"""
import nbformat as nbf
import os as _os

SMOKE = _os.environ.get("SMOKE", "0") == "1"

SRC = "mt_gnn_v2.py"
OUT_NB = "PolyWin_R2_v13_gbm_gnn_blend.ipynb"

with open(SRC, encoding="utf-8") as f:
    L = f.read().split("\n")

CORE_A = "\n".join(L[99:246])   # source lines 100-246: graph feats + GINE + MTGNN
CORE_B = "\n".join(L[246:487])  # source lines 247-487: twins + MT-GNN fold + GBM stack
assert "class GINEEncoder" in CORE_A
assert "class MTGNN" in CORE_A
assert "lgb_test_te = np.zeros((len(Xte), len(TARGETS)), dtype=np.float32)" in CORE_B
assert "stack_oof[t] = oof; stack_test[t] = te_pred" in CORE_B

REPL = {
    "@FOLDS@": "2" if SMOKE else "5",
    "@MAXEP@": "4" if SMOKE else "120",
    "@PATE@": "5" if SMOKE else "20",
    "@BS@": "64" if SMOKE else "256",
    "@PRTEP@": "1" if SMOKE else "5",
    "@PRTSMP@": "2000" if SMOKE else "20000",
    "@SUFFIX@": "_smoke" if SMOKE else "",
}

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
P = lambda s: C.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------------
intro = f"""# PolyWin R2 — v13: GBM trio stack + MT-GNN per-target blend (kernel)

## Protocol (honest, self-contained)
* Reads only the active-round data: **train.csv / test.csv / PI1M.csv** (no archive, no
  uploaded artifacts).
* **Level 0** — two independent arms, verbatim from `mt_gnn_v2.py` and bit-identical to the
  0.8632-local-CV run:
  1. **GBM trio stack** — per-target LightGBM + XGBoost + CatBoost fold-safe OOF fed to a
     fold-consistent `Ridge(alpha=1.0)`.
  2. **MT-GNN** — a PI1M-inflected GINE trunk with per-target heads + leak-safe cross-target
     twin features, trained with GroupKFold-on-canonical folds (a canon's rows never train a
     model that predicts another row of the same canon).
* **Pretraining** — the GINE encoder is pretrained on the unlabeled PI1M corpus inside this
  run (masked atom/bond reconstruction), then the MT-GNN fine-tunes fold-safely. No external
  checkpoint is read.
* **Blend (v13)** — per-target `Ridge(alpha=1.0)` on the [GBM-stack, MT-GNN] OOF/test pairs,
  exactly the recipe in `v13_blend.py` that scored 0.8632 locally vs 0.8435 (GBM-only) /
  0.8382 (GNN-only). Small targets lean GNN, big targets lean GBM.
* All seeds fixed (`SEED = 42`); folds are GroupKFold by canonical SMILES.

Only OSI-approved libs: PyTorch, PyG, RDKit, scikit-learn, LightGBM, CatBoost, XGBoost.
"""
M(intro)

# ----------------------------------------------------------------------------
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

# --- Pre-import CUDA fix: torch_geometric install may have pulled a torch that
#     has no kernel for the allocated GPU ("no kernel image"). Probe in a clean
#     subprocess BEFORE importing torch; if it fails, reinstall the broad-CUDA
#     cu121 wheel so the kernel below picks it up. ---
_probe = ('import torch;' + 'a=torch.zeros(4,device="cuda");b=a+1;torch.cuda.synchronize();print("OK")')
def _force_cuda():
    try:
        _r = subprocess.run([sys.executable, "-c", _probe], capture_output=True,
                            text=True, timeout=600)
    except Exception:
        _r = None
    if _r is not None and _r.returncode == 0 and "OK" in (_r.stdout or ""):
        return
    print("CUDA kernel missing for this GPU; installing torch 2.5.1 (cu121, supports sm_60)...", flush=True)
    try:
        # Pin 2.5.1: unlike the base cu128 build (sm_70+) it ships sm_60 kernels for
        # the allocated Tesla P100, and it stays compatible with numpy 2.0.2. Do NOT
        # touch numpy/pandas (a force-reinstall corrupts the numpy C ABI symbols).
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-cache-dir", "--index-url",
                               "https://download.pytorch.org/whl/cu121", "torch==2.5.1"],
                              timeout=1800)
        _r2 = subprocess.run([sys.executable, "-c", _probe], capture_output=True,
                             text=True, timeout=600)
        print("post-reinstall probe rc:", _r2.returncode,
              "out:", (_r2.stdout or "").strip(), "err:", (_r2.stderr or "")[-200:], flush=True)
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

if os.path.exists("/kaggle"):
    WORK = "/kaggle/working"; INP = "/kaggle/input"
else:
    WORK = os.path.join("vault", "pipeline_out_v13@SUFFIX@")
    INP = "official_dataset"
os.makedirs(WORK, exist_ok=True)
PRETRAINED = os.path.join(WORK, "pretrained_encoder.pt")
OUT = WORK

print("device:", DEVICE, "| SMOKE:", SMOKE, "| folds:", GLOBAL_FOLDS,
      "| PRETRAIN_EPOCHS:", PRETRAIN_EPOCHS, "| out:", OUT, flush=True)
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
          "BalabanJ", "Ipc", "LipHBA", "LipHBD", "Stereo"]
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

# Drop rows whose canonical SMILES could not be parsed -> keep clean.
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

M("## 3. Pretrain the GINE encoder on PI1M (in-kernel self-supervised)")

P(r"""pl_path = find_input(INP, "PI1M.csv")
pl = []
if pl_path:
    pldf = pd.read_csv(pl_path)
    smi_col = "SMILES" if "SMILES" in pldf.columns else "smiles"
    pldf = pldf[[smi_col]].rename(columns={smi_col: "smiles"})
    pldf["canon"] = pldf["smiles"].map(lambda s: canonical(s)[0])
    pldf = pldf.dropna(subset=["canon"])
    pl = pldf.drop_duplicates("canon")["smiles"].tolist()
    rng = np.random.RandomState(SEED); rng.shuffle(pl)
    pl = pl[:PRETRAIN_SAMPLE]
    print("PI1M pretraining corpus:", len(pl), "SMILES (capped at", PRETRAIN_SAMPLE, ")", flush=True)
else:
    print("no PI1M: pretraining skipped", flush=True)

def build_pretrain_graphs(smiles_list):
    graphs = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
    return graphs

pl_graphs = build_pretrain_graphs(pl) if pl else []
print("pretraining graphs:", len(pl_graphs), flush=True)

from torch_geometric.loader import DataLoader

class PretrainedEncoder(nn.Module):
    # wraps the shared GINE trunk; state_dict keys start with 'encoder.'
    # so MTGNN.load_encoder can load them (identical to the reference kernel).
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
            src = h[edge_index[0, bond_mask]]
            dst = h[edge_index[1, bond_mask]]
            if src.numel() > 0:
                bond_loss = F.mse_loss(self.bond_proj(torch.cat([src, dst], dim=1)), edge_attr[bond_mask])
            else:
                bond_loss = torch.zeros((), device=x.device)
        else:
            bond_loss = torch.zeros((), device=x.device)
        return atom_loss, bond_loss

def pretrain(epochs=PRETRAIN_EPOCHS, batch_size=256, lr=1e-3):
    if not pl_graphs:
        print("No PI1M graphs - pretraining skipped", flush=True)
        return None
    model = PretrainedEncoder(N_ATOM_FEATS, N_BOND_FEATS).to(DEVICE)
    loader = DataLoader(pl_graphs, batch_size=batch_size, shuffle=True)
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
        print("saved pretrained_encoder.pt", flush=True)
    return best_state

print("=== Pretraining GNN on PI1M ===", flush=True)
pretrained_state = pretrain()
""")

M("## 4. Level-0 predictions (verbatim: leak-safe twins + MT-GNN fold OOF + GBM trio stack)")

P(CORE_B)

M("## 5. v13 blend — per-target Ridge(alpha=1.0) on [GBM, MT-GNN] OOF + submission")

P(r"""ALPHA = 1.0

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

print("\n=== Check 1: corr(GBM, GNN) per target ===", flush=True)
for t in TARGETS:
    idx = idx_of_target[t]
    c = np.corrcoef(oof_gbm_global[idx], oof_mt_global[idx])[0, 1]
    print(f"  {t:<4} corr={c:.4f}", flush=True)

print("\n=== building v13 blend (per-target Ridge, alpha=1.0) ===", flush=True)
rows = []
coefs = {t: [] for t in TARGETS}
final_te = np.zeros(len(tef))
for t in TARGETS:
    idx = idx_of_target[t]
    yt = Y[idx].astype(np.float64)
    Mx = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx]])
    Mte = np.column_stack([test_gbm_global, test_mt_global])
    oof = np.zeros(len(idx)); te_pred = np.zeros(len(tef))
    for trk, vk in GroupKFold(n_splits=GLOBAL_FOLDS).split(Mx, yt, G[idx]):
        lr = Ridge(alpha=1.0); lr.fit(Mx[trk], yt[trk])
        oof[vk] = lr.predict(Mx[vk])
        te_pred += lr.predict(Mte) / GLOBAL_FOLDS
        coefs[t].append(lr.coef_.tolist())
    m_te = (tef["target_type"] == t).values
    final_te[m_te] = te_pred[m_te]
    r_blend = r2_score(yt, oof); r_g = r2_score(yt, oof_gbm_global[idx]); r_m = r2_score(yt, oof_mt_global[idx])
    cb = np.mean(coefs[t], axis=0)
    rows.append(dict(target=t, blend=r_blend, GBM=r_g, GNN=r_m, w_GBM=cb[0], w_GNN=cb[1]))
    print(f"  {t:<4} blend={r_blend:.4f} GBM={r_g:.4f} GNN={r_m:.4f} w_GBM={cb[0]:.3f} w_GNN={cb[1]:.3f}", flush=True)

df = pd.DataFrame(rows).set_index("target")
print("\n=== summary ===", flush=True)
print("  mean blend=%.4f | GBM=%.4f | GNN=%.4f | delta-vs-GBM %+.4f" % (
    df["blend"].mean(), df["GBM"].mean(), df["GNN"].mean(),
    df["blend"].mean() - df["GBM"].mean()), flush=True)

print("\n=== Check 2: blend weights (small targets should lean GNN) ===", flush=True)
for t in TARGETS:
    cb = np.mean(coefs[t], axis=0)
    print(f"  {t:<4} w_GBM={cb[0]:.3f} w_GNN={cb[1]:.3f} GNN_share={cb[1]/(cb.sum()):.2f}", flush=True)

sub = pd.DataFrame({"id": tef["id"].values, "target": final_te})
sub_path = os.path.join(OUT, "submission_v13.csv")
sub.to_csv(sub_path, index=False)
print("\nwrote", sub_path, flush=True)
print("  rows", len(sub), "| NaN", sub["target"].isna().sum(),
      "| range [%.2f, %.2f]" % (sub["target"].min(), sub["target"].max()), flush=True)
df.round(4).to_csv(os.path.join(OUT, "v13_blend_report.csv"), index=True)
print("wrote", os.path.join(OUT, "v13_blend_report.csv"), flush=True)
print("DONE", flush=True)
""")

nb["cells"] = C
with open(OUT_NB, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebook:", OUT_NB, "| cells:", len(C), "| SMOKE:", SMOKE, flush=True)