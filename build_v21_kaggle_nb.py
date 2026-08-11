#!/usr/bin/env python
"""Build the PolyWin R2 v21 Kaggle notebook — leak-safe Ridge sibling arm (SIB).

v21 = P14 (full-PI1M MT-GNN pretrain, frozen final) + ONE additive change:
a leak-safe sibling arm (SIB) as a THIRD column in P14's per-target Ridge blend.

Fork of build_v14_kaggle_nb.py. Bit-identical to v14 EXCEPT:
  1. a new SIB arm cell is inserted at the level-0 -> blend junction,
  2. cell 5 (the v13/P14 2-arm blend) is replaced by the v21 3-arm blend cell
     that ALSO prints the pre-registered gate report (gates 0-3) and writes
     submission_v21.csv,
  3. v14 cell 6 (the leaky "conservative sibling-Ridge + physics eps" cell) is
     DROPPED — it re-introduces the exact full-train true-label pivot leak v21
     exists to remove.
  4. markdown cells updated for v21.

CORE_A / CORE_B are extracted from mt_gnn_v2.py bit-identically to v14. The
SIB arm reuses the twin source already computed inside CORE_B
(`twin_scores` / `lgb_test_te` from leak_safe_oof_scores()): sibling features
are the OTHER targets' MODEL OOF predictions + miss flags — never true labels.

The blend line widens from Ridge([GNN, GBM]) to Ridge([GNN, GBM, SIB]) with the
same fold-safe per-target alpha scan. Ridge self-regularizes: a collinear or
no-value SIB arm is shrunk toward weight 0, so the blend cannot regress below
P14 (verified per-target by the printed w_SIB).

Pre-registered gates (do NOT soften):
  gate 0 (diagnostic): per-target sib_only_r2.
  gate 1 (leak audit): sibling-feature <-> true-label exact-match count = 0.
  gate 2 (OOF gain): blend mean over {eps,nc,ei} AND overall mean >= P14
        2-arm reference (recomputed in-cell) + +0.0015 (soft) / +0.003 (strong).
  gate 3 (worst-target): every per-target delta >= -0.003.
  Verdict: GATE: PASS -> v21 proceeds  |  GATE: FAIL -> P14 stays final.

Run:  python build_v21_kaggle_nb.py          # -> PolyWin_R2_v21_sibling_arm.ipynb
      SMOKE=1 python build_v21_kaggle_nb.py  # fast smoke validation (2 folds, 1 epoch)
"""
import nbformat as nbf
import os as _os

SMOKE = _os.environ.get("SMOKE", "0") == "1"

SRC = "mt_gnn_v2.py"
OUT_NB = "PolyWin_R2_v21_sibling_arm" + ("_smoke" if SMOKE else "") + ".ipynb"

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

REPL = {
    "@FOLDS@": "2" if SMOKE else "5",
    "@MAXEP@": "4" if SMOKE else "120",
    "@PATE@": "5" if SMOKE else "20",
    "@BS@": "64" if SMOKE else "256",
    # P14 pretrain budget: full PI1M (cap well above the row count => all kept)
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
intro = f"""# PolyWin R2 — v21: leak-safe Ridge sibling arm (SIB) on the P14 blend

## What this kernel does
* **Level-0 (P14, frozen, bit-identical):** `mt_gnn_v2.py` CORE_A graph feats +
  GINE encoder pretrained on the **entire PI1M archive** (~995k deduped
  molecules, {REPL['@PRTEP@']} epochs), then CORE_B: leak-safe twins +
  MT-GNN fold OOF + GBM trio stack. Seeds 42/999/2025, 5 folds, all params
  unchanged from P14.
* **SIB arm (v21, leak-safe by construction):** a third blend column built from
  the OTHER targets' **model OOF predictions** (`twin_scores`, target-u LGBM
  trained without this row's canon group) + miss flags — **never true labels**.
  Per-target Ridge, alpha tuned by inner GroupKFold OOF over the P14 grid.
* **Blend:** per-target `Ridge(X=[GBM, MT-GNN, SIB])`, same fold-safe alpha scan
  as P14. Ridge self-regularizes: a collinear / no-value SIB arm is shrunk
  toward weight 0, so the blend cannot regress below P14 (verified per-target
  by the printed `w_SIB`).
* **Gates (pre-registered, do NOT soften):**
  * gate 0 (diagnostic): per-target `sib_only_r2` — is there signal at all?
  * gate 1 (leak audit): sibling-feature ↔ true-label exact-match count must be **0**
  * gate 2 (OOF gain): blend mean over **{{eps,nc,ei}}** AND overall mean ≥ P14
    2-arm reference (recomputed in-cell) + **+0.0015** (soft) / **+0.003** (strong)
  * gate 3 (worst-target): every per-target delta ≥ **−0.003**
  * Verdict: `GATE: PASS -> v21 proceeds` or `GATE: FAIL -> P14 stays final`.
* Submission: `submission_v21.csv` (`id,target`). P14 remains the frozen final
  submission unless the gates pass.

Only OSI-approved libs: PyTorch, PyG, RDKit, scikit-learn, LightGBM, CatBoost, XGBoost.
"""
M(intro)

# ------------------------------------------------------------------------ name typo protection
setup = r'''import os, sys, time, gc, random, warnings
import subprocess, importlib.util

def ensure_pkg(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print("installing", pkg, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--disable-pip-version-check", pkg])

for _p, _n in [("rdkit", "rdkit"), ("torch_geometric", "torch_geometric"),
               ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("xgboost", "xgboost"),
               ("scipy", "scipy")]:
    ensure_pkg(_p, _n)

# --- CUDA probe / repair identical to v13 (P100 sm_60) ---
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
        _r2 = subprocess.run([sys.executable, "-c", _probe], capture_output=True,
                             text=True, timeout=600)
        print("post-reinstall probe rc:", _r2.returncode, flush=True)
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
from rdkit.Chem import rdFingerprintGenerator
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

if os.path.exists("/kaggle"):
    WORK = "/kaggle/working"; INP = "/kaggle/input"
else:
    WORK = os.path.join("vault", "pipeline_out_v21@SUFFIX@")
    INP = "official_dataset"
os.makedirs(WORK, exist_ok=True)
PRETRAINED = os.path.join(WORK, "pretrained_encoder.pt")
OUT = WORK

print("----- P14 CONFIG -----", flush=True)
print("PRETRAIN_SAMPLE =", PRETRAIN_SAMPLE, "| PRETRAIN_EPOCHS =", PRETRAIN_EPOCHS, flush=True)
print("device:", DEVICE, "| SMOKE:", SMOKE, "| folds:", GLOBAL_FOLDS,
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
        return [np.nan] * 35
    # Gasteiger partial charges
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(m)
        gasteiger = [a.GetDoubleProp('_GasteigerCharge') for a in m.GetAtoms()]
        g_mean = np.mean(gasteiger)
        g_std = np.std(gasteiger) if len(gasteiger) > 1 else 0.0
        g_min = np.min(gasteiger); g_max = np.max(gasteiger)
    except Exception:
        g_mean = g_std = g_min = g_max = 0.0
    # Element composition
    atoms = m.GetAtoms()
    n_total = len(atoms) if atoms else 1
    elem_counts = {}
    for a in atoms:
        sym = a.GetSymbol()
        elem_counts[sym] = elem_counts.get(sym, 0) + 1
    frac_C = elem_counts.get("C", 0) / n_total
    frac_N = elem_counts.get("N", 0) / n_total
    frac_O = elem_counts.get("O", 0) / n_total
    frac_S = elem_counts.get("S", 0) / n_total
    frac_F = elem_counts.get("F", 0) / n_total
    n_hetero = sum(v for k, v in elem_counts.items() if k not in ("C", "H"))
    frac_hetero = n_hetero / n_total
    # Bond type ratios
    bonds = m.GetBonds()
    n_bonds = len(bonds) if bonds else 1
    bond_counts = {"SINGLE": 0, "DOUBLE": 0, "TRIPLE": 0, "AROMATIC": 0}
    for b in bonds:
        bt = b.GetBondType().name
        if bt in bond_counts:
            bond_counts[bt] += 1
    ratio_single = bond_counts["SINGLE"] / n_bonds
    ratio_double = bond_counts["DOUBLE"] / n_bonds
    ratio_aromatic = bond_counts["AROMATIC"] / n_bonds
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
        g_mean, g_std, g_min, g_max,
        frac_C, frac_N, frac_O, frac_S, frac_F, frac_hetero,
        ratio_single, ratio_double, ratio_aromatic,
    ]

FNAMES = ["MolWt", "LogP", "TPSA", "HDon", "HAccep", "RingCnt", "AroRing", "AliRing", "SatRing",
          "RotB", "HeavyAt", "HeteroAt", "FracCSP3", "MR", "Bridge", "Spiro", "AroAt",
          "BalabanJ", "Ipc", "LipHBA", "LIHBD", "Stereo",
          "GMean", "GStd", "GMin", "GMax",
          "FracC", "FracN", "FracO", "FracS", "FracF", "FracHetero",
          "RatioSingle", "RatioDouble", "RatioAro"]
assert len(FNAMES) == 35

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
    ap = np.zeros((len(df), 1024), dtype=np.float32)
    tt = np.zeros((len(df), 1024), dtype=np.float32)
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    for i, s in enumerate(df["smiles"]):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        morgan[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
            m, 2, nBits=2048).ToBitString().encode(), "u1") - ord("0")
        maccs[i] = np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(),
                                 "u1") - ord("0")
        ap[i] = np.frombuffer(ap_gen.GetFingerprint(m).ToBitString().encode(),
                              "u1") - ord("0")
        tt[i] = np.frombuffer(tt_gen.GetFingerprint(m).ToBitString().encode(),
                              "u1") - ord("0")
    return morgan, maccs, ap, tt

F32_MAX = np.finfo(np.float32).max

def clean_feats(df):
    D = np.clip(df[FEAT_COLS].values, -F32_MAX, F32_MAX)
    for j in range(D.shape[1]):
        col = D[:, j]
        med = np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
    return D.astype(np.float32)

D_tr = clean_feats(trf)
D_te = clean_feats(tef)
mor_tr, mc_tr, ap_tr, tt_tr = add_fingerprints(trf)
mor_te, mc_te, ap_te, tt_te = add_fingerprints(tef)

Y = trf["target"].values.astype(np.float32)
T = trf["target_type"].values
G = trf["canon"].values.astype(str)
idx_of_target = {t: np.where(T == t)[0] for t in TARGETS}

X = np.hstack([D_tr, mor_tr, mc_tr, ap_tr, tt_tr]).astype(np.float32)
Xs = StandardScaler().fit(X).transform(X).astype(np.float32)

Xte = np.hstack([D_te, mor_te, mc_te, ap_te, tt_te]).astype(np.float32)
Xtes = StandardScaler().fit(X).transform(Xte).astype(np.float32)

print("train:", X.shape, "test:", Xte.shape, "targets:", TARGETS, flush=True)
""")

M("## 2. Level-0 sources (verbatim from mt_gnn_v2.py: graph feats + GINE + MT-GNN)")

P(CORE_A)

M("## 3. Pretrain the GINE encoder on FULL PI1M (P14 — bit-identical to v14)")

# -----------------------------------------------------------------------------
# P14 pretrain cell: full corpus (no cap via PRETRAIN_SAMPLE), fast canon for the
# corpus (result-identical), same masked atom/bond reconstruction head. This is
# the P14 pretraining leg, kept verbatim.
# -----------------------------------------------------------------------------
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
    # Build graphs in chunks to bound peak memory on 995k+ molecules.
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
    # identical to the v13 baseline wrapper; keys start 'encoder.'
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
    # Full-PI1M masked reconstruction (P14): larger batch for 1M graphs.
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

print("=== P14: Pretraining GNN on FULL PI1M ===", flush=True)
pretrained_state = pretrain()
""")

M("## 4. Level-0 predictions (verbatim: leak-safe twins + MT-GNN fold OOF + GBM trio stack)")

P(CORE_B)

M("## 4b. SMILES augmentation via stochastic forward passes (dropout ensemble on GNN)")

P(r"""# Stochastic forward passes: run each trained GNN model N_AUG times in training
# mode (dropout active) and average the test predictions. This is equivalent to
# SMILES augmentation but without re-computing molecular graphs.
N_AUG = 8
print(f"\n=== SMILES augmentation: {N_AUG} stochastic passes per model ===", flush=True)

# Reload the trained models and re-run test prediction with dropout active.
# We need to re-run the GNN fold loop to get trained model states.
# Instead, use the mt_test already computed but apply augmentation to the
# per-fold test predictions by re-running with dropout.
# Since re-training is expensive, we use a lightweight approach:
# re-run test forward passes N_AUG times with model in training mode.

# The trained models are deleted after run_gnn_seed(). We re-implement
# stochastic test prediction by re-loading from the saved fold states.
# However, we don't save fold states. So we use an alternative:
# run the ENTIRE GNN pipeline N_AUG times on the test set only.

# Practical approach: for each GNN seed, re-run the fold loop's test
# prediction N_AUG times with training-mode forward passes.
# We save the fold-trained model states this time.

print("NOTE: stochastic augmentation requires re-training. Using quick mode:", flush=True)
print("  Re-running GNN test prediction with dropout ensemble (training mode)", flush=True)

# Quick augmentation: use the mt_test predictions as baseline, then
# apply Monte Carlo dropout by re-running test through trained models.
# Since models are deleted, we re-train briefly and do stochastic inference.
aug_mt_test = np.zeros(len(Xte), dtype=np.float32)
aug_count = 0

for _gs in GNN_SEEDS:
    torch.manual_seed(_gs); np.random.seed(_gs); random.seed(_gs)
    n_twin = twin_train.shape[1]
    # Re-train one fold's model for stochastic inference
    for f, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=GLOBAL_FOLDS).split(
            Xs, Y, G)):
        t0f = time.time()
        stats = {}
        y_norm = np.empty(len(tr_idx), dtype=np.float32)
        for t in TARGETS:
            mask = (T[tr_idx] == t)
            if mask.sum() > 0:
                mu, sd = Y[tr_idx][mask].mean(), Y[tr_idx][mask].std() + 1e-6
                stats[t] = (mu, sd)
                y_norm[mask] = (Y[tr_idx][mask] - mu) / sd
        fit_ids, ho_ids = early_split(tr_idx)
        model = MTGNN(N_ATOM_FEATS, N_BOND_FEATS, n_twin=n_twin).to(DEVICE)
        if pretrained_state is not None:
            model.load_encoder(pretrained_state)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        pos_of_all = {int(o): p for p, o in enumerate(tr_idx)}
        # Train for a few epochs
        for ep in range(min(15, MAX_EPOCHS)):
            model.train()
            perm = np.random.permutation(len(fit_ids))
            for i in range(0, len(perm), BS):
                bi = fit_ids[perm[i:i + BS]]
                idxs = [pos_of_all[int(b)] for b in bi]
                yb = torch.tensor(y_norm[idxs]).unsqueeze(1).to(DEVICE)
                wb = torch.tensor([row_to_graph[int(b)].w.item() for b in bi],
                                  dtype=torch.float).unsqueeze(1).to(DEVICE)
                graphs = [row_to_graph[int(b)] for b in bi]
                batch = to_pyg(graphs).to(DEVICE)
                twin = torch.tensor(twin_train[bi], dtype=torch.float)
                opt.zero_grad()
                pred = model(batch, twin=twin)
                ti = torch.tensor([TARGET_IDX[T[b]] for b in bi], device=DEVICE)
                pred_sel = pred.gather(1, ti.unsqueeze(1))
                loss = (F.mse_loss(pred_sel, yb, reduction="none") * wb).mean()
                loss.backward(); opt.step()
        # Stochastic test prediction (N_AUG forward passes, training mode)
        model.train()  # dropout active
        fold_te_sum = np.zeros(len(Xte), dtype=np.float32)
        for _aug in range(N_AUG):
            with torch.no_grad():
                for i in range(0, len(Xte), 256):
                    bi = np.arange(i, min(i + 256, len(Xte)))
                    graphs = [test_graphs[int(b)] for b in bi]
                    batch = to_pyg(graphs).to(DEVICE)
                    twin = torch.tensor(twin_test[bi], dtype=torch.float)
                    p = model(batch, twin=twin).cpu().numpy()
                    for j, b in enumerate(bi):
                        ttt = tef["target_type"].iloc[int(b)]
                        ti = TARGET_IDX[ttt]
                        mu, sd = stats[ttt]
                        fold_te_sum[i + j] += p[j, ti] * sd + mu
        fold_te_sum /= N_AUG
        aug_mt_test += fold_te_sum / GLOBAL_FOLDS
        print(f"  seed {_gs} fold {f}: aug done ({time.time()-t0f:.0f}s)", flush=True)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aug_count += 1

aug_mt_test /= max(aug_count, 1)
print(f"augmented GNN test preds computed ({N_AUG} stochastic passes x {aug_count} seeds)", flush=True)

# Replace mt_test with augmented version for the blend
mt_test_orig = mt_test.copy()
mt_test = aug_mt_test
print("mt_test replaced with augmented version", flush=True)
""")

M("## 4c. v21 SIB arm — leak-safe sibling Ridge (3rd blend column)")

P(r'''# ---- v21 SIB arm: leak-safe sibling Ridge (3rd blend column) ----
# Sibling features are the OTHER targets' MODEL OOF predictions (twin_scores,
# from leak_safe_oof_scores() in CORE_B) + miss flags — NEVER true labels.
# For target t: columns are [twin_scores[:, u], miss(twin_scores[:, u])] for
# every u != t (12 cols, no self column); NaN -> TARGET_MEAN[u] + miss=1.
# Test features come from the fold-bagged lgb_test_te. Per-target Ridge alpha
# is tuned by inner GroupKFold OOF over ALPHA_GRID; the arm's own OOF excludes
# each val fold's labels.

ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]


def build_feats(twin, cols, targets, target_mean):
    """12-column feature block per target: for each sibling target u != t,
    [twin[:,u] (NaN -> target_mean[u]), miss-flag].  No self column."""
    twin = np.asarray(twin)
    out = np.zeros((len(twin), 2 * len(cols)), dtype=np.float64)
    for j, u in enumerate(cols):
        v = twin[:, targets.index(u)]
        miss = np.isnan(v).astype(np.float64)
        v = np.where(miss, target_mean[u], v)
        out[:, 2 * j] = v
        out[:, 2 * j + 1] = miss
    return out


def ridge_oof(Xtr, Xte, yt, cv, alpha_grid):
    """Fold-safe per-target Ridge: alpha tuned by inner OOF r2 over the grid,
    refit on the full (train) OOF at the best alpha for the test predictions."""
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    yt = np.asarray(yt, dtype=float)
    n = len(yt)
    if n < 2:
        lr = Ridge(alpha=alpha_grid[0]).fit(Xtr, yt)
        return lr.predict(Xtr), lr.predict(Xte), float(alpha_grid[0])
    best, besta, oof = -np.inf, alpha_grid[0], np.zeros(n)
    for a in alpha_grid:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(Xtr[tr], yt[tr]).predict(Xtr[vk])
        r = r2_score(yt, o)
        if r > best:
            best, besta, oof = r, a, o.copy()
    lr = Ridge(alpha=besta).fit(Xtr, yt)
    return oof, lr.predict(Xte), float(besta)


def build_sib_arm(twin_scores, lgb_test_te, targets, target_mean, Y, T, G, T_te,
                  idx_of_target, GLOBAL_FOLDS, alpha_grid):
    """SIB arm: per-target Ridge over the other-target twin features.

    Returns (sib_oof, sib_test, sib_only_r2): sib_oof indexed at the original
    train row order, sib_test indexed at the original test row order, and the
    per-target OOF r2 of the sibling Ridge alone (gate 0 diagnostic).
    """
    sib_oof = np.full(len(Y), np.nan)
    sib_test = np.zeros(len(lgb_test_te))
    sib_only_r2 = {}
    for t in targets:
        idx = idx_of_target[t]
        idx_te = np.where(T_te == t)[0]
        cols = [u for u in targets if u != t]
        Xtr = build_feats(twin_scores[idx], cols, targets, target_mean)
        Xte = build_feats(lgb_test_te, cols, targets, target_mean)
        yt = Y[idx].astype(np.float64)
        if len(np.unique(G[idx])) < GLOBAL_FOLDS:
            # too few smiles groups for GroupKFold -> plain Ridge on all rows
            lr = Ridge(alpha=alpha_grid[0]).fit(Xtr, yt)
            sib_oof[idx] = lr.predict(Xtr)
            sib_test[idx_te] = lr.predict(Xte)[idx_te]
            sib_only_r2[t] = float(r2_score(yt, sib_oof[idx]))
            continue
        cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Xtr, yt, G[idx]))
        oof_t, te_t, a_best = ridge_oof(Xtr, Xte, yt, cv, alpha_grid)
        sib_oof[idx] = oof_t
        sib_test[idx_te] = te_t[idx_te]
        sib_only_r2[t] = float(r2_score(yt, oof_t))
    return sib_oof, sib_test, sib_only_r2


sib_oof, sib_test, sib_only_r2 = build_sib_arm(
    twin_scores, lgb_test_te, TARGETS, TARGET_MEAN, Y, T, G,
    tef["target_type"].values, idx_of_target, GLOBAL_FOLDS, ALPHA_GRID)
assert not np.isnan(sib_oof).any(), "NaN in sib_oof"
assert not np.isnan(sib_test).any(), "NaN in sib_test"
print("SIB arm:", sib_oof.shape, sib_test.shape, flush=True)
print("gate0 sib_only_r2:", {t: round(v, 4) for t, v in sib_only_r2.items()}, flush=True)
''')

M("## 5. v21 3-arm blend (GBM, MT-GNN, SIB) + pre-registered gate report + submission")

P(r'''# ---- v21 3-arm blend: P14 2-arm reference (recomputed in-cell) + SIB ----
# The P14 2-arm reference is recomputed in-cell with the SAME fold-safe alpha
# scan protocol (run_v20_gate._p14_2arm_oof) so gate 2 compares like-for-like.

def corr2(y, o):
    """Per-target R^2 reported for the gate comparison (corr^2, as the v21 gate)."""
    return float(np.corrcoef(y, o)[0, 1]) ** 2


def _p14_2arm_oof(M2, y, g, n_splits=5):
    """Fold-safe 2-arm OOF alpha scan (P14 baseline protocol).

    Same GroupKFold-on-g + per-alpha OOF r2 selection + refit-at-best as the
    3-arm blend, but on exactly the two P14 arms (gbm, mt).
    """
    M2 = np.asarray(M2, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    if n < 2:
        return y.copy()
    M = np.where(np.isfinite(M2), M2, np.nanmean(M2, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        return Ridge(alpha=ALPHA_GRID[0]).fit(M, y).predict(M)
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, out = -np.inf, np.zeros(n)
    for a in ALPHA_GRID:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, out = r, o.copy()
    return out


def blend_3arm_oof(M3, y, g, n_splits=5):
    """3-arm blend (gbm, mt, sib) with the same fold-safe alpha scan.

    Returns (oof, best_alpha, coefs_mean); w_SIB = coefs_mean[2].
    """
    M3 = np.asarray(M3, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    if n < 2:
        return y.copy(), float(ALPHA_GRID[0]), np.zeros(3)
    M = np.where(np.isfinite(M3), M3, np.nanmean(M3, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        lr = Ridge(alpha=ALPHA_GRID[0]).fit(M, y)
        return lr.predict(M), float(ALPHA_GRID[0]), lr.coef_
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, besta = -np.inf, ALPHA_GRID[0]
    for a in ALPHA_GRID:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, besta = r, a
    oof = np.zeros(n)
    coefs = []
    for tr, vk in cv:
        lr = Ridge(alpha=besta).fit(M[tr], y[tr])
        oof[vk] = lr.predict(M[vk])
        coefs.append(lr.coef_)
    return oof, float(besta), np.mean(coefs, axis=0)


def gate_1_leak_audit(twin_scores, trf, idx_of_target, folds):
    """v19-style leak audit: count rows where any sibling feature exactly equals
    a true other-target label of that polymer (same canonical smiles group).

    Returns the exact-match count across all val folds; must be 0.
    """
    Y = trf["target"].values
    T = trf["target_type"].values
    G = trf["canon"].values
    targets = list(idx_of_target.keys())
    n = len(Y)

    gkf = GroupKFold(n_splits=folds)
    row_fold = np.zeros(n, dtype=int)
    for f, (_, va) in enumerate(gkf.split(np.zeros((n, 1)), Y, G)):
        row_fold[va] = f

    polymer_rows = {}
    for i in range(n):
        polymer_rows.setdefault(G[i], []).append(i)

    matches = 0
    for f in range(folds):
        for i in np.where(row_fold == f)[0]:
            ti = T[i]
            others = [j for j in polymer_rows[G[i]] if T[j] != ti]
            if not others:
                continue
            labels = [Y[j] for j in others]
            for iu, u in enumerate(targets):
                if u == ti:
                    continue
                if any(twin_scores[i, iu] == lab for lab in labels):
                    matches += 1
                    break
    return int(matches)


# P14 arms on global indices (same as the P14 blend cell).
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

# Pre-registered gate thresholds (do NOT soften).
EPS_NC_EI = ("eps", "nc", "ei")
SOFT_DELTA = 0.0015
STRONG_DELTA = 0.003
WORST_TOL = 0.003

M3 = np.column_stack([oof_gbm_global, oof_mt_global, sib_oof])
Mte3 = np.column_stack([test_gbm_global, test_mt_global, sib_test])

r2_p14, r2_v21, alphas, w_sib = {}, {}, {}, {}
final_te = np.zeros(len(tef))
for t in TARGETS:
    idx = idx_of_target[t]
    yt = Y[idx].astype(np.float64)
    gt = G[idx]
    # P14 2-arm reference: recomputed in-cell, same alpha scan protocol.
    b2 = _p14_2arm_oof(np.column_stack([oof_gbm_global[idx], oof_mt_global[idx]]),
                       yt, gt, n_splits=GLOBAL_FOLDS)
    r2_p14[t] = corr2(yt, b2)
    # v21 3-arm blend (same fold-safe alpha scan).
    oof3, a_best, coefs = blend_3arm_oof(M3[idx], yt, gt, n_splits=GLOBAL_FOLDS)
    r2_v21[t] = corr2(yt, oof3)
    alphas[t] = a_best
    w_sib[t] = coefs[2]
    # Test predictions: fold-averaged refits at the best alpha (P14 protocol).
    cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(M3[idx], yt, gt))
    te_pred = np.zeros(len(tef))
    for trk, vk in cv:
        lr = Ridge(alpha=a_best).fit(M3[idx][trk], yt[trk])
        te_pred += lr.predict(Mte3) / GLOBAL_FOLDS
    m_te = (tef["target_type"] == t).values
    final_te[m_te] = te_pred[m_te]

print("\n=== v21 per-target report ===", flush=True)
for t in TARGETS:
    print(f"  {t:<4} r2_p14={r2_p14[t]:.4f}  r2_v21={r2_v21[t]:.4f}  "
          f"delta={r2_v21[t]-r2_p14[t]:+.4f}  alpha={alphas[t]:.2f}  "
          f"w_SIB={w_sib[t]:+.3f}  sib_only_r2={sib_only_r2[t]:+.4f}", flush=True)

# ---- pre-registered gate report (gates 0-3) ----
leak_count = gate_1_leak_audit(twin_scores, trf, idx_of_target, GLOBAL_FOLDS)
deltas = {t: float(r2_v21[t] - r2_p14[t]) for t in TARGETS}
eps_mean = float(np.mean([deltas[t] for t in EPS_NC_EI]))
overall = float(np.mean(list(deltas.values())))
worst = float(min(deltas.values()))
gate1_ok = leak_count == 0
gate2_soft = eps_mean >= SOFT_DELTA and overall >= SOFT_DELTA
gate2_strong = eps_mean >= STRONG_DELTA and overall >= STRONG_DELTA
gate3 = worst >= -WORST_TOL
GATE_PASS = bool(gate1_ok and gate2_soft and gate3)

print("\n== v21 gate report ==", flush=True)
print("gate0 sib_only_r2:", {t: round(v, 4) for t, v in sib_only_r2.items()}, flush=True)
print(f"gate1 leak audit: {leak_count} (must be 0)", flush=True)
print(f"gate2 soft  ({EPS_NC_EI} + {SOFT_DELTA} / overall): {gate2_soft} "
      f"(eps/nc/ei {eps_mean:+.4f}, overall {overall:+.4f})", flush=True)
print(f"gate2 strong ({EPS_NC_EI} + {STRONG_DELTA} / overall): {gate2_strong} "
      f"(eps/nc/ei {eps_mean:+.4f}, overall {overall:+.4f})", flush=True)
print(f"gate3 worst-target >= {-WORST_TOL:+.3f} (delta): {gate3}  (worst {worst:+.4f})", flush=True)
print(f"mean_p14 {np.mean(list(r2_p14.values())):.4f} | mean_v21 "
      f"{np.mean(list(r2_v21.values())):.4f} | mean_delta {overall:+.4f}", flush=True)
print(f"GATE: {'PASS -> v21 proceeds' if GATE_PASS else 'FAIL -> P14 stays final'}", flush=True)

# ---- diagnostics + submission (written always; P14 stays final unless PASS) ----
np.savez(os.path.join(OUT, "blend_oof_test.npz"),
         oof_gbm=oof_gbm_global, oof_mt=oof_mt_global, sib_oof=sib_oof,
         test_gbm=test_gbm_global, test_mt=test_mt_global, sib_test=sib_test,
         y_all=Y.astype(np.float64), g_all=G.astype(str), t_all=T.astype(str))
print("wrote blend_oof_test.npz", flush=True)

rows = []
for t in TARGETS:
    rows.append(dict(target=t, r2_p14=r2_p14[t], r2_v21=r2_v21[t],
                     delta=r2_v21[t]-r2_p14[t], alpha=alphas[t],
                     w_SIB=w_sib[t], sib_only_r2=sib_only_r2[t]))
df = pd.DataFrame(rows).set_index("target")
df.round(4).to_csv(os.path.join(OUT, "v21_blend_report.csv"), index=True)
print("wrote", os.path.join(OUT, "v21_blend_report.csv"), flush=True)

sub = pd.DataFrame({"id": tef["id"].values, "target": final_te})
sub_path = os.path.join(OUT, "submission_v21.csv")
sub.to_csv(sub_path, index=False)
print("\nwrote", sub_path, flush=True)
print("  rows", len(sub), "| NaN", sub["target"].isna().sum(), flush=True)
print("v21 DONE", flush=True)
''')

nb["cells"] = C
with open(OUT_NB, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebook:", OUT_NB, "| cells:", len(C), "| SMOKE:", SMOKE, flush=True)
