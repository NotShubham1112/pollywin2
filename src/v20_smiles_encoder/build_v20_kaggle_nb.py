#!/usr/bin/env python
"""Build the PolyWin R2 v20 self-contained Kaggle notebook.

The v20 notebook mirrors the v14 P14 kernel structure (IMPORTS -> DATA ->
CORE_A -> GINE pretrain on PI1M -> CORE_B) but:

  * RECOMPUTES the P14 arms (oof_gbm/oof_mt/test_gbm/test_mt) IN-KERNEL from
    train.csv/test.csv/PI1M.csv ONLY (no superblend_oof.npz, no externals),
  * inlines the v20 module sources (codec/encoder/arm_cv/blend/gate_report)
    verbatim so the Kagle kernel is fully self-contained,
  * adds the self-trained SMILES-encoder arm (oof_trf/test_trf), blends the
    three arms per target with the P14 fold-safe alpha sweep, evaluates the
    pre-registered gate, and writes submission.csv ONLY on PASS (else prints
    GATE=FAIL -> P14 stays final).

Runs TWO GBMs? No -- CORE_A/CORE_B are the SAME level-0 sources as P14
(mt_gnn_v2.py). The gate runs natively in the kernel.

Run:
    python build_v20_kaggle_nb.py          # FULL config -> vault/kernel-v20-embed/
    SMOKE=1 python build_v20_kaggle_nb.py  # fast smoke validation
"""
import os
from pathlib import Path

import nbformat as nbf

_src_dir = Path(__file__).resolve().parent
SRC = _src_dir / "src/core/mt_gnn_v2.py"
OUT_DIR = _src_dir / "vault" / "kernel-v20-embed"
OUT_NB_NAME = "PolyWin_R2_v20_embed_submit.ipynb"


def _read_detail(p):
    return (p.read_text(encoding="utf-8")).splitlines()


def _idx(lines, marker):
    for i, line in enumerate(lines):
        if marker in line:
            return i
    raise SystemExit("marker not found: " + marker)


def build(out, smoke=False):
    """Assemble the v20 notebook and write it to `out` (path-like).

    Returns the nbformat notebook object. `smoke` selects the short config:
    2 folds, 1 GNN seed, 2000 PI1M pretrain rows, d=32 / layers=2 / 1 epoch
    for the v20 encoder, 300 tokenizer PI1M rows.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    L = _read_detail(SRC)
    _A_START = _idx(L, "# Graph featurization")
    _A_END = _idx(L, "# Twin source:")
    _B_END = _idx(L, "    stack_oof[t] = oof; stack_test[t] = te_pred") + 1
    CORE_A = "\n".join(L[_A_START:_A_END])
    CORE_B = "\n".join(L[_A_END:_B_END])
    assert "class GINEEncoder" in CORE_A
    assert "class MTGNN" in CORE_A
    assert "lgb_test_te = np.zeros((len(Xte), len(TARGETS)), dtype=np.float32)" in CORE_B
    assert "stack_oof[t] = oof; stack_test[t] = te_pred" in CORE_B

    # ------------------------------------------------------------------ config
    smoke_flags = {
        "smoke": smoke,
        "folds": 2 if smoke else 5,
        "maxep": 4 if smoke else 120,
        "pate": 5 if smoke else 20,
        "bs": 64 if smoke else 256,
        "prtep": 1 if smoke else os.environ.get("PRETRAIN_EPOCHS", "10"),
        "prtsmp": 2000 if smoke else os.environ.get("PRETRAIN_SAMPLE", "2000000"),
        "gnnseeds": os.environ.get("GNN_SEEDS", "1" if smoke else "42,999,2025"),
        "pi_count": 300 if smoke else os.environ.get("V20_PI_COUNT", "20000"),
        "d": 32 if smoke else os.environ.get("V20_D", "256"),
        "layers": 2 if smoke else os.environ.get("V20_LAYERS", "4"),
        "epochs": 1 if smoke else os.environ.get("V20_EPOCHS", "2"),
        "seed": os.environ.get("V20_SEED", "42"),
    }
    REPL = {
        "@FOLDS@": str(smoke_flags["folds"]),
        "@MAXEP@": str(smoke_flags["maxep"]),
        "@PATE@": str(smoke_flags["pate"]),
        "@BS@": str(smoke_flags["bs"]),
        "@PRTEP@": str(smoke_flags["prtep"]),
        "@PRTSMP@": str(smoke_flags["prtsmp"]),
        "@GNNSEEDS@": smoke_flags["gnnseeds"],
        "@PICOUNT@": str(smoke_flags["pi_count"]),
        "@VD@": str(smoke_flags["d"]),
        "@VLAYERS@": str(smoke_flags["layers"]),
        "@VEPOCHS@": str(smoke_flags["epochs"]),
        "@VSEED@": smoke_flags["seed"],
    }

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}

    C = []
    M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    P = lambda s: C.append(nbf.v4.new_code_cell(s))

    # ------------------------------------------------------------------ intro
    intro = f"""# PolyWin R2 — v20: self-trained SMILES-encoder arm on the P14 blend

## What this kernel does
* **Level-0:** verbatim P14 sources from `mt_gnn_v2.py` (CORE_A graph feats +
  GINE encoder, then CORE_B: twins + MT-GNN fold OOF + GBM trio stack). The
  P14 arms (`oof_gbm`/`oof_mt`/`test_gbm`/`test_mt`) are **recomputed in-kernel**
  from `train.csv`/`test.csv`/`PI1M.csv` ONLY.
* **v20 arm:** a self-trained, label-free SMILES-token MaskEncoder (codec +
  encoder inlined below) pretrained on the competition SMILES, pooled, then
  per-target Ridge heads (fold-safe on canonical smiles) produce `oof_trf`/`test_trf`.
* **Blend:** P14 fold-safe per-target Ridge alpha sweep on the 3 arms
  (gbm, mt, trf); alpha <= 0.30 is a **pre-registered gate**.
* **Gate (pre-registered, do not soften):**
  * mean_v20 - mean_p14 >= 0.003
  * worst target delta >= -0.003
  * all per-target alphas <= 0.30
  * If ALL pass -> write `submission.csv`. Else -> **GATE=FAIL -> P14 stays
    final** and NO submission is written.

Only OSI-approved libs: PyTorch, RDKit, scikit-learn, LightGBM, CatBoost, XGBoost.
"""
    M(intro)

    # ------------------------------------------------------------------ setup
    setup = r"""import os, sys, time, gc, random, warnings
import subprocess, importlib.util

def ensure_pkg(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print("installing", pkg, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--disable-pip-version-check", pkg])

for _p, _n in [("rdkit", "rdkit"), ("torch_geometric", "torch_geometric"),
               ("lightgbm", "lightgbm"), ("catboost", "catboost"),
               ("xgboost", "xgboost"), ("scipy", "scipy")]:
    ensure_pkg(_p, _n)

# --- CUDA probe / repair identical to v14/v16 (P100 sm_60 needs torch 2.5.1) ---
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
        _idx = "http" + "s://download.pytorch.org/whl/cu121"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-cache-dir", "--index-url", _idx,
                               "torch==2.5.1"], timeout=1800)
        _r2 = subprocess.run([sys.executable, "-c", _probe], capture_output=True,
                             text=True, timeout=600)
        print("post-reinstall probe rc:", _r2.returncode, flush=True)
    except Exception as _e:
        print("torch reinstall errored:", repr(_e)[:200], flush=True)
if os.path.exists("/kaggle"):
    _force_cuda()

# --- CUDA probe (no reinstall needed; Kaggle GPU has a working build) ---
def _cuda_ok():
    try:
        if not torch.cuda.is_available():
            return False
        a = torch.zeros(4, device="cuda"); b = a + 1; torch.cuda.synchronize(); del a, b
        return True
    except Exception:
        return False

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
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
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
GNN_SEEDS = "@GNNSEEDS@"
os.environ["GNN_SEEDS"] = GNN_SEEDS

# v20 encoder + gate config ------------------------------------------------
V20_PI_COUNT = @PICOUNT@
V20_D = @VD@
V20_LAYERS = @VLAYERS@
V20_EPOCHS = @VEPOCHS@
V20_SEED = @VSEED@

# Kaggle-only data source: the competition input tree. Base is the /kaggle/input
# root so find_input probes every mount layout (direct, <slug>, competitions/<slug>).
if os.path.exists("/kaggle"):
    INP = "/kaggle/input"
    WORK = "/kaggle/working"
    os.environ.setdefault("SMOKE", "1" if SMOKE else "0")
else:
    INP = os.path.join("vault", "official_data")
    WORK = os.path.join("vault", "kernel-v20-embed")
os.makedirs(WORK, exist_ok=True)
PRETRAINED = os.path.join(WORK, "pretrained_encoder.pt")
OUT = WORK

print("----- v20 CONFIG -----", flush=True)
print("folds:", GLOBAL_FOLDS, "| GNN_SEEDS:", GNN_SEEDS,
      "| PRETRAIN_SAMPLE:", PRETRAIN_SAMPLE, flush=True)
print("v20: PI_COUNT", V20_PI_COUNT, "d", V20_D, "layers", V20_LAYERS,
      "epochs", V20_EPOCHS, "seed", V20_SEED, flush=True)
print("device:", DEVICE, "| SMOKE:", SMOKE, "| out:", OUT, flush=True)
"""
    for _k, _v in REPL.items():
        setup = setup.replace(_k, _v)
    P(setup)

    # ------------------------------------------------------------------ targets
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
    """MolToSmiles only — identical string to canonical(s)[0] for the PI1M
    pretrain corpus, avoids ~N MolToInchiKey computations, no result change."""
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
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(m)
        gasteiger = [a.GetDoubleProp('_GasteigerCharge') for a in m.GetAtoms()]
        g_mean = np.mean(gasteiger)
        g_std = np.std(gasteiger) if len(gasteiger) > 1 else 0.0
        g_min = np.min(gasteiger); g_max = np.max(gasteiger)
    except Exception:
        g_mean = g_std = g_min = g_max = 0.0
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

    M("## 3. Pretrain the GINE encoder on PI1M")
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
        print("saved pretrained_encoder.pt", flush=True)
    return best_state

print("=== P14: Pretraining GNN on PI1M ===", flush=True)
pretrained_state = pretrain()
""")

    M("## 4. Level-0 predictions (verbatim: leak-safe twins + MT-GNN fold OOF + GBM trio stack)")
    P(CORE_B)

    M("## 5. Recompute the P14 blend arms in-kernel (no npz, no externals)")
    P(r"""# P14 arms on global indices, recomputed from CORE_B outputs (stack_oof/mt_oof
# per target, stack_test/mt_test per test row). These are the P14 arms the
# gate compares against.
oof_gbm_global = np.full(len(X), np.nan, dtype=np.float32)
oof_mt_global = np.full(len(X), np.nan, dtype=np.float32)
for t in TARGETS:
    idx = idx_of_target[t]
    oof_gbm_global[idx] = stack_oof[t]
    oof_mt_global[idx] = mt_oof[t]
test_gbm_global = np.zeros(len(Xte), dtype=np.float32)
test_mt_global = np.zeros(len(Xte), dtype=np.float32)
for t in TARGETS:
    m_te = (tef["target_type"] == t).values
    test_gbm_global[m_te] = stack_test[t][m_te]
    test_mt_global[m_te] = mt_test[m_te]
assert not np.isnan(oof_gbm_global).any() and not np.isnan(oof_mt_global).any()
print("P14 arms recomputed in-kernel:",
      oof_gbm_global.shape, test_gbm_global.shape, flush=True)
""")

    # ---------------------------------------------------------- v20 module sources
    M("## 6. v20 self-trained SMILES-encoder module sources (inlined verbatim)")
    for _mod in ("v20_codec.py", "v20_encoder.py", "v20_arm_cv.py",
                 "v20_blend.py", "v20_gate_report.py"):
        P((_src_dir / _mod).read_text(encoding="utf-8"))

    # ---------------------------------------------------------- gate cell
    M("## 7. v20 arm + pre-registered gate (submission only on PASS)")

    # _p14_2arm_oof is the P14 baseline protocol helper from run_v20_gate.py.
    _gate_run = (Path("run_v20_gate.py").read_text(encoding="utf-8")
                 if (Path("run_v20_gate.py")).exists() else "")
    _f_start = _gate_run.index("def _p14_2arm_oof")
    _f_end = _gate_run.index("def run_gate")
    _p14_2arm = _gate_run[_f_start:_f_end].rstrip()

    gate = _p14_2arm + r'''


def _make_p14_ref(M2, y, g, n_splits=5):
    # Alias kept for the gate cell; identical protocol.
    return _p14_2arm_oof(M2, y, g, n_splits)

# ---- tokenizer on a PI1M sample (label-free) ----
p1_path = find_input(INP, "PI1M.csv")
p1_smiles = pd.read_csv(p1_path, nrows=V20_PI_COUNT)
smi_col = "SMILES" if "SMILES" in p1_smiles.columns else "smiles"
p1_smiles = p1_smiles[smi_col].astype(str).tolist()
print("[v20] building tokenizer on", len(p1_smiles), "PI1M rows", flush=True)
tok = build_tokenizer(p1_smiles, max_vocab=1600, min_count=2)
print("[v20] vocab:", len(tok["tok2id"]), flush=True)

ids_tr = tokenize_batch(tok, trf["smiles"].values, max_len=128)
ids_te = tokenize_batch(tok, tef["smiles"].values, max_len=128)

model = MaskEncoder(vocab=len(tok["tok2id"]), d=V20_D, layers=V20_LAYERS,
                    max_len=128, dropout=0.1)
pretrain_ids = np.concatenate([ids_tr, ids_te]).astype(np.int64)
random.seed(V20_SEED); np.random.seed(V20_SEED); torch.manual_seed(V20_SEED)
pretrain_encoder(model, pretrain_ids, epochs=V20_EPOCHS, bs=64, lr=3e-4,
                 seed=V20_SEED, mask_p=0.15)

pool_tr = pool_embeddings(model, ids_tr.astype(np.int64))
pool_te = pool_embeddings(model, ids_te.astype(np.int64))
print("pooled embeddings:", pool_tr.shape, pool_te.shape, flush=True)

oof_trf, test_trf = compute_trf_arm(
    pool_tr, pool_te, Y, T, tef["target_type"].values,
    G, n_splits=5, seed=V20_SEED)
print("trf arm done:", oof_trf.shape, test_trf.shape, flush=True)

# ---- 3-arm blend (P14 fold-safe alpha sweep) ----
oof_v20 = np.zeros(len(Y))
alphas, r2_p14, r2_v20 = {}, {}, {}
for t in TARGETS:
    idx = idx_of_target[t]
    gt = G[idx]
    M3 = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx], oof_trf[idx]])
    oof_v20[idx], _coefs, alphas[t] = blend_3d(
        M3, Y[idx], gt, alphas=ALPHAS, n_splits=5)
    r2_v20[t] = float(np.corrcoef(Y[idx], oof_v20[idx])[0, 1]) ** 2

    M2 = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx]])
    b2 = _p14_2arm_oof(M2, Y[idx], gt, n_splits=5)
    r2_p14[t] = float(np.corrcoef(Y[idx], b2)[0, 1]) ** 2

mean_p14 = float(np.mean(list(r2_p14.values())))
assert abs(mean_p14 - 0.8641) <= 0.005, (
    f"recomputed P14 {mean_p14:.4f} deviates from reference 0.8641")
mean_v20 = float(np.mean(list(r2_v20.values())))
# pre-registered gate channel 1 (mean R2 gain): mean_delta >= 0.003
mean_delta = mean_v20 - mean_p14
worst_delta = float(min(r2_v20[t] - r2_p14[t] for t in TARGETS))
report = compute_gate_report(
    mean_delta, worst_delta, list(alphas.values()),
    thr_mean=0.003, thr_worst=0.003, alpha_cap=0.30)

print()
print("==" * 34)
print("target   r2_p14    r2_v20   delta    alpha")
for t in TARGETS:
    print(f"{t:6s}   {r2_p14[t]:.4f}   {r2_v20[t]:.4f}   "
          f"{r2_v20[t]-r2_p14[t]:+.4f}   {alphas[t]:.2f}")
print("-" * 34)
print(f"mean_v20 {mean_v20:.4f}  mean_p14 {mean_p14:.4f}  "
      f"mean_delta {mean_delta:+.4f}")
print(f"worst_delta {worst_delta:+.4f}  alphas_ok {report['alphas_ok']}")
print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}")
print("==" * 34)

GATE = report["pass"]
test_pred = None
if report["pass"]:
    test_pred = np.zeros(len(tef))
    for t in TARGETS:
        idx = idx_of_target[t]
        idx_te = np.where(tef["target_type"].values == t)[0]
        M_tr = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx], oof_trf[idx]])
        M_te = np.column_stack(
            [test_gbm_global[idx_te], test_mt_global[idx_te], test_trf[idx_te]])
        lr = Ridge(alpha=alphas[t], fit_intercept=True).fit(M_tr, Y[idx])
        test_pred[idx_te] = lr.predict(M_te)
    assert np.isfinite(test_pred).all(), "NaN in test predictions"
    sub = pd.DataFrame({"id": tef["id"].values, "target": test_pred})
    write_submission(sub, os.path.join(OUT, "submission.csv"))
    print("GATE=PASS -> wrote submission.csv", flush=True)
else:
    print("GATE=FAIL -> P14 stays final", flush=True)
print("DONE", flush=True)
'''
    P(gate)

    nb["cells"] = C
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote notebook:", out, "| cells:", len(C), "| SMOKE:", smoke, flush=True)
    return nb


def _write_metadata():
    data = {
        "id": "shubhamkambli11/polywin-r2-v20-embed",
        "title": "polywin r2 v20 embed",
        "code_file": OUT_NB_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": ["ppp-round-2"],
        "dataset_sources": [],
    }
    with open(OUT_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)
    print("wrote metadata:", OUT_DIR / "kernel-metadata.json", flush=True)


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE", "0") == "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build(OUT_DIR / OUT_NB_NAME, smoke=smoke)
    _write_metadata()