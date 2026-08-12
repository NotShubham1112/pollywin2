#!/usr/bin/env python
"""Build the PolyWin R2 v22 self-contained Kaggle notebook — BERT-arm on P14.

v22 = P14 (bit-identical level-0: CORE_A graph feats + FULL-PI1M GINE pretrain
+ CORE_B fold OOF + GBM trio stack) + ONE additive change: a self-contained,
label-free BERT-style SMILES arm (BPE tokenizer -> masked-token MLM pretraining
-> mean-pooled embeddings -> per-target fold-safe Ridge heads) as a THIRD
column in P14's per-target Ridge blend, plus the pre-registered gates 0-3.

Fork of build_v20_kaggle_nb.py. The v20 self-trained codec/encoder arm is
replaced by the v22 BERT arm; the gate cell is the port of the v22 harness
(run_v22_gate.py) body into notebook form. CORE_A/CORE_B are extracted from
mt_gnn_v2.py bit-identically to v14 (same markers, same slicing).

Gate cell mirrors v21 exactly:
  * EPS_NC_EI=("eps","nc","ei"), SOFT_DELTA=0.0015, STRONG_DELTA=0.003,
    WORST_TOL=0.003, corr2, gate_1_leak_audit on
    np.column_stack([oof_gbm_global, oof_mt_global, oof_bert]),
  * verdict line: `GATE: PASS -> v22 proceeds` / `GATE: FAIL -> P14 stays final`,
  * writes `v22_blend_report.csv` always and `submission_v22.csv` ONLY on PASS.

Run:
    python build_v22_kaggle_nb.py          # FULL config -> New folder/
    SMOKE=1 python build_v22_kaggle_nb.py  # fast smoke validation
"""
import os
from pathlib import Path

import nbformat as nbf

_src_dir = Path(__file__).resolve().parent
SRC = _src_dir.parent / "mt_gnn_v2.py"
OUT_NB_NAME = "PolyWin_R2_v22_bert_arm.ipynb"


def _idx(lines, marker):
    for i, line in enumerate(lines):
        if marker in line:
            return i
    raise SystemExit("marker not found: " + marker)


def build(out, smoke=False):
    """Assemble the v22 notebook and write it to `out` (path-like).

    Returns the nbformat notebook object. `smoke` selects the short config:
    2 folds, 1 GNN seed, 2000 PI1M pretrain rows, and the small v22 BERT arm
    (d=32 / layers=2 / heads=4 / 1 epoch / 400 vocab / 2000 BPE subset).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(SRC, encoding="utf-8") as f:
        L = f.read().split("\n")

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
        "pi_count": 2000 if smoke else os.environ.get("V22_PI_COUNT", "-1"),
        "d": 32 if smoke else os.environ.get("V22_D", "384"),
        "layers": 2 if smoke else os.environ.get("V22_LAYERS", "6"),
        "heads": 4 if smoke else os.environ.get("V22_HEADS", "8"),
        "epochs": 1 if smoke else os.environ.get("V22_EPOCHS", "1"),
        "bpe_subset": 2000 if smoke else os.environ.get("V22_BPE_SUBSET", "150000"),
        "vocab": 400 if smoke else os.environ.get("V22_VOCAB", "4000"),
        "seed": os.environ.get("V22_SEED", "42"),
    }
    REPL = {
        "@FOLDS@": str(smoke_flags["folds"]),
        "@MAXEP@": str(smoke_flags["maxep"]),
        "@PATE@": str(smoke_flags["pate"]),
        "@BS@": str(smoke_flags["bs"]),
        "@PRTEP@": str(smoke_flags["prtep"]),
        "@PRTSMP@": str(smoke_flags["prtsmp"]),
        "@GNNSEEDS@": smoke_flags["gnnseeds"],
        "@V22PICOUNT@": str(smoke_flags["pi_count"]),
        "@V22D@": str(smoke_flags["d"]),
        "@V22LAYERS@": str(smoke_flags["layers"]),
        "@V22HEADS@": str(smoke_flags["heads"]),
        "@V22EPOCHS@": str(smoke_flags["epochs"]),
        "@V22BPESUBSET@": str(smoke_flags["bpe_subset"]),
        "@V22VOCAB@": str(smoke_flags["vocab"]),
        "@V22SEED@": smoke_flags["seed"],
    }

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}

    C = []
    M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    P = lambda s: C.append(nbf.v4.new_code_cell(s))

    # ------------------------------------------------------------------ intro
    intro = """# PolyWin R2 — v22: BERT-arm (BPE + SMILES MLM) on the P14 blend

## What this kernel does
* **Level-0 (P14, frozen, bit-identical):** verbatim P14 sources from
  `mt_gnn_v2.py` (CORE_A graph feats + GINE encoder pretrained on the entire
  PI1M archive, then CORE_B: leak-safe twins + MT-GNN fold OOF + GBM trio
  stack). Seeds 42/999/2025, 5 folds, all params unchanged from P14.
* **v22 arm:** a fully in-notebook, label-free BERT-style SMILES encoder:
  BPE tokenizer (trained on the PI1M sample) -> masked-token MLM pretraining
  (`BertEncoder`, pure torch) -> mean-pooled embeddings -> per-target fold-safe
  Ridge heads (`compute_bert_arm`, GroupKFold on canonical smiles) producing
  `oof_bert` / `test_bert`.
* **Blend:** P14 fold-safe per-target alpha sweep on the 3 arms (gbm, mt, bert)
  via `blend_narm_oof`; the P14 2-arm reference is recomputed in-cell with
  `_p14_2arm_oof` so gate 2 compares like-for-like.
* **Gates (pre-registered, do NOT soften):** gates 0-3 mirror v21:
  * gate 1 (leak audit): BERT-arm features vs true other-target labels exact
    match count must be 0
  * gate 2 (OOF gain): mean over {eps,nc,ei} AND overall >= P14 reference
    + +0.0015 (soft) / +0.003 (strong)
  * gate 3 (worst-target): every per-target delta >= -0.003
  * Verdict: `GATE: PASS -> v22 proceeds` or `GATE: FAIL -> P14 stays final`.
* Submission: `submission_v22.csv` (`id,target`, P14 format) written ONLY on
  PASS; `v22_blend_report.csv` is always written.

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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-cache-dir", "--index-url",
                               "https://download.pytorch.org/whl/cu121",
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

# v22 BERT-arm config ------------------------------------------------
V22_PI_COUNT = @V22PICOUNT@
V22_D = @V22D@
V22_LAYERS = @V22LAYERS@
V22_HEADS = @V22HEADS@
V22_EPOCHS = @V22EPOCHS@
V22_BPE_SUBSET = @V22BPESUBSET@
V22_VOCAB = @V22VOCAB@
V22_SEED = @V22SEED@

# Kaggle-only data source: the competition input tree. Base is the /kaggle/input
# root so find_input probes every mount layout (direct, <slug>, competitions/<slug>).
if os.path.exists("/kaggle"):
    INP = "/kaggle/input"
    WORK = "/kaggle/working"
    os.environ.setdefault("SMOKE", "1" if SMOKE else "0")
else:
    INP = "official_dataset"
    WORK = os.path.join("vault", "pipeline_out_v22")
os.makedirs(WORK, exist_ok=True)
PRETRAINED = os.path.join(WORK, "pretrained_encoder.pt")
OUT = WORK

print("----- v22 CONFIG -----", flush=True)
print("folds:", GLOBAL_FOLDS, "| GNN_SEEDS:", GNN_SEEDS,
      "| PRETRAIN_SAMPLE:", PRETRAIN_SAMPLE, flush=True)
print("v22: PI_COUNT", V22_PI_COUNT, "d", V22_D, "layers", V22_LAYERS,
      "heads", V22_HEADS, "epochs", V22_EPOCHS, "bpe_subset", V22_BPE_SUBSET,
      "vocab", V22_VOCAB, "seed", V22_SEED, flush=True)
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
# v22 gate compares against.
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

    # ---------------------------------------------------------- v22 module sources
    M("## v22 BERT-arm module sources (inlined verbatim)")
    for _mod in ("v22_tokenizer.py", "v22_encoder.py", "v22_arm_cv.py",
                 "v22_blend.py", "v22_gate_report.py"):
        P((_src_dir / _mod).read_text(encoding="utf-8"))

    # ---------------------------------------------------------- arm + gate cell
    M("## v22 BERT arm + pre-registered gate (submission_v22.csv only on PASS)")

    gate = r'''# ---- v22 BERT arm + pre-registered gates 0-3 (port of run_v22_gate.py) ----
# The P14 arms (oof_gbm_global/oof_mt_global/test_gbm_global/test_mt_global)
# were recomputed in-kernel above; G/Y/idx_of_target/trf come from the P14
# fork cells. Gates mirror v21 exactly (EPS_NC_EI, SOFT/STRONG/WORST deltas,
# corr2, gate_1_leak_audit).

def corr2(y, o):
    """Per-target R^2 reported for the gate comparison (corr^2, as the v21 gate)."""
    return float(np.corrcoef(y, o)[0, 1]) ** 2

# ---- BPE tokenizer on a PI1M sample (label-free) ----
p1_path = find_input(INP, "PI1M.csv")
p1_smiles = pd.read_csv(p1_path, nrows=V22_PI_COUNT if V22_PI_COUNT > 0 else None)
smi_col = "SMILES" if "SMILES" in p1_smiles.columns else "smiles"
p1_smiles = p1_smiles[smi_col].astype(str).tolist()
print("[v22] learning BPE on", len(p1_smiles), "PI1M rows", flush=True)
tok = learn_bpe(p1_smiles, vocab_target=V22_VOCAB,
                max_subset=V22_BPE_SUBSET, seed=V22_SEED)
print("[v22] vocab:", len(tok["tok2id"]), flush=True)

ids_tr = tokenize_batch(tok, trf["smiles"].values, max_len=128)
ids_te = tokenize_batch(tok, tef["smiles"].values, max_len=128)

protected_ids = tuple(tok["tok2id"].get(p, -1) for p in tok["protect"])
protected_ids = tuple(p for p in protected_ids if p >= 0)
model = BertEncoder(vocab=len(tok["tok2id"]), d=V22_D,
                    layers=V22_LAYERS, heads=V22_HEADS)
pretrain_ids = np.concatenate([ids_tr, ids_te]).astype(np.int64)
n_val = max(1, int(0.05 * len(pretrain_ids)))
pretrain_mlm(model, pretrain_ids[:-n_val], epochs=V22_EPOCHS,
             bs=256, lr=3e-4, seed=V22_SEED, mask_p=0.15,
             protected_ids=protected_ids, val_ids=pretrain_ids[-n_val:],
             device="cpu")

pool_tr = pool_embeddings(model, ids_tr.astype(np.int64))
pool_te = pool_embeddings(model, ids_te.astype(np.int64))
print("pooled embeddings:", pool_tr.shape, pool_te.shape, flush=True)

oof_bert, test_bert = compute_bert_arm(
    pool_tr, pool_te, Y, T, tef["target_type"].values,
    G, n_splits=GLOBAL_FOLDS, seed=V22_SEED)
print("bert arm done:", oof_bert.shape, test_bert.shape, flush=True)

# ---- 3-arm blend (P14 fold-safe alpha sweep) ----
oof_v22 = np.zeros(len(Y))
alphas, w_bert, r2_p14, r2_v22 = {}, {}, {}, {}
for t in TARGETS:
    idx = idx_of_target[t]
    gt = G[idx]
    M3 = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx], oof_bert[idx]])
    oof_v22[idx], a, coefs = blend_narm_oof(M3, Y[idx].astype(np.float64), gt,
                                            n_splits=GLOBAL_FOLDS)
    alphas[t] = a
    w_bert[t] = float(coefs[2])
    r2_v22[t] = corr2(Y[idx], oof_v22[idx])

    M2 = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx]])
    b2 = _p14_2arm_oof(M2, Y[idx].astype(np.float64), gt,
                       n_splits=GLOBAL_FOLDS)
    r2_p14[t] = corr2(Y[idx], b2)

mean_p14 = float(np.mean(list(r2_p14.values())))
if not SMOKE:
    assert abs(mean_p14 - 0.8641) <= 0.005, (
        f"recomputed P14 {mean_p14:.4f} deviates from reference 0.8641")
else:
    # SMOKE recomputes the P14 arms in-kernel at reduced fidelity
    # (folds=2, 4 epochs, 2k samples, 1 GNN seed), so the in-kernel P14
    # cannot reproduce the full-fidelity reference 0.8641. The gate verdict
    # still runs below; only the plumbing sanity check is relaxed here.
    print(f"[SMOKE] recomputed P14 {mean_p14:.4f} vs reference 0.8641 "
          f"(deviation expected at reduced fidelity)", flush=True)
mean_v22 = float(np.mean(list(r2_v22.values())))
deltas = {t: r2_v22[t] - r2_p14[t] for t in TARGETS}
eps_delta = float(np.mean([deltas[t] for t in EPS_NC_EI]))
overall = float(np.mean(list(deltas.values())))
worst = float(min(deltas.values()))

leak_count = gate_1_leak_audit(
    np.column_stack([oof_gbm_global, oof_mt_global, oof_bert]),
    trf, idx_of_target, GLOBAL_FOLDS)
bert_only_r2 = compute_bert_only_r2(oof_bert, Y, T)
report = gate_report(r2_p14, r2_v22, leak_count=leak_count,
                     bert_only_r2=bert_only_r2)

print("\n==" * 34)
print("target   r2_p14    r2_v22   delta    w_BERT   bert_only_r2")
for t in TARGETS:
    print(f"{t:6s} {r2_p14[t]:.4f} {r2_v22[t]:.4f} "
          f"{deltas[t]:+.4f}  {w_bert[t]:+.3f}  {bert_only_r2.get(t, 0.0):+.4f}")
print("-" * 34)
print(f"mean_v22 {mean_v22:.4f}  mean_p14 {mean_p14:.4f}  mean_delta {overall:+.4f}")
print(f"eps/nc/ei delta {eps_delta:+.4f}  worst_delta {worst:+.4f}")
print(f"gate1 leak_count {leak_count}  gate2 soft {report['gate2_soft']}  "
      f"gate3 {report['gate3']}")
print(f"GATE: {'PASS -> v22 proceeds' if report['pass'] else 'FAIL -> P14 stays final'}")
print("==" * 34)

rows = [{"target": t, "r2_p14": r2_p14[t], "r2_v22": r2_v22[t],
         "delta": deltas[t], "alpha": alphas[t],
         "w_bert": w_bert[t], "bert_only_r2": bert_only_r2.get(t, float("nan"))}
        for t in TARGETS]
rows.append({"target": "mean", "r2_p14": mean_p14, "r2_v22": mean_v22,
             "delta": overall, "alpha": float("nan"),
             "w_bert": float("nan"), "bert_only_r2": float("nan")})
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "v22_blend_report.csv"),
                                   index=False)
print("wrote", os.path.join(OUT, "v22_blend_report.csv"), flush=True)

final_te = np.zeros(len(tef))
if report["pass"]:
    test_pred = np.zeros(len(tef))
    for t in TARGETS:
        idx = idx_of_target[t]
        idx_te = np.where(tef["target_type"].values == t)[0]
        M_tr = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx],
                                oof_bert[idx]])
        M_te = np.column_stack([test_gbm_global[idx_te], test_mt_global[idx_te],
                                test_bert[idx_te]])
        lr = Ridge(alpha=alphas[t], fit_intercept=True).fit(M_tr, Y[idx])
        test_pred[idx_te] = lr.predict(M_te)
    final_te = test_pred
    assert np.isfinite(final_te).all(), "NaN in test predictions"
    sub = pd.DataFrame({"id": tef["id"].values, "target": final_te})
    write_submission(sub, os.path.join(OUT, "submission_v22.csv"))
    print("GATE=PASS -> wrote submission_v22.csv", flush=True)
else:
    print("GATE=FAIL -> P14 stays final; no v22 submission", flush=True)
print("v22 DONE", flush=True)
'''
    P(gate)

    nb["cells"] = C
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote notebook:", out, "| cells:", len(C), "| SMOKE:", smoke, flush=True)
    return nb


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE", "0") == "1"
    _out = _src_dir / ("PolyWin_R2_v22_bert_arm" + ("_smoke" if smoke else "") + ".ipynb")
    build(_out, smoke=smoke)
