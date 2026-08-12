#!/usr/bin/env python
"""Build PolyWin_R2_v10_pretrain.ipynb — Phase 3: PI1M self-supervised GINE pretraining.

Reproducible Kaggle competition notebook that answers ONE question honestly:

    Does initializing the per-target GINE from a PI1M-pretrained encoder beat a
    randomly-initialized (scratch) GINE on the SAME fold-safe OOF + trust check?

Pipeline:
  1. load train/test from /kaggle/input (or local official_dataset), canonicalize,
     dedupe, GroupKFold folds (persisted).
  2. Self-supervised pretraining on unlabeled PI1M: GINE encoder + attribute-masking
     decoder (mask ~15% atoms, reconstruct masked atom features; masked-bond co-task).
     PI1M has NO labels, so this never touches validation/test labels and is explicitly
     a permitted use of the archive (rules: "masked atom/token prediction on molecular
     graphs", "representation learning — train embeddings on PI1M").
  3. Fine-tune two per-target GINEs on the SAME folds:
        arm A = encoder initialized randomly (scratch)
        arm B = encoder initialized from the pretrained weights (heads stay random)
     both use the v9 fold-safe protocol with a 15% internal trust-check holdout.
  4. Honest A/B table (OOF R2, trust R2, gap) + submission from the pretrained arm.

Labelled-GNN cost is small (7.4k rows); pretraining is the cost. Tunables (env or SMOKE):
  PRETRAIN_SAMPLE  -> PI1M graphs kept for pretraining (default 20000)
  PRETRAIN_EPOCHS  -> self-supervised epochs        (default 5)
  MINI_EPOCHS      -> per-fold fine-tune epochs     (default 40)

Rules: no hand labelling; PI1M allowed; OSI-approved libs only (PyTorch, PyG, RDKit, sklearn).

Run:  python build_pretrain_kaggle_nb.py      # writes PolyWin_R2_v10_pretrain.ipynb
"""
import nbformat as nbf

OUT = "PolyWin_R2_v10_pretrain.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

C = []
M = lambda s: C.append(nbf.v4.new_markdown_cell(s))
P = lambda s: C.append(nbf.v4.new_code_cell(s))

# =====================================================================
M("""# PolyWin R2 — v10: PI1M Self-Supervised Pretraining (Kaggle kernel)

## Question
Does a GINE encoder **pretrained on the unlabeled PI1M archive** beat a **randomly
initialized (scratch)** GINE when both are fine-tuned on the 7 polymer targets?

## Why this matters (R3 roadmap)
* The residual leaderboard gap (0.847 → 0.90+) is likely a **representation** gap, not a
  stacking gap. Pretraining on ~1M unlabeled polymers is the biggest untested lever.
* PI1M is **explicitly** allowed: the rules list "self-supervised pretraining — masked
  atom/token prediction on SMILES or molecular graphs" and "representation learning —
  train embeddings on PI1M, fine-tune on the 7 Target Properties".

## Protocol (honest)
* GroupKFold on canonical polymer, identical folds across BOTH arms.
* Fold-safe OOF + a further 15% internal **trust-check** holdout scored separately.
* Arm A = scratch GINE; Arm B = GINE initialized from the pretrained encoder.
* Compare OOF R² and trust R² per target; submission uses the pretrained arm.

Only OSI-approved libs: PyTorch, PyTorch Geometric, RDKit, scikit-learn.
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

for _p, _n in [("rdkit", "rdkit"), ("torch_geometric", "torch_geometric")]:
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

if ON_KAGGLE:
    WORK = "/kaggle/working"; INP = "/kaggle/input"
else:
    WORK = os.path.join("vault", "pipeline_out_pretrain_smoke" if SMOKE else "pipeline_out_pretrain")
    INP = "competition/data/raw"
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
      "| PRETRAIN_SAMPLE:", PRETRAIN_SAMPLE)
""")

# =====================================================================
M("## 1. Data — canonicalize, dedupe, folds, PI1M sample")

# =====================================================================
P("""TARGETS = ["tg","egc","egb","eps","nc","ei","eea"]
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

def canon_key(smiles):
    return smiles.replace("*", "").replace("[*]", "")

def parse_mol(smiles):
    m = Chem.MolFromSmiles(smiles.replace("*", "[*]"))
    if m is None:
        m = Chem.MolFromSmiles(smiles.replace("*", "C"))
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
print("rows after dedupe:", len(dedup))

FOLDS_CSV = os.path.join(WORK, "pretrain_folds.csv")
if os.path.exists(FOLDS_CSV):
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy()
    assert len(folds) == len(dedup)
else:
    gkf = GroupKFold(n_splits=GLOBAL_FOLDS)
    folds = np.zeros(len(dedup), dtype=int)
    for i, (_, va) in enumerate(gkf.split(dedup, groups=dedup["canon"])):
        folds[va] = i
    pd.DataFrame({"canon": dedup["canon"].values, "fold": folds}).to_csv(FOLDS_CSV, index=False)
dedup["fold"] = folds
print("fold sizes:", np.bincount(folds))

# PI1M pretraining corpus (unlabeled). Column is `SMILES` in the archive.
if pl_path:
    pl = pd.read_csv(pl_path)
    smi_col = "SMILES" if "SMILES" in pl.columns else "smiles"
    pl = pl[[smi_col]].rename(columns={smi_col: "smiles"})
    pl["canon"] = pl["smiles"].map(canon_key)
    pl = pl.drop_duplicates("canon")["smiles"].tolist()
    rng = np.random.RandomState(SEED)
    rng.shuffle(pl)
    pl = pl[:PRETRAIN_SAMPLE]
    print("PI1M pretraining corpus:", len(pl), "unique SMILES (capped at", PRETRAIN_SAMPLE, ")")
else:
    pl = []
    print("PI1M.csv not found - pretraining will be skipped; only scratch arm runs")
""")

# =====================================================================
M("## 2. Graph featurization (shared by pretraining + fine-tuning)")

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

# per-target z-score stats (train only, no fold leakage)
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

t0 = time.time()
train_graphs = build_train_graphs()
test_graphs = build_test_graphs()
print(f"Built {len(train_graphs)} train graphs, {len(test_graphs)} test graphs "
      f"({len(dedup)-len(train_graphs)} train SMILES failed) in {time.time()-t0:.0f}s")

# pretraining-only graphs (no labels)
def build_pretrain_graphs(smiles_list):
    graphs = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
    return graphs

if pl:
    t0 = time.time()
    pl_graphs = build_pretrain_graphs(pl)
    print(f"Built {len(pl_graphs)} pretraining graphs from {len(pl)} SMILES in {time.time()-t0:.0f}s")
else:
    pl_graphs = []
""")

# =====================================================================
M("## 3. Models — GINE encoder shared between pretraining and fine-tuning\n\nThe encoder is a standalone module so the pretrained weights can be dropped straight into the fine-tune trunk. The fine-tune model adds per-target heads (target-aware like the Round-2 base pipeline) on top of a mean+add pooled graph embedding.")

# =====================================================================
P("""class GINEEncoder(nn.Module):
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
    \"\"\"Attribute-masking self-supervision (GraphMAE-style).

    Corrupts 15% of atom features and 20% of bond features (zeroed), encodes the
    corrupted graph, and decodes the masked elements back to their original features.
    No labels are used -- purely structural/feature reconstruction.
    \"\"\"
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

# =====================================================================
M("## 4. Self-supervised pretraining on PI1M\n\nAttribute masking: 15% of atom features and 20% of bond features are zeroed per graph; the encoder must reconstruct them from context. This is the rules-approved \"masked atom/token prediction on molecular graphs\" task. The encoder weights are saved to `pretrained_encoder.pt` for the fine-tune arm B.")

# =====================================================================
P("""def pretrain(epochs=PRETRAIN_EPOCHS, batch_size=256, lr=1e-3, patience=5):
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
        print(f"pretrain ep {epoch+1}/{epochs}: atom={avg_a:.4f} bond={avg_b:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(WORK, "pretrained_encoder.pt"))
    print("saved pretrained_encoder.pt")
    return best_state

pretrain_state = pretrain()
""")

# =====================================================================
M("## 5. Fold-safe fine-tuning, both arms (same folds, same budget)\n\nArm A (scratch) and Arm B (pretrained init) share every hyperparameter and the exact fold split + 15% trust-check holdout, so the only difference is encoder initialization.")

# =====================================================================
P("""def train_gnn(init_state=None, epochs=MINI_EPOCHS, batch_size=64, lr=1e-3, patience=10,
               trust_frac=0.15):
    row_to_graph = {g.row_id: g for g in train_graphs}
    n_train = len(dedup)
    oof = np.full(n_train, np.nan)
    trust_scores = {t: [] for t in TARGETS}
    fold_states = []
    for fold in sorted(dedup["fold"].unique()):
        fold_train = dedup.index[dedup["fold"] != fold]
        val = dedup.index[dedup["fold"] == fold]
        rng = np.random.RandomState(SEED + fold)
        trust_mask = rng.rand(len(fold_train)) < trust_frac
        trust_ids = fold_train[trust_mask]; tr_ids = fold_train[~trust_mask]
        tr_graphs = [row_to_graph[i] for i in tr_ids if i in row_to_graph]
        val_graphs = [row_to_graph[i] for i in val if i in row_to_graph]
        trust_graphs = [row_to_graph[i] for i in trust_ids if i in row_to_graph]
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
        with torch.no_grad():
            for g in trust_graphs:
                gb = Batch.from_data_list([g]).to(device)
                pred = model(gb)
                ti = int(g.target_idx.item()); t_name = TARGETS[ti]
                mean_, std_ = target_stats[t_name]
                trust_scores[t_name].append((g.y.item()*std_+mean_, pred[0, ti].item()*std_+mean_))
        fold_states.append((fold, best_state))
        print(f"  fold {fold}: best val MSE (norm)={best_val:.4f} ({time.time()-ft0:.0f}s, ep={epochs_used})",
              flush=True)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, trust_scores, fold_states

print("\\n=== Arm A: scratch init ===")
scratch_oof, scratch_trust, scratch_states = train_gnn(init_state=None)
print("\\n=== Arm B: pretrained init ===")
pt_oof, pt_trust, pt_states = train_gnn(init_state=pretrain_state)
""")

# =====================================================================
M("## 6. Honest A/B comparison\n\nFor each target we report scratch vs pretrained OOF R² and trust R². If the pretrained arm is meaningfully better on OOF *and* trust (no trust-overfit), pretraining earned its place.")

# =====================================================================
P("""true_y = dedup["target"].values

def target_r2(oof, trust_scores):
    out = {}
    for t in TARGETS:
        mask = (dedup["target_type"] == t).values
        yp = oof[mask]; fin = ~np.isnan(yp)
        o = r2_score(true_y[mask][fin], yp[fin]) if fin.sum() >= 5 else np.nan
        pr = trust_scores[t]
        r = r2_score(*zip(*pr)) if len(pr) >= 5 else np.nan
        out[t] = (o, r)
    return out

r_scratch = target_r2(scratch_oof, scratch_trust)
r_pretr   = target_r2(pt_oof, pt_trust)

print(f"{'target':<6} {'sc-OOF':>10} {'sc-trust':>10} {'pt-OOF':>10} {'pt-trust':>10} {'dOOF':>8}")
rows = []
for t in TARGETS:
    so, st = r_scratch[t]; po, pt = r_pretr[t]
    d = po - so if not (np.isnan(so) or np.isnan(po)) else np.nan
    rows.append({"target": t, "scratch_oof": so, "scratch_trust": st,
                 "pretrain_oof": po, "pretrain_trust": pt, "delta_oof": d})
    print(f"{t:<6} {so:>10.4f} {st:>10.4f} {po:>10.4f} {pt:>10.4f} "
          f"{('%.4f'%d) if not np.isnan(d) else '  nan'}")
ab = pd.DataFrame(rows)
ab.to_csv(os.path.join(WORK, "pretrain_ab.csv"), index=False)
m_s = np.nanmean([r[0] for r in r_scratch.values()]); m_p = np.nanmean([r[0] for r in r_pretr.values()])
print(f"\\nmean OOF R2: scratch {m_s:.4f} | pretrained {m_p:.4f} | delta {m_p-m_s:+.4f}")
print("\\nVerdict:", "PRETRAINING WINS" if m_p > m_s else "SCRATCH >= PRETRAIN (pretraining not yet worth it)")
""")

# =====================================================================
M("## 7. Test predictions + submission\n\nBag over the fold models of the pretrained arm (or the better arm), write `gnn_oof.csv`/`gnn_test.csv` for downstream MoE blending, and emit a valid `submission.csv` with the standard physics bounds.")

# =====================================================================
P("""def predict_graphs_on(graphs, state):
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

# choose the arm with the better mean OOF
use_states = pt_states if m_p >= m_s else scratch_states
arm_name = "pretrained" if m_p >= m_s else "scratch"
test_preds = {}
print(f"Computing test predictions ({arm_name} arm, bag over fold models)...")
for fold, state in use_states:
    p = predict_graphs_on(test_graphs, state)
    for rid, v in p.items():
        test_preds[rid] = test_preds.get(rid, 0.0) + v / len(use_states)
    del p
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

win_oof = pt_oof if m_p >= m_s else scratch_oof
pd.DataFrame({"row_id": list(dedup.index), "target_type": dedup["target_type"].values,
              "gnn_oof": win_oof}).to_csv(os.path.join(WORK, "gnn_oof.csv"), index=False)
pd.DataFrame({"row_id": list(test.index), "target_type": test["target_type"].values,
              "gnn_test": [test_preds.get(r, np.nan) for r in test.index]}) \\
    .to_csv(os.path.join(WORK, "gnn_test.csv"), index=False)
print("wrote gnn_oof.csv, gnn_test.csv")

final = np.zeros(len(test))
for tt in TARGETS:
    mte = (test["target_type"] == tt).values
    final[mte] = np.array([test_preds.get(r, np.nan) for r in test.index])[mte]
for _tt in ("egc", "egb", "ei"):
    _mm = (test["target_type"].values == _tt); final[_mm] = np.maximum(final[_mm], 0.0)
_mm = (test["target_type"].values == "eps"); final[_mm] = np.maximum(final[_mm], 1.0)
_mm = (test["target_type"].values == "nc"); final[_mm] = np.clip(final[_mm], 1.0, 3.0)
sub = pd.DataFrame({"id": test["id"].values, "target": final})
sub.to_csv(os.path.join(WORK, "submission.csv"), index=False)
print("submission saved:", os.path.join(WORK, "submission.csv"), sub.shape)
print("\\nPrediction stats by target:")
print(pd.DataFrame({"target": test["target_type"], "pred": final}).groupby("target")["pred"].describe().round(3).to_string())

# ---- figures ----
def savefig(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved", name)

pal = sns.color_palette("viridis", len(TARGETS))
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(TARGETS)); w = 0.38
so = [r[0] for r in r_scratch.values()]; po = [r[0] for r in r_pretr.values()]
st = [r[1] for r in r_scratch.values()]; pt = [r[1] for r in r_pretr.values()]
ax.bar(x - w/2, so, w, label="scratch OOF", color="#999999")
ax.bar(x + w/2, po, w, label="pretrain OOF", color="#2a6fb0")
ax.set_xticks(x); ax.set_xticklabels(TARGETS); ax.set_ylabel("R2")
ax.set_title("A/B: scratch vs pretrained-init GINE, fold-safe OOF R2"); ax.legend()
savefig(fig, "10_pretrain_ab_oof.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, st, w, label="scratch trust", color="#999999")
ax.bar(x + w/2, pt, w, label="pretrain trust", color="#2a6fb0")
ax.set_xticks(x); ax.set_xticklabels(TARGETS); ax.set_ylabel("R2")
ax.set_title("A/B trust-check R2 (15% internal holdout)"); ax.legend()
savefig(fig, "11_pretrain_ab_trust.png")
print("\\n==== PIPELINE COMPLETE ====")
""")

nb.cells = C
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", OUT, "with", len(C), "cells")