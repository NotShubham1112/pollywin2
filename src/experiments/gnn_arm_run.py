"""v9 GNN base-model arm — command driver over real pipeline data.

Runs the GNN notebook logic (GINE trunk, joint 7-target training, GroupKFold OOF,
internal trust-check holdout) against the real `dedup.pkl` / `test.pkl` from the
base pipeline's WORK dir. Fixes two issues in the scratch notebook:
  1. target_type case (pipeline uses lowercase tg/egc/...; notebook used capitals)
  2. finishes the test arm as a bag-of-fold-models -> writes gnn_test.csv

Usage:
  conda/Minconda3 python (has torch+cuda+pyg+rdkit):
    python gnn_arm_run.py [--epochs N] [--out DIR]
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from rdkit import Chem
from sklearn.metrics import r2_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# lowercase, matching the pipeline's target_type column
ALL_TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
TARGET_IDX = {t: i for i, t in enumerate(ALL_TARGETS)}
TRUST_THRESHOLD = 0.05

# ── rdkit -> graph ───────────────────────────────────────────────────────────
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "Si", "P", "OTHER"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]


def one_hot(value, choices):
    vec = [0.0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec


def atom_features(atom):
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization().name
    feats = (
        one_hot(sym, ATOM_SYMBOLS)
        + one_hot(hyb, HYBRIDIZATIONS)
        + [
            atom.GetIsAromatic() * 1.0,
            atom.IsInRing() * 1.0,
            atom.GetDegree() / 4.0,
            atom.GetTotalNumHs() / 4.0,
            atom.GetFormalCharge() / 2.0,
        ]
    )
    return feats


BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def bond_features(bond):
    bt = bond.GetBondType().name
    feats = one_hot(bt, BOND_TYPES) + [
        bond.GetIsConjugated() * 1.0,
        bond.IsInRing() * 1.0,
    ]
    return feats


N_ATOM_FEATS = len(ATOM_SYMBOLS) + len(HYBRIDIZATIONS) + 5
N_BOND_FEATS = len(BOND_TYPES) + 2


def smiles_to_graph(smiles, target_idx=None, y=None, sample_weight=1.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_attr = [[0.0] * N_BOND_FEATS]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target_idx is not None:
        data.target_idx = torch.tensor([target_idx], dtype=torch.long)
        data.y = torch.tensor([y], dtype=torch.float)
        data.w = torch.tensor([sample_weight], dtype=torch.float)
    return data


# ── model: shared GINE trunk, per-target heads ──────────────────────────────
class GNNTrunk(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden=128, n_layers=4,
                 n_targets=len(ALL_TARGETS), dropout=0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(n_atom_feats, hidden)
        self.bond_encoder = nn.ModuleList([nn.Linear(n_bond_feats, hidden) for _ in range(n_layers)])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head_in = hidden * 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.head_in, 64), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(64, 1))
            for _ in range(n_targets)
        ])

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.atom_encoder(x)
        for conv, bn, bond_enc in zip(self.convs, self.bns, self.bond_encoder):
            e = bond_enc(edge_attr)
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        out = torch.cat([head(pooled) for head in self.heads], dim=1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--trust-frac", type=float, default=0.15)
    ap.add_argument("--base-dir", default=r"vault\pipeline_out")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT = args.out or os.path.join(args.base_dir, "gnn_arm")
    os.makedirs(OUT, exist_ok=True)

    dedup = pd.read_pickle(os.path.join(args.base_dir, "dedup.pkl"))
    test = pd.read_pickle(os.path.join(args.base_dir, "test.pkl"))
    print(f"device: {device}")
    print(f"train rows {len(dedup)}, test rows {len(test)}")
    print(dedup["target_type"].value_counts().to_string())

    # ---- assemble the objects the notebook logic expects ----
    # use raw `smiles` (polymer SMILES with * attachment atoms), NOT `canon`
    # (dedup key that strips * -> invalid `()` notation RDKit cannot parse;
    # using canon silently dropped ~34% of rows, biased to sparse targets).
    train = dedup.copy()
    train["canon_smiles"] = train["smiles"].values
    test["canon_smiles"] = test["smiles"].values
    # train_X only needs fold + target_type for this arm
    train_X = pd.DataFrame({"fold": train["fold"].values,
                            "target_type": train["target_type"].values},
                           index=train.index)
    # OOF array is aligned to train_X/dedup row order
    n_train = len(train)

    # per-target z-score stats (train only)
    target_stats = {}
    for t in ALL_TARGETS:
        vals = train.loc[train["target_type"] == t, "target"]
        target_stats[t] = (vals.mean(), vals.std() + 1e-9)

    def build_graph_list(df, has_target=True):
        graphs = []
        freq = df["target_type"].value_counts(normalize=True) if has_target else None
        for row_id, row in zip(df.index, df.itertuples()):
            ti = TARGET_IDX[row.target_type]
            if has_target:
                mean_, std_ = target_stats[row.target_type]
                y_norm = (row.target - mean_) / std_
                w = 1.0 / freq[row.target_type]
                g = smiles_to_graph(row.canon_smiles, target_idx=ti, y=y_norm, sample_weight=w)
            else:
                g = smiles_to_graph(row.canon_smiles, target_idx=ti, y=0.0, sample_weight=1.0)
            if g is not None:
                g.row_id = row_id
                graphs.append(g)
        return graphs

    t0 = time.time()
    train_graphs = build_graph_list(train, has_target=True)
    test_graphs = build_graph_list(test, has_target=False)
    print(f"Built {len(train_graphs)} train graphs, {len(test_graphs)} test graphs "
          f"({n_train - len(train_graphs)} train SMILES failed) in {time.time()-t0:.0f}s")

    # ---- train_gnn: OOF + trust scores, returns fold model states for test bag ----
    def train_gnn(epochs, batch_size, patience, trust_frac):
        return _train_gnn(train_graphs, train_X, n_train, target_stats,
                          epochs, batch_size, patience, trust_frac)


def _train_gnn(train_graphs, train_X, n_train, target_stats,
               epochs, batch_size, patience, trust_frac):
    row_to_graph = {g.row_id: g for g in train_graphs}
    oof = np.full(n_train, np.nan)
    trust_scores = {t: [] for t in ALL_TARGETS}
    fold_states = []   # list of (fold, best_state)
    for fold in sorted(train_X["fold"].unique()):
        fold_train_ids = train_X.index[train_X["fold"] != fold]
        val_ids = train_X.index[train_X["fold"] == fold]
        rng = np.random.RandomState(SEED + fold)
        trust_mask = rng.rand(len(fold_train_ids)) < trust_frac
        trust_ids = fold_train_ids[trust_mask]
        tr_ids = fold_train_ids[~trust_mask]
        tr_graphs = [row_to_graph[i] for i in tr_ids if i in row_to_graph]
        val_graphs = [row_to_graph[i] for i in val_ids if i in row_to_graph]
        trust_graphs = [row_to_graph[i] for i in trust_ids if i in row_to_graph]
        tr_loader = DataLoader(tr_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)

        model = GNNTrunk(N_ATOM_FEATS, N_BOND_FEATS).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

        best_val, bad_epochs, best_state = np.inf, 0, None
        epochs_used = 0
        ft0 = time.time()
        for epoch in range(epochs):
            epochs_used = epoch + 1
            model.train()
            for batch in tr_loader:
                batch = batch.to(device)
                opt.zero_grad()
                pred = model(batch)
                pred_sel = pred.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
                loss = (F.mse_loss(pred_sel, batch.y, reduction="none") * batch.w).mean()
                loss.backward()
                opt.step()
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    pred_sel = pred.gather(1, batch.target_idx.unsqueeze(1)).squeeze(1)
                    val_losses.append(F.mse_loss(pred_sel, batch.y).item())
            val_loss = np.mean(val_losses) if val_losses else np.inf
            sched.step(val_loss)
            if val_loss < best_val:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        oof_this_fold = {}
        with torch.no_grad():
            for g in val_graphs:
                gb = Batch.from_data_list([g]).to(device)
                pred = model(gb)
                ti = int(g.target_idx.item())
                mean_, std_ = target_stats[ALL_TARGETS[ti]]
                oof_this_fold[g.row_id] = pred[0, ti].item() * std_ + mean_
        for rid, val in oof_this_fold.items():
            oof[train_X.index.get_loc(rid)] = val

        with torch.no_grad():
            for g in trust_graphs:
                gb = Batch.from_data_list([g]).to(device)
                pred = model(gb)
                ti = int(g.target_idx.item())
                t_name = ALL_TARGETS[ti]
                mean_, std_ = target_stats[t_name]
                pred_val = pred[0, ti].item() * std_ + mean_
                true_val = g.y.item() * std_ + mean_
                trust_scores[t_name].append((true_val, pred_val))

        fold_states.append((fold, best_state))
        print(f"fold {fold}: best val MSE (norm) = {best_val:.4f} "
              f"({time.time()-ft0:.0f}s), epochs_used={epochs_used}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, trust_scores, fold_states

    gnn_oof, trust_scores, fold_states = train_gnn(
        args.epochs, args.batch, args.patience, args.trust_frac)

    # ---- honest OOF vs trust table ----
    print(f"\n{'target':<6} {'OOF R2':>10} {'trust R2':>10} {'gap':>8}")
    gap_flags = {}
    true_y = train["target"].values
    for t in ALL_TARGETS:
        mask = (train_X["target_type"] == t).values
        yt = true_y[mask]
        yp = gnn_oof[mask]
        fin = ~np.isnan(yp)
        oof_r2 = r2_score(yt[fin], yp[fin]) if fin.sum() >= 5 else np.nan
        pairs = trust_scores[t]
        if len(pairs) >= 5:
            y_true, y_pred = zip(*pairs)
            trust_r2 = r2_score(y_true, y_pred)
        else:
            trust_r2 = np.nan
        gap = oof_r2 - trust_r2 if not (np.isnan(oof_r2) or np.isnan(trust_r2)) else np.nan
        gap_flags[t] = gap
        print(f"{t:<6} {oof_r2:>10.4f} {trust_r2:>10.4f} {gap:>8.4f}")

    suspect = [t for t, g in gap_flags.items() if not np.isnan(g) and g > TRUST_THRESHOLD]
    print("\nTargets to exclude from meta-blend pending investigation:", suspect or "none")

    # ---- test arm: bag over fold models ----
    def predict_graphs_on(graphs, state):
        model = GNNTrunk(N_ATOM_FEATS, N_BOND_FEATS).to(device)
        model.load_state_dict(state)
        model.eval()
        preds = {}
        with torch.no_grad():
            for g in graphs:
                gb = Batch.from_data_list([g]).to(device)
                pred = model(gb)
                ti = int(g.target_idx.item())
                t_name = ALL_TARGETS[ti]
                mean_, std_ = target_stats[t_name]
                preds[g.row_id] = pred[0, ti].item() * std_ + mean_
        return preds

    print("\nComputing test predictions (bag over fold models)...")
    test_preds = {}
    for fold, state in fold_states:
        p = predict_graphs_on(test_graphs, state)
        for rid, v in p.items():
            test_preds[rid] = test_preds.get(rid, 0.0) + v / len(fold_states)
        del p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_rids = list(test.index)
    gnn_test_df = pd.DataFrame({
        "row_id": test_rids,
        "target_type": [test["target_type"][r] for r in test_rids],
        "gnn_test": [test_preds.get(r, np.nan) for r in test_rids],
    })

    gnn_oof_df = pd.DataFrame({
        "row_id": list(train_X.index),
        "target_type": train_X["target_type"].values,
        "gnn_oof": gnn_oof,
    })

    gnn_oof_df.to_csv(os.path.join(OUT, "gnn_oof.csv"), index=False)
    gnn_test_df.to_csv(os.path.join(OUT, "gnn_test.csv"), index=False)
    print(f"\nwrote {OUT}/gnn_oof.csv and {OUT}/gnn_test.csv")
    print(f"suspect targets: {suspect or 'none'}")

    # RMSE summary (like leaderboard)
    print("\nper-target OOF RMSE:")
    for t in ALL_TARGETS:
        mask = (train_X["target_type"] == t).values
        yp = gnn_oof[mask]
        fin = ~np.isnan(yp)
        rmse = float(np.sqrt(np.mean((true_y[mask][fin] - yp[fin]) ** 2))) if fin.sum() else np.nan
        print(f"  {t:<6} {rmse:.4f}")


if __name__ == "__main__":
    main()