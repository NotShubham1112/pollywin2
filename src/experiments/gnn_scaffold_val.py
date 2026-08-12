"""GNN-only scaffold-split validator.

Purpose: re-validate the GNN arm under a chemistry-aware OOF instead of the
canon-grouped GroupKFold used by `gnn_arm_run.py`. The real train->test shift we
quantified is structural (novel Bemis-Murcko scaffolds in test, esp. tg 31.8%,
egc 11.9%). A random/group-GKFold can overstate generalisation if the solver is
just memorising scaffolds. Here we split *by scaffold* (StratifiedGroupKFold,
groups = scaffold, stratified by target) so every fold sees only scaffolds the
model has never trained on -> an honest upper-bound-free estimate.

Steps:
  1. compute Bemis-Murcko scaffold per train row (from `smiles`, the correct col)
  2. StratifiedGroupKFold(10, shuffle, seed) keyed on scaffold -> scaffold_folds.csv
  3. retrain the shared GNN arm on those folds (imports _train_gnn from gnn_arm_run)
  4. emit random-vs-scaffold OOF R2 comparison per target

Usage (Miniconda3 python has torch+cuda+pyg+rdkit+sklearn 1.9):
  python gnn_scaffold_val.py [--epochs N] [--out DIR]
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score

from gnn_arm_run import (
    ALL_TARGETS, TARGET_IDX, SEED, device,
    N_ATOM_FEATS, N_BOND_FEATS, smiles_to_graph, _train_gnn,
)


def scaffold(mol):
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--trust-frac", type=float, default=0.15)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--base-dir", default=r"vault\pipeline_out")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT = args.out or os.path.join(args.base_dir, "gnn_arm")
    os.makedirs(OUT, exist_ok=True)
    dedup = pd.read_pickle(os.path.join(args.base_dir, "dedup.pkl"))
    print(f"device: {device}; train rows {len(dedup)}")

    train = dedup.copy()
    train["canon_smiles"] = train["smiles"].values

    # ---- 1. scaffolds ----
    t0 = time.time()
    scaff = train["smiles"].map(lambda s: scaffold(Chem.MolFromSmiles(s)))
    n_scaffold = scaff.notna().sum()
    n_unique = scaff.nunique()
    print(f"scaffolds computed in {time.time()-t0:.0f}s: {n_scaffold}/{len(train)} rows, "
          f"{n_unique} unique scaffolds")
    bad = train["smiles"][scaff.isna()].index.tolist()
    if bad:
        print(f"  NOTE: {len(bad)} rows got no scaffold: {bad[:10]}")

    # ---- 2. scaffold-grouped stratified folds ----
    y_strat = train["target_type"].astype("category").cat.codes.values
    groups = train.index  # rows are unique when identical scaffold -> same fold
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    fold_assign = np.full(len(train), -1, dtype=int)
    valid = scaff.notna().values
    # group key: scaffold string (rows sharing a scaffold share a fold)
    gkey = pd.factorize(scaff.fillna("__NA__"))[0]
    for fold_i, (_, te) in enumerate(sgkf.split(np.zeros(len(train)), y_strat, gkey)):
        fold_assign[te] = fold_i
    # rows with no scaffold (should be none, aisle) -> fold -1 (excluded)
    n_excl = int((fold_assign < 0).sum())
    train["scaffold"] = scaff.values
    train["scaffold_fold"] = fold_assign
    print(f"scaffold folds assigned (rows per fold): "
          f"{np.bincount(fold_assign[fold_assign >= 0]).tolist()}; excluded {n_excl}")

    # per-fold target coverage check
    cov = pd.crosstab(train["scaffold_fold"], train["target_type"]).reindex(
        index=[f for f in range(args.n_splits)], columns=ALL_TARGETS, fill_value=0)
    missing = cov[cov < 3].stack()
    if len(missing):
        print("WARN folds with <3 rows of a target:\n", missing.head(20).to_string())
    else:
        print("OK: every fold has >=3 rows of every target")

    cov.to_csv(os.path.join(OUT, "scaffold_fold_coverage.csv"))
    train[["scaffold", "scaffold_fold"]].to_csv(os.path.join(OUT, "scaffold_folds.csv"))

    # ---- 3. retrain on scaffold folds ----
    train_X = pd.DataFrame({"fold": train["scaffold_fold"].values,
                            "target_type": train["target_type"].values},
                           index=train.index)
    n_train = len(train)
    target_stats = {}
    for t in ALL_TARGETS:
        vals = train.loc[train["target_type"] == t, "target"]
        target_stats[t] = (vals.mean(), vals.std() + 1e-9)

    def build_graph_list(df):
        graphs = []
        freq = df["target_type"].value_counts(normalize=True)
        for row_id, row in zip(df.index, df.itertuples()):
            ti = TARGET_IDX[row.target_type]
            mean_, std_ = target_stats[row.target_type]
            y_norm = (row.target - mean_) / std_
            w = 1.0 / freq[row.target_type]
            g = smiles_to_graph(row.canon_smiles, target_idx=ti, y=y_norm, sample_weight=w)
            if g is not None:
                g.row_id = row_id
                graphs.append(g)
        return graphs

    t0 = time.time()
    train_graphs = build_graph_list(train)
    print(f"Built {len(train_graphs)} train graphs in {time.time()-t0:.0f}s")

    gnn_oof_scaffold, trust_scores, fold_states = _train_gnn(
        train_graphs, train_X, n_train, target_stats,
        args.epochs, args.batch, args.patience, args.trust_frac)

    # ---- 4. random vs scaffold comparison ----
    random_df = pd.read_csv(os.path.join(OUT, "gnn_oof.csv"))
    true_y = train["target"].values
    rows, x = [], 0.0
    print(f"\n{'target':<6} {'randomOOF':>10} {'scaffoldOOF':>12} {'dR2':>9}   shift%test")
    for t in ALL_TARGETS:
        mask = (train_X["target_type"] == t).values
        yt = true_y[mask]

        m = (random_df["target_type"] == t).values
        row_ids = train_X.index[mask]
        yp_rand = random_df.set_index("row_id").loc[row_ids, "gnn_oof"].values
        yp_scaf = gnn_oof_scaffold[mask]

        fin_r = ~np.isnan(yp_rand)
        fin_s = ~np.isnan(yp_scaf)
        rr = r2_score(yt[fin_r], yp_rand[fin_r]) if fin_r.sum() >= 5 else np.nan
        rs = r2_score(yt[fin_s], yp_scaf[fin_s]) if fin_s.sum() >= 5 else np.nan
        rows.append((t, rr, rs))
        print(f"{t:<6} {rr if not np.isnan(rr) else float('nan'):>10.4f} "
              f"{rs if not np.isnan(rs) else float('nan'):>12.4f} "
              f"{rs-rr if not (np.isnan(rr) or np.isnan(rs)) else float('nan'):>9.4f}")

    n_mean_r = np.mean([r for _, r, _ in rows if not np.isnan(r)])
    n_mean_s = np.mean([s for _, _, s in rows if not np.isnan(s)])
    print(f"\nmean over targets: random {n_mean_r:.4f} vs scaffold {n_mean_s:.4f}")

    pd.DataFrame([(t, r, s) for t, r, s in rows],
                 columns=["target", "random_oof_r2", "scaffold_oof_r2"]).round(4)\
        .to_csv(os.path.join(OUT, "gnn_random_vs_scaffold.csv"), index=False)
    print(f"\nwrote {OUT}/scaffold_folds.csv, scaffold_fold_coverage.csv, "
          f"gnn_random_vs_scaffold.csv, gnn_oof_scaffold.csv")


if __name__ == "__main__":
    main()