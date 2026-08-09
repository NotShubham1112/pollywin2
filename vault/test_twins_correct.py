"""The v7 retrieval design claimed 17% test twins. Let me find them with the RIGHT method.

Two polymers are 'twins' if their backbone (ignoring attachment points) is the same.
"""
import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd, numpy as np

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

def backbone(s):
    """Strip attachment points and canonicalize."""
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        # Remove dummy atoms [*]
        rw = Chem.RWMol(m)
        atoms_to_remove = []
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atom (e.g., *)
                atoms_to_remove.append(atom.GetIdx())
        for idx in sorted(atoms_to_remove, reverse=True):
            rw.RemoveAtom(idx)
        return Chem.MolToSmiles(rw, canonical=True)
    except: return None

train["bb"] = train["smiles"].apply(backbone)
test["bb"]  = test["smiles"].apply(backbone)
print("Failed:", train["bb"].isna().sum(), test["bb"].isna().sum())
print("Unique bb train:", train["bb"].nunique(), "test:", test["bb"].nunique())

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
train_lookup = {}
for _, row in train.dropna(subset=["bb"]).iterrows():
    train_lookup.setdefault((row["bb"], row["target_type"]), []).append(row["target"])

print("\nTest rows with TRUE twin (same backbone + same target_type):")
total = 0
test_v = test.dropna(subset=["bb"])
for tt in TARGETS:
    sub = test_v[test_v["target_type"]==tt]
    n_match = sum(1 for b in sub["bb"] if (b, tt) in train_lookup)
    print(f"  {tt}: {n_match}/{len(sub)} = {100*n_match/len(sub):.1f}%")
    total += n_match
print(f"TOTAL: {total}/{len(test_v)} = {100*total/len(test_v):.1f}%")

# Just backbone match (any target_type)
train_bb_set = set(train["bb"].dropna())
print(f"\nTest rows sharing backbone with ANY train row: {int(test_v['bb'].isin(train_bb_set).sum())}/{len(test_v)} = {100*test_v['bb'].isin(train_bb_set).mean():.1f}%")

# Train twin rows (same backbone + same target)
train_twin = train.groupby(["bb","target_type"]).size()
print(f"\nTrain twin groups: {(train_twin>1).sum()}/{len(train_twin)} have >1 row")
print(f"Total train twin rows: {train_twin[train_twin>1].sum()}/{len(train)}")

# P14 OOF on twin rows
import numpy as np
from sklearn.metrics import r2_score
npz = np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)
oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
oof_mt  = np.asarray(npz["oof_mt"], dtype=float)
y       = np.asarray(npz["y_train"], dtype=float)
t_arr   = np.asarray(npz["target_type_train"])

# Align bb to train (assumes same order)
train["bb_aligned"] = train["bb"].values
is_twin = train_twin[train_twin > 1].reindex(list(zip(train["bb"], train["target_type"])), fill_value=0).values > 1
print(f"\nTrain twin rows (same bb + target): {is_twin.sum()}/{len(train)}")

# Per-target OOF R^2 on twin vs non-twin
p14 = 0.5*oof_gbm + 0.5*oof_mt
print("\nP14 OOF R^2 on twin vs non-twin:")
for tt in TARGETS:
    idx = np.where(t_arr == tt)[0]
    twin_idx = idx[is_twin[idx]]
    ntwin_idx = idx[~is_twin[idx]]
    if len(twin_idx) > 5:
        r2_twin = r2_score(y[twin_idx], p14[twin_idx])
    else: r2_twin = float("nan")
    if len(ntwin_idx) > 5:
        r2_nt = r2_score(y[ntwin_idx], p14[ntwin_idx])
    else: r2_nt = float("nan")
    print(f"  {tt:<4}: twin n={len(twin_idx)} R2={r2_twin:.4f} | non-twin n={len(ntwin_idx)} R2={r2_nt:.4f}")
