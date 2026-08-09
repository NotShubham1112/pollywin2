"""KEY INSIGHT: For test rows with TRUE twin in train (exact backbone+target_type match),
we should use the TRUE value, not the model prediction.

We saw earlier: backbone+target_type twin match is rare (0-2% per target) using backbone stripping.
But using FULL SMILES (with *) + canonical SMILES match, we should get 17% per prior research.

Let me re-check using the canonical full-SMILES approach (matching what v14/v16 actually do).
"""
import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np, pandas as pd
from rdkit import Chem

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

# Try matching with [*] (full SMILES, attachment points preserved)
def canon_full(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None
train["canon_full"] = train["smiles"].apply(canon_full)
test["canon_full"]  = test["smiles"].apply(canon_full)
train = train.dropna(subset=["canon_full"])
test  = test.dropna(subset=["canon_full"])

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]

# Build lookup: (canon_full, target_type) -> list of train target values
lookup = {}
for _, row in train.iterrows():
    key = (row["canon_full"], row["target_type"])
    lookup.setdefault(key, []).append(row["target"])

# Match on FULL canon (same SMILES including *)
print("=== Full canon (incl. *) + target_type twin match ===")
for tt in TARGETS:
    sub = test[test["target_type"]==tt]
    n_match = 0
    for c in sub["canon_full"]:
        if (c, tt) in lookup:
            n_match += 1
    print(f"  {tt}: {n_match}/{len(sub)} = {100*n_match/len(sub):.1f}% test rows have TRUE train label")

# Now compute the ESTIMATED LB GAIN from label replacement on these matched rows.
# Approach: estimate P14 OOF error on rows with known siblings.
# If P14 OOF error on these rows is high, replacement with TRUE values would help a lot.
import os
npz = np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)
oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
oof_mt  = np.asarray(npz["oof_mt"], dtype=float)
y       = np.asarray(npz["y_train"], dtype=float)
t_arr   = np.asarray(npz["target_type_train"])

# We need to know which train rows have a TWIN (i.e., another train row with same canon + same target_type)
# Build a map: canon -> set of target_types in train
canon_to_tts = train.groupby("canon_full")["target_type"].apply(set).to_dict()

# For each train row, does it have a TWIN (another row with same canon + target_type)?
is_twin = np.zeros(len(train), dtype=bool)
for i, row in train.reset_index(drop=True).iterrows():
    key = (row["canon_full"], row["target_type"])
    if key in lookup and len(lookup[key]) > 1:
        is_twin[i] = True

print(f"\n=== Twin rows in TRAIN (multi-rows with same canon + target_type) ===")
print(f"Total twin rows: {is_twin.sum()}/{len(train)} = {100*is_twin.mean():.1f}%")
for tt in TARGETS:
    idx = np.where(t_arr == tt)[0]
    n_twin = is_twin[idx].sum()
    print(f"  {tt}: {n_twin}/{len(idx)} = {100*n_twin/len(idx):.1f}%")

# How well does P14 predict twin rows vs non-twin rows?
print(f"\n=== P14 OOF R^2 on twin vs non-twin rows ===")
p14 = 0.5*oof_gbm + 0.5*oof_mt
for tt in TARGETS:
    idx = np.where(t_arr == tt)[0]
    twin_idx = idx[is_twin[idx]]
    ntwin_idx = idx[~is_twin[idx]]
    if len(twin_idx) > 5:
        from sklearn.metrics import r2_score
        r2_twin = r2_score(y[twin_idx], p14[twin_idx])
    else: r2_twin = float("nan")
    if len(ntwin_idx) > 5:
        r2_nt = r2_score(y[ntwin_idx], p14[ntwin_idx])
    else: r2_nt = float("nan")
    print(f"  {tt:<4}: twin n={len(twin_idx)} R2={r2_twin:.4f} | non-twin n={len(ntwin_idx)} R2={r2_nt:.4f}")
