import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
import pandas as pd, numpy as np

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

# Use InChI but strip the connection table portion to handle attachment points
# Standard InChI for a polymer with [*] is broken, so use the 'fixedH' or skip polymer layer
def inchi_block1(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        inchi = Chem.MolToInchi(m)
        # Block 1 = connectivity only (no stereo, no polymer layer)
        if inchi is None: return None
        return inchi.split("-")[0]  # only the connectivity block
    except: return None

train["ik1"] = train["smiles"].apply(inchi_block1)
test["ik1"]  = test["smiles"].apply(inchi_block1)
print("Failed:", train["ik1"].isna().sum(), test["ik1"].isna().sum())
print("Unique ik1 train:", train["ik1"].nunique(), "test:", test["ik1"].nunique())

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
train_lookup = {}
for _, row in train.dropna(subset=["ik1"]).iterrows():
    train_lookup.setdefault((row["ik1"], row["target_type"]), []).append(row["target"])

print("\nTest rows with TRUE twin (ik1 = InChI block 1 only + same target_type):")
total = 0
test_v = test.dropna(subset=["ik1"])
for tt in TARGETS:
    sub = test_v[test_v["target_type"]==tt]
    n_match = sum(1 for k in sub["ik1"] if (k, tt) in train_lookup)
    print(f"  {tt}: {n_match}/{len(sub)} = {100*n_match/len(sub):.1f}%")
    total += n_match
print(f"TOTAL: {total}/{len(test_v)} = {100*total/len(test_v):.1f}%")

# Train twin rows (same ik1 + same target)
train_twin = train.groupby(["ik1","target_type"]).size()
print(f"\nTrain twin groups: {(train_twin>1).sum()}/{len(train_twin)} have >1 row")
print(f"Total train twin rows: {train_twin[train_twin>1].sum()}/{len(train)}")
