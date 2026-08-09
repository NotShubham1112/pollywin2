import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
import pandas as pd

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

def canon_dummy(s):
    s2 = s.replace("[*]", "[*:0]")
    m = Chem.MolFromSmiles(s2)
    if m is None:
        s2 = s.replace("[*]", "[He]").replace("*", "[He]")
        m = Chem.MolFromSmiles(s2)
    return Chem.MolToSmiles(m, canonical=True) if m else None

train["canon_d"] = train["smiles"].apply(canon_dummy)
test["canon_d"]  = test["smiles"].apply(canon_dummy)
print("Failed canon_d:", train["canon_d"].isna().sum(), test["canon_d"].isna().sum())
print("Unique canon_d: train", train["canon_d"].nunique(), "test", test["canon_d"].nunique())

print("Examples:")
for i in range(5):
    raw = train["smiles"].iloc[i]
    canon = train["canon_d"].iloc[i]
    print(f"  {raw[:60]} -> {canon}")

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
lookup = {}
for _, row in train.dropna(subset=["canon_d"]).iterrows():
    key = (row["canon_d"], row["target_type"])
    lookup.setdefault(key, []).append(row["target"])

print("\nTest rows with TRUE twin in train:")
total_match = 0
for tt in TARGETS:
    sub = test.dropna(subset=["canon_d"])
    sub = sub[sub["target_type"]==tt]
    n_match = sum(1 for c in sub["canon_d"] if (c, tt) in lookup)
    print(f"  {tt}: {n_match}/{len(sub)} = {100*n_match/len(sub):.1f}%")
    total_match += n_match
print(f"TOTAL: {total_match}/{len(test.dropna(subset=['canon_d']))} = {100*total_match/len(test.dropna(subset=['canon_d'])):.1f}%")

# Train rows that have a twin (same canon + same target)
train_twin = train.groupby(["canon_d","target_type"]).size()
n_twin = (train_twin > 1).sum()
print(f"\nTrain twin rows: {n_twin}/{len(train_twin)} canon-target pairs have >1 row")
