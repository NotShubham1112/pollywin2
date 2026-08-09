import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
import pandas as pd, numpy as np

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

def inchikey(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        return Chem.MolToInchiKey(m)
    except: return None

train["ik"] = train["smiles"].apply(inchikey)
test["ik"]  = test["smiles"].apply(inchikey)
print("Failed InChIKey:", train["ik"].isna().sum(), test["ik"].isna().sum())

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
train_lookup = {}
for _, row in train.dropna(subset=["ik"]).iterrows():
    train_lookup.setdefault((row["ik"], row["target_type"]), []).append(row["target"])

print("\nTest rows with TRUE twin (same InChIKey + same target_type):")
total_match = 0
test_v = test.dropna(subset=["ik"])
for tt in TARGETS:
    sub = test_v[test_v["target_type"]==tt]
    n_match = sum(1 for k in sub["ik"] if (k, tt) in train_lookup)
    print(f"  {tt}: {n_match}/{len(sub)} = {100*n_match/len(sub):.1f}%")
    total_match += n_match
print(f"TOTAL: {total_match}/{len(test_v)} = {100*total_match/len(test_v):.1f}%")

train_ik_set = set(train["ik"].dropna())
print(f"\nTest rows sharing InChIKey with ANY train row: {int(test_v['ik'].isin(train_ik_set).sum())}/{len(test_v)} = {100*test_v['ik'].isin(train_ik_set).mean():.1f}%")

# Per-target breakdown of 'any twin' (sibling indicator)
for tt in TARGETS:
    sub = test_v[test_v["target_type"]==tt]
    n_tw = int(sub["ik"].isin(train_ik_set).sum())
    print(f"  {tt}: {n_tw}/{len(sub)} = {100*n_tw/len(sub):.1f}%")
