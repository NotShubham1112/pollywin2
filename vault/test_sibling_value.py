"""The v7 design's 17% was likely measured by considering SMILES with same InChI or same backbone+attachment-points.
Let me try yet another approach: use InChI block 1 + dummy removal.

Actually, the v7 design used `canon_key` which is the canonical SMILES with [*] -> [X]. 
This works when [*] is preserved as an atom. But [*] -> [X] (atom type 0) may not be recognized by RDKit canonical SMILES.

Let me check the v7 design code:
"""
import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
import pandas as pd, numpy as np

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

# Try the v7 design's "canon" with [*] treated as atomic number 0 directly
def canon_v7(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        return Chem.MolToSmiles(m, canonical=True)
    except: return None

train["c"] = train["smiles"].apply(canon_v7)
test["c"]  = test["smiles"].apply(canon_v7)

# v7 design uses canon_key with [*] -> [X] BEFORE parsing
def canon_v7b(s):
    try:
        s2 = s.replace("[*]", "[X]").replace("*", "[X]")
        m = Chem.MolFromSmiles(s2)
        if m is None: return None
        return Chem.MolToSmiles(m, canonical=True)
    except: return None

train["cb"] = train["smiles"].apply(canon_v7b)
test["cb"]  = test["smiles"].apply(canon_v7b)

print("Unique canon (default [*]): train", train["c"].nunique(), "test", test["c"].nunique())
print("Unique canon ([*]->[X]): train", train["cb"].nunique(), "test", test["cb"].nunique())

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
for col, name in [("c", "default"), ("cb", "[*]->[X]")]:
    train_v = train.dropna(subset=[col])
    test_v  = test.dropna(subset=[col])
    train_lookup = {}
    for _, row in train_v.iterrows():
        train_lookup.setdefault((row[col], row["target_type"]), []).append(row["target"])
    total = 0
    print(f"\n=== {name}: twin (same {col} + same target_type) ===")
    for tt in TARGETS:
        sub = test_v[test_v["target_type"]==tt]
        n_match = sum(1 for k in sub[col] if (k, tt) in train_lookup)
        print(f"  {tt}: {n_match}/{len(sub)} = {100*n_match/len(sub):.1f}%")
        total += n_match
    print(f"  TOTAL: {total}/{len(test_v)} = {100*total/len(test_v):.1f}%")

# Also: cross-target sibling count (any train twin with different target_type)
print("\n=== Cross-target sibling availability ===")
for col, name in [("c", "default"), ("cb", "[*]->[X]")]:
    train_v = train.dropna(subset=[col])
    test_v = test.dropna(subset=[col])
    sib_count = {}
    for tt in TARGETS:
        sub = test_v[test_v["target_type"]==tt]
        n = 0
        for c in sub[col]:
            for tt2 in train_v[ train_v[col]==c ]["target_type"].unique():
                if tt2 != tt:
                    n += 1
                    break
        sib_count[tt] = n
    print(f"\n--- {name}: test rows with cross-target sibling ---")
    for tt in TARGETS:
        sub = test_v[test_v["target_type"]==tt]
        print(f"  {tt}: {sib_count[tt]}/{len(sub)} = {100*sib_count[tt]/len(sub):.1f}%")
