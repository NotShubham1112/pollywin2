import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np, pandas as pd
from rdkit import Chem

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

# Match test SMILES (raw) with train SMILES (raw, any target)
train_smiles_to_target = {}
for _, row in train.iterrows():
    train_smiles_to_target.setdefault(row["smiles"], []).append((row["target_type"], row["target"]))

# For each test row, find the train siblings
test_sibs = []
for _, row in test.iterrows():
    if row["smiles"] in train_smiles_to_target:
        for tt, val in train_smiles_to_target[row["smiles"]]:
            if tt != row["target_type"]:
                test_sibs.append({"id": row["id"], "test_type": row["target_type"], "sib_type": tt, "sib_value": val})
                break
    else:
        test_sibs.append({"id": row["id"], "test_type": row["target_type"], "sib_type": None, "sib_value": None})
df = pd.DataFrame(test_sibs)
print(f"Test rows with >=1 cross-target sibling: {df['sib_type'].notna().sum()}/{len(test)} ({100*df['sib_type'].notna().mean():.1f}%)")
print()
print("Per test target_type:")
print(df.groupby("test_type")["sib_type"].apply(lambda s: int(s.notna().sum())))
print()
print("Sibling types distribution per test_type:")
print(df.groupby("test_type")["sib_type"].value_counts())

df.to_csv(os.path.join(WORK, "vault", "test_cross_target_siblings.csv"), index=False)
print("Saved.")
