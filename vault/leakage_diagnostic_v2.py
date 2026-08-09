"""Leakage exploitation diagnostic v2 — backbone-based matching.

Polymer SMILES have * as attachment points. Two SMILES with * at different
positions may represent the same polymer. Strategy:
1. Strip all * from SMILES to get backbone
2. Canonicalize backbone (no attachment points)
3. Match on backbone + target_type
4. Also try raw string matching (no canonicalization) as sanity check
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from collections import Counter

TRAIN = r"D:\Parth\ploywin r2\official_dataset\train.csv"
TEST = r"D:\Parth\ploywin r2\official_dataset\test.csv"
SUPERBLEND = r"D:\Parth\ploywin r2\vault\pipeline_out_pretrain\superblend_oof.npz"

print("Loading data...")
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
print(f"Train: {len(train)} rows, Test: {len(test)} rows")

# === Approach 1: Raw string matching (no canonicalization) ===
print(f"\n{'='*60}")
print("APPROACH 1: Raw SMILES string matching")
print(f"{'='*60}")

train_raw = set(zip(train["smiles"], train["target_type"]))
test_keys_raw = list(zip(test["smiles"], test["target_type"]))
matches_raw = sum(1 for k in test_keys_raw if k in train_raw)
print(f"Raw string matches: {matches_raw}/{len(test)} ({100*matches_raw/len(test):.1f}%)")

# === Approach 2: Backbone matching (strip *, canonicalize) ===
print(f"\n{'='*60}")
print("APPROACH 2: Backbone matching (strip *, canonicalize)")
print(f"{'='*60}")

def get_backbone(smiles):
    """Strip all * from SMILES and canonicalize the backbone."""
    s = smiles.replace("*", "")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

train["backbone"] = train["smiles"].apply(get_backbone)
test["backbone"] = test["smiles"].apply(get_backbone)

n_fail_train = train["backbone"].isna().sum()
n_fail_test = test["backbone"].isna().sum()
print(f"Failed backbone canonicalization: train={n_fail_train}, test={n_fail_test}")

train = train.dropna(subset=["backbone"]).reset_index(drop=True)
test = test.dropna(subset=["backbone"]).reset_index(drop=True)

# Build lookup
backbone_lookup = {}
for _, row in train.iterrows():
    key = (row["backbone"], row["target_type"])
    if key not in backbone_lookup:
        backbone_lookup[key] = []
    backbone_lookup[key].append((row["smiles"], row["target"]))

# Find matches
test_matches = []
for _, row in test.iterrows():
    key = (row["backbone"], row["target_type"])
    if key in backbone_lookup:
        vals = backbone_lookup[key]
        test_matches.append({
            "id": row["id"],
            "test_smiles": row["smiles"],
            "backbone": row["backbone"],
            "target_type": row["target_type"],
            "n_train_matches": len(vals),
            "train_labels": [v[1] for v in vals],
            "train_smiles": [v[0] for v in vals],
        })

match_df = pd.DataFrame(test_matches)
print(f"\nBackbone matches: {len(match_df)}/{len(test)} ({100*len(match_df)/len(test):.1f}%)")

# Per-target
print(f"\nPer-target backbone matches:")
for tgt in sorted(test["target_type"].unique()):
    n_test = len(test[test["target_type"] == tgt])
    n_match = len(match_df[match_df["target_type"] == tgt])
    print(f"  {tgt}: {n_match}/{n_test} ({100*n_match/n_test:.1f}%)")

# Show some examples
if len(match_df) > 0:
    print(f"\nSample matches (first 10):")
    for _, m in match_df.head(10).iterrows():
        labels = m["train_labels"]
        print(f"  id={m['id']}, target={m['target_type']}, "
              f"n_matches={m['n_train_matches']}, train_labels={labels}")

# === Approach 3: Exact SMILES match (raw, no canonicalization) ===
print(f"\n{'='*60}")
print("APPROACH 3: Exact raw SMILES + target_type match")
print(f"{'='*60}")

raw_lookup = {}
for _, row in train.iterrows():
    key = (row["smiles"], row["target_type"])
    if key not in raw_lookup:
        raw_lookup[key] = []
    raw_lookup[key].append(row["target"])

raw_matches = []
for _, row in test.iterrows():
    key = (row["smiles"], row["target_type"])
    if key in raw_lookup:
        raw_matches.append({
            "id": row["id"],
            "smiles": row["smiles"],
            "target_type": row["target_type"],
            "n_train": len(raw_lookup[key]),
            "train_labels": raw_lookup[key],
        })

raw_match_df = pd.DataFrame(raw_matches)
print(f"Exact raw matches: {len(raw_match_df)}/{len(test)} ({100*len(raw_match_df)/len(test):.1f}%)")
for tgt in sorted(test["target_type"].unique()):
    n_test = len(test[test["target_type"] == tgt])
    n_m = len(raw_match_df[raw_match_df["target_type"] == tgt])
    print(f"  {tgt}: {n_m}/{n_test} ({100*n_m/n_test:.1f}%)")

if len(raw_match_df) > 0:
    print(f"\nSample raw matches:")
    for _, m in raw_match_df.head(5).iterrows():
        print(f"  id={m['id']}, target={m['target_type']}, labels={m['train_labels']}, smiles={m['smiles'][:60]}...")

# === Approach 4: Count duplicate SMILES across train and test ===
print(f"\n{'='*60}")
print("APPROACH 4: SMILES overlap analysis")
print(f"{'='*60}")
train_smiles_set = set(train["smiles"].unique())
test_smiles_set = set(test["smiles"].unique())
overlap = train_smiles_set & test_smiles_set
print(f"Unique train SMILES: {len(train_smiles_set)}")
print(f"Unique test SMILES: {len(test_smiles_set)}")
print(f"Overlap (same raw SMILES): {len(overlap)}")
print(f"Overlap % of test: {100*len(overlap)/len(test_smiles_set):.1f}%")

# Check backbone overlap
train_backbones = set(train["backbone"].unique())
test_backbones = set(test["backbone"].unique())
bb_overlap = train_backbones & test_backbones
print(f"\nUnique train backbones: {len(train_backbones)}")
print(f"Unique test backbones: {len(test_backbones)}")
print(f"Backbone overlap: {len(bb_overlap)}")
print(f"Backbone overlap % of test: {100*len(bb_overlap)/len(test_backbones):.1f}%")
