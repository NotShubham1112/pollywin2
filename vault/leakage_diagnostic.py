"""Leakage exploitation diagnostic for P14.

Steps:
1. Canonicalize SMILES in train/test
2. Find exact-match rows (same canonical SMILES + same target_type)
3. Report per-target match counts and label conflicts
4. Compute P14's residual on match rows using superblend OOF
"""
import numpy as np
import pandas as pd
from rdkit import Chem

TRAIN = r"D:\Parth\ploywin r2\official_dataset\train.csv"
TEST = r"D:\Parth\ploywin r2\official_dataset\test.csv"
SUPERBLEND = r"D:\Parth\ploywin r2\vault\pipeline_out_pretrain\superblend_oof.npz"

def canonicalize(smiles):
    """Canonicalize SMILES, replacing attachment point markers consistently."""
    s = smiles.strip()
    # Replace [*] and * with a consistent placeholder
    s = s.replace("[*]", "[X]")
    s = s.replace("*", "[X]")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        # Try without modification
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

print("Loading data...")
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
print(f"Train: {len(train)} rows, Test: {len(test)} rows")

print("\nCanonicalizing SMILES...")
train["canon"] = train["smiles"].apply(canonicalize)
test["canon"] = test["smiles"].apply(canonicalize)

n_fail_train = train["canon"].isna().sum()
n_fail_test = test["canon"].isna().sum()
print(f"Failed canonicalization: train={n_fail_train}, test={n_fail_test}")

# Drop failed rows
train = train.dropna(subset=["canon"]).reset_index(drop=True)
test = test.dropna(subset=["canon"]).reset_index(drop=True)

# Build lookup: (canon_smiles, target_type) -> list of train values
lookup = {}
for _, row in train.iterrows():
    key = (row["canon"], row["target_type"])
    if key not in lookup:
        lookup[key] = []
    lookup[key].append(row["target"])

# Find matches in test
test_matches = []
for _, row in test.iterrows():
    key = (row["canon"], row["target_type"])
    if key in lookup:
        vals = lookup[key]
        test_matches.append({
            "id": row["id"],
            "canon": row["canon"],
            "target_type": row["target_type"],
            "n_train_matches": len(vals),
            "train_mean": np.mean(vals),
            "train_vals": vals,
        })

match_df = pd.DataFrame(test_matches)
print(f"\n{'='*60}")
print(f"EXACT-MATCH DIAGNOSTIC RESULTS")
print(f"{'='*60}")
print(f"Total test rows: {len(test)}")
print(f"Exact-match rows: {len(match_df)} ({100*len(match_df)/len(test):.1f}%)")

# Per-target breakdown
print(f"\nPer-target matches:")
for tgt in sorted(test["target_type"].unique()):
    n_test = len(test[test["target_type"] == tgt])
    n_match = len(match_df[match_df["target_type"] == tgt])
    print(f"  {tgt}: {n_match}/{n_test} ({100*n_match/n_test:.1f}%)")

# Check for label conflicts (same SMILES+target, different values)
print(f"\nLabel conflict check:")
conflicts = 0
for key, vals in lookup.items():
    if len(vals) > 1:
        unique_vals = set(vals)
        if len(unique_vals) > 1:
            conflicts += 1
            if conflicts <= 5:
                print(f"  {key}: {vals} (conflict!)")
print(f"Total conflicting keys: {conflicts} / {len(lookup)}")

# Now compute P14's residual on match rows using superblend OOF
print(f"\n{'='*60}")
print(f"P14 RESIDUAL ANALYSIS (using superblend OOF)")
print(f"{'='*60}")

npz = np.load(SUPERBLEND, allow_pickle=True)
oof_gbm = npz["oof_gbm"]
oof_mt = npz["oof_mt"]
y = npz["y_train"].astype(np.float64)
tt = npz["target_type_train"].astype(str)

# Compute Ridge-blended OOF for each target (using the blend weights from eval_blend)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
idx_all = {t: np.where(tt == t)[0] for t in TARGETS}

# Train Ridge blender per target to get blended OOF
blended_oof = np.zeros(len(y))
ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
for t in TARGETS:
    ix = idx_all[t]
    yt = y[ix]
    Mx = np.column_stack([oof_gbm[ix], oof_mt[ix]])
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(Mx, yt))
    best, besta = -np.inf, 1.0
    for a in ALPHA_GRID:
        o = np.zeros(len(ix))
        for trk, vk in cv:
            o[vk] = Ridge(alpha=a).fit(Mx[trk], yt[trk]).predict(Mx[vk])
        r = r2_score(yt, o)
        if r > best:
            best, besta = r, a
    lr = Ridge(alpha=besta)
    lr.fit(Mx, yt)
    blended_oof[ix] = lr.predict(Mx)

# Now: find train rows that are exact matches to test rows
# For each such train row, compute residual = blended_oof - y
# This tells us how well P14 already predicts on "leaked" rows

# Build reverse lookup: for each test match, find corresponding train indices
print("\nMatching train OOF to test match rows...")
train_idx_lookup = {}
for i, (_, row) in enumerate(train.iterrows()):
    key = (row["canon"], row["target_type"])
    if key not in train_idx_lookup:
        train_idx_lookup[key] = []
    train_idx_lookup[key].append(i)

# For matched test rows, find train indices with same key
matched_train_indices = []
matched_test_ids = []
for _, mrow in match_df.iterrows():
    key = (mrow["canon"], mrow["target_type"])
    if key in train_idx_lookup:
        # Take the first match (or average later)
        for ti in train_idx_lookup[key]:
            matched_train_indices.append(ti)
            matched_test_ids.append(mrow["id"])

matched_train_indices = np.array(matched_train_indices)
print(f"Matched train indices: {len(matched_train_indices)}")

if len(matched_train_indices) > 0:
    # Compute P14's R2 on these matched rows
    y_matched = y[matched_train_indices]
    oof_matched = blended_oof[matched_train_indices]
    
    # Overall R2 on match rows
    ss_res = ((y_matched - oof_matched) ** 2).sum()
    ss_tot = ((y_matched - y_matched.mean()) ** 2).sum()
    r2_match = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    
    # Per-target R2 on match rows
    print(f"\nP14 blended OOF R2 on MATCH rows: {r2_match:.4f}")
    print(f"{'='*60}")
    
    per_target_results = []
    for t in TARGETS:
        ix = idx_all[t]
        # Find which of our matched indices belong to this target
        mask = tt[matched_train_indices] == t
        if mask.sum() == 0:
            continue
        yt = y_matched[mask]
        oof = oof_matched[mask]
        ss_r = ((yt - oof) ** 2).sum()
        ss_t = ((yt - yt.mean()) ** 2).sum()
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 1.0
        per_target_results.append({"target": t, "n": mask.sum(), "r2": r2, "mae": np.abs(yt - oof).mean()})
    
    pt_df = pd.DataFrame(per_target_results).set_index("target")
    print(pt_df.round(4).to_string())
    
    # Compute expected gain from label replacement
    print(f"\n{'='*60}")
    print(f"EXPECTED GAIN FROM LABEL REPLACEMENT")
    print(f"{'='*60}")
    
    total_gain = 0
    for t in TARGETS:
        ix = idx_all[t]
        mask = tt[matched_train_indices] == t
        if mask.sum() == 0:
            continue
        yt = y_matched[mask]
        oof = oof_matched[mask]
        # If we replace oof with yt (perfect prediction), residual goes to 0
        # Current residual:
        current_ss_res = ((yt - oof) ** 2).sum()
        # After replacement: residual = 0 for these rows
        # Gain = current_ss_res (we eliminate all error on these rows)
        # But we need to express as R2 gain on the FULL target
        # Full target SS_tot:
        ss_tot_full = ((y[ix] - y[ix].mean()) ** 2).sum()
        gain = current_ss_res / ss_tot_full  # R2 improvement on full target
        total_gain += gain
        print(f"  {t}: SS_res={current_ss_res:.2f}, SS_tot_full={ss_tot_full:.2f}, R2 gain={gain:.4f} ({mask.sum()} rows)")
    
    mean_gain = total_gain / len(TARGETS)
    print(f"\n  Estimated mean R2 gain: {mean_gain:.4f}")
    print(f"  P14 blend OOF: {0.8652:.4f}")
    print(f"  Expected after replacement: {0.8652 + mean_gain:.4f}")
    
    # Diagnostic gate
    if r2_match >= 0.995:
        print(f"\n*** DIAGNOSTIC: ABORT — P14 R2 on match rows = {r2_match:.4f} >= 0.995 ***")
        print("P14 already captures the leakage. No room for improvement.")
    elif mean_gain < 0.001:
        print(f"\n*** DIAGNOSTIC: ABORT — Expected R2 gain = {mean_gain:.4f} < 0.001 ***")
        print("Effect is too small to matter.")
    else:
        print(f"\n*** DIAGNOSTIC: PASS — Expected R2 gain = {mean_gain:.4f} >= 0.001 ***")
        print("Proceed with label replacement.")
else:
    print("No matched train indices found. Cannot compute residual.")
