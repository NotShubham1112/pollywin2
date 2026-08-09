"""CRUCIAL FINDING: For eps, physics imputation (eps = a*nc^2 + b) applied to sib rows
ADDS +0.05 to +0.09 R^2 over P14 alone.

For other targets, physics imputation is too sparse (only 0-82 rows have computable phys).

Test-time implication:
- eps has 153 test rows; how many have nc sibling? Let me check.
"""
import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np, pandas as pd
from rdkit import Chem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
SMALL5 = ["eea","egb","eps","nc","ei"]

train_pivot = train.dropna(subset=["target"]).pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="first")

# For TEST, how many test rows of each target_type have the REQUIRED sibling for physics?
# eps requires nc, egb requires egc, egc requires (ei AND eea)
print("=== Physics imputation availability on TEST ===")
for tt, recipe in [
    ("eps", lambda sib: np.isfinite(sib[:,5])),       # nc at index 5
    ("nc", lambda sib: np.isfinite(sib[:,4])),         # eps at index 4
    ("egb", lambda sib: np.isfinite(sib[:,2])),        # egc at index 2
    ("egc", lambda sib: np.isfinite(sib[:,3]) & np.isfinite(sib[:,0])),  # ei, eea
    ("tg", lambda sib: np.zeros(len(sib), dtype=bool)),
    ("eea", lambda sib: np.zeros(len(sib), dtype=bool)),
    ("ei", lambda sib: np.zeros(len(sib), dtype=bool)),
]:
    sub = test[test["target_type"]==tt]
    sib = np.full((len(sub), 7), np.nan)
    for i, s in enumerate(sub["smiles"].values):
        if s in train_pivot.index:
            row = train_pivot.loc[s]
            for j, t2 in enumerate(TARGETS):
                if t2 in row.index and pd.notna(row[t2]):
                    sib[i, j] = row[t2]
    mask = recipe(sib)
    print(f"  {tt}: {mask.sum()}/{len(sub)} = {100*mask.sum()/len(sub):.1f}% test rows have phys-imputable sibling")

# Now: per target, train sib Ridge with all available sibling features, predict on TEST
# Use only eps + nc since they have many computable physics recipes
print("\n=== eps with physics imputation on TEST ===")
# eps = a*nc^2 + b (fit on train pairs)
mask_tr = np.isfinite(train_pivot["nc"]) & np.isfinite(train_pivot["eps"])
nc_v = train_pivot.loc[mask_tr, "nc"].values
eps_v = train_pivot.loc[mask_tr, "eps"].values
A = np.linalg.lstsq(np.column_stack([nc_v**2, np.ones_like(nc_v)]), eps_v, rcond=None)[0]
print(f"  eps = {A[0]:.4f} * nc^2 + {A[1]:.4f}")

# Build test predictions for eps with the physics imputation
test_eps_idx = np.where(test["target_type"]=="eps")[0]
test_eps_sib = np.full((len(test_eps_idx), 7), np.nan)
for i, idx in enumerate(test_eps_idx):
    s = test.iloc[idx]["smiles"]
    if s in train_pivot.index:
        row = train_pivot.loc[s]
        for j, t2 in enumerate(TARGETS):
            if t2 in row.index and pd.notna(row[t2]):
                test_eps_sib[i, j] = row[t2]
# eps from physics
nc_idx = TARGETS.index("nc")
mask_nc = np.isfinite(test_eps_sib[:, nc_idx])
phys_pred = np.full(len(test_eps_idx), np.nan)
phys_pred[mask_nc] = A[0]*test_eps_sib[mask_nc, nc_idx]**2 + A[1]
print(f"  Test eps rows with nc sibling: {mask_nc.sum()}/{len(test_eps_idx)}")

# Compare phys_pred to P14 prediction
test_gbm = np.asarray(np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)["test_gbm"], dtype=float)
test_mt  = np.asarray(np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)["test_mt"], dtype=float)
test_tt  = np.asarray(np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)["target_type_test"])
test_te_eps_gbm = test_gbm[test_tt=="eps"]
test_te_eps_mt  = test_mt[test_tt=="eps"]
p14_eps = 0.5*test_te_eps_gbm + 0.5*test_te_eps_mt

# Apply conservative blend on TEST
print("\n=== Conservative blend on TEST (eps) ===")
for a in [0.0, 0.10, 0.20, 0.30]:
    pred = np.where(np.isnan(phys_pred), p14_eps, (1-a)*p14_eps + a*phys_pred)
    print(f"  alpha={a:.2f}: test_pred mean={pred.mean():.3f} std={pred.std():.3f}")

# Save blended test prediction for eps
test_blended = p14_eps.copy()
mask_valid = np.isfinite(phys_pred)
test_blended[mask_valid] = (1-0.30)*p14_eps[mask_valid] + 0.30*phys_pred[mask_valid]
print(f"  With alpha=0.30: {mask_valid.sum()} test eps rows modified.")

# Now do the same for nc (eps -> nc via Maxwell relation? no direct recipe)
# Actually no direct physics for nc from eps alone. Skip.

# Save for further analysis
np.savez(os.path.join(WORK, "vault", "phys_eps_test.npz"),
         phys_pred=phys_pred, p14_pred=p14_eps,
         test_idx=test_eps_idx)
print("Saved phys_eps_test.npz")
