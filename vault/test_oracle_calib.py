"""KEY BREAKTHROUGH CANDIDATE: 
For test rows with cross-target twin in train (837/4940 = 17%), we have TRUE sibling values.
Apply a CONSERVATIVE physics imputation blend on these rows ONLY:
  pred_new = (1 - alpha) * pred_p14 + alpha * pred_phys(sib)
where alpha is small (0.1-0.3), and pred_phys is computed from the sibling value.

The key question: does this beat P14 on the multi-labeled TRAIN subset when applied
honestly with the same cross-target twin structure?
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
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}

# Build train pivot (raw SMILES match, just like test_cross_target_siblings.csv uses)
train_pivot = train.dropna(subset=["target"]).pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="first")
print(f"train_pivot shape: {train_pivot.shape} unique SMILES x {train_pivot.shape[1]} targets")

# Build sib matrix for train (each row -> its polymer's sibling values)
def get_sibs(df):
    sib = np.full((len(df), 7), np.nan)
    for i, s in enumerate(df["smiles"].values):
        if s in train_pivot.index:
            row = train_pivot.loc[s]
            for j, tt in enumerate(TARGETS):
                if tt in row.index and pd.notna(row[tt]):
                    sib[i, j] = row[tt]
    return sib

train_sib = get_sibs(train)
test_sib = get_sibs(test)
print(f"train_sib: {train_sib.shape}, has_sibs: {(~np.isnan(train_sib)).any(axis=1).sum()}/{len(train)}")
print(f"test_sib: {test_sib.shape}, has_sibs: {(~np.isnan(test_sib)).any(axis=1).sum()}/{len(test)}")

# Per-target, build physics arm
# egc = ei - eea; egb = a*egc + b; eps = a*nc^2 + b
phys = np.full_like(train_sib, np.nan)
mask = np.isfinite(train_sib[:,3]) & np.isfinite(train_sib[:,0])
phys[mask, 2] = train_sib[mask,3] - train_sib[mask,0]  # egc = ei - eea
mask = np.isfinite(train_sib[:,2]) & np.isfinite(train_sib[:,1])
A = np.linalg.lstsq(np.column_stack([train_sib[mask,2], np.ones(mask.sum())]), train_sib[mask,1], rcond=None)[0]
ok = np.isfinite(train_sib[:,2])
phys[ok, 1] = A[0]*train_sib[ok,2] + A[1]  # egb = a*egc + b
mask = np.isfinite(train_sib[:,5]) & np.isfinite(train_sib[:,4])
A = np.linalg.lstsq(np.column_stack([train_sib[mask,5]**2, np.ones(mask.sum())]), train_sib[mask,4], rcond=None)[0]
ok = np.isfinite(train_sib[:,5])
phys[ok, 4] = A[0]*train_sib[ok,5]**2 + A[1]  # eps = a*nc^2 + b

# Apply to test
phys_te = np.full_like(test_sib, np.nan)
mask = np.isfinite(test_sib[:,3]) & np.isfinite(test_sib[:,0])
phys_te[mask, 2] = test_sib[mask,3] - test_sib[mask,0]
ok = np.isfinite(test_sib[:,2])
phys_te[ok, 1] = A[0]*test_sib[ok,2] + A[1]
ok = np.isfinite(test_sib[:,5])
phys_te[ok, 4] = A[0]*test_sib[ok,5]**2 + A[1]

# Now: load P14 OOF and test preds
npz = np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)
oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
oof_mt  = np.asarray(npz["oof_mt"], dtype=float)
y       = np.asarray(npz["y_train"], dtype=float)
t_arr   = np.asarray(npz["target_type_train"])
test_gbm = np.asarray(npz["test_gbm"], dtype=float)
test_mt  = np.asarray(npz["test_mt"], dtype=float)
test_tt  = np.asarray(npz["target_type_test"])

# Per-target P14 OOF R^2 (baseline)
print("\n=== P14 OOF R^2 (per target) ===")
for tt in TARGETS:
    idx = np.where(t_arr == tt)[0]
    p14 = 0.5*oof_gbm[idx] + 0.5*oof_mt[idx]
    r2 = r2_score(y[idx], p14)
    print(f"  {tt}: R2={r2:.4f}  n={len(idx)}")

# Apply: for each target, on rows WITH sib, blend P14 + phys with small alpha
print("\n=== P14 + alpha*phys blend (rows with sib only) ===")
print("target | alpha | R2_P14_only | R2_blend (on sib rows) | delta | R2_blend (on all rows)")
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    keep_rows = np.where(np.isfinite(train_sib[idx]).any(axis=1))[0]  # rows with ANY sib
    sib_rows = idx[keep_rows]
    n_sib = len(sib_rows)
    if n_sib < 30:
        continue
    p14_sib = 0.5*oof_gbm[sib_rows] + 0.5*oof_mt[sib_rows]
    p14_all = 0.5*oof_gbm[idx] + 0.5*oof_mt[idx]
    r2_p14_sib = r2_score(y[sib_rows], p14_sib)
    r2_p14_all = r2_score(y[idx], p14_all)
    phys_v = phys[sib_rows, j]
    # Find rows where phys is computable
    has_phys = np.isfinite(phys_v)
    if has_phys.sum() < 30:
        print(f"  {tt}: only {has_phys.sum()} rows with computable phys; skip")
        continue
    for a in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        pred = (1-a)*p14_sib[has_phys] + a*phys_v[has_phys]
        r2 = r2_score(y[sib_rows][has_phys], pred)
        delta = r2 - r2_p14_sib
        print(f"  {tt:<4} | alpha={a:.2f} | sib_R2={r2_p14_sib:.4f} | phys_blend={r2:.4f} | delta_sib={delta:+.4f} | n_with_phys={has_phys.sum()}")
    print()
