"""Analyze calibration: the simplest, lowest-risk improvement to a Kaggle submission.

For each target, learn a calibration on OOF predictions:
  y_calibrated = a * y_pred + b
Then apply on test. This is what NeurIPS 1st place did (Tg += 0.5644 * std(Tg)).

Our P14 has per-target Ridge over [GBM, MT-GNN]. The Ridge itself is a calibration.
But there's still residual bias we can mop up.
"""
import os, warnings, sys
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
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None
train["canon"] = train["smiles"].apply(canon)
test["canon"]  = test["smiles"].apply(canon)
train = train.dropna(subset=["canon"]).reset_index(drop=True)
test  = test.dropna(subset=["canon"]).reset_index(drop=True)

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
piv = train.dropna(subset=["target"]).pivot_table(index="canon", columns="target_type", values="target", aggfunc="first")

npz = np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)
oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
oof_mt  = np.asarray(npz["oof_mt"], dtype=float)
y       = np.asarray(npz["y_train"], dtype=float)
t_arr   = np.asarray(npz["target_type_train"])
test_gbm = np.asarray(npz["test_gbm"], dtype=float)
test_mt  = np.asarray(npz["test_mt"], dtype=float)
test_tt  = np.asarray(npz["target_type_test"])

canon_arr = train["canon"].values
sib = np.full((len(train), 7), np.nan)
for j, tt in enumerate(TARGETS):
    sib[:, j] = piv[tt].reindex(canon_arr).values
y_arr = train["target"].values.astype(float)
groups = canon_arr

# Per-target: fold-safe Ridge over [GBM, MT] (this is the P14 baseline)
print("=== P14 baseline (per-target Ridge over GBM, MT) ===")
print("target | R2_P14 | best_alpha")
P14_PRED = np.zeros(len(train))
ALPHA_P14 = {}
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    M = np.column_stack([oof_gbm[idx], oof_mt[idx]])
    yt = y_arr[idx]; g = groups[idx]
    cv = list(GroupKFold(n_splits=5).split(M, yt, g))
    best, besta = -np.inf, 1.0
    for a in [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]:
        o = np.zeros(len(idx))
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], yt[tr]).predict(M[vk])
        r = r2_score(yt, o)
        if r > best: best, besta = r, a
    oof = np.zeros(len(idx))
    for tr, vk in cv:
        oof[vk] = Ridge(alpha=besta).fit(M[tr], yt[tr]).predict(M[vk])
    P14_PRED[idx] = oof
    ALPHA_P14[tt] = besta
    print(f"  {tt:<4} | {best:.4f}  | {besta}")

# Compute residual bias per target on OOF
print("\n=== Residual bias per target (mean(y_true - y_pred)) ===")
for tt in TARGETS:
    idx = np.where(t_arr == tt)[0]
    bias = float(np.mean(y_arr[idx] - P14_PRED[idx]))
    print(f"  {tt}: mean residual = {bias:+.4f} (train sigma = {y_arr[idx].std():.3f})")

# Add an additive calibration shift on OOF, see if it improves
print("\n=== Per-target additive calibration shift (fold-safe, learned on OTHER folds) ===")
print("target | R2_P14 | R2_P14+shift | best_shift | std_test")
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    yt = y_arr[idx]
    p14 = P14_PRED[idx]
    r2_p14 = r2_score(yt, p14)
    # Tune shift on OOF (proper nested CV)
    cv = list(GroupKFold(n_splits=5).split(p14, yt, groups[idx]))
    final = np.zeros(len(idx))
    chosen = []
    for tr, vk in cv:
        bias_tr = float(np.mean(yt[tr] - p14[tr]))
        chosen.append(bias_tr)
        final[vk] = p14[vk] + bias_tr
    r2_c = r2_score(yt, final)
    print(f"  {tt:<4} | {r2_p14:.4f}  | {r2_c:.4f}  | {np.mean(chosen):+.4f} | std test = {test_gbm[test_tt==tt].std():.3f}")

# Also: multiplicative shift
print("\n=== Per-target multiplicative calibration (a * pred + b) learned on OOF ===")
print("target | R2_P14 | R2_calibrated | a | b")
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    yt = y_arr[idx]
    p14 = P14_PRED[idx]
    r2_p14 = r2_score(yt, p14)
    # Fold-safe linear regression
    cv = list(GroupKFold(n_splits=5).split(p14, yt, groups[idx]))
    final = np.zeros(len(idx))
    coefs = []
    for tr, vk in cv:
        A = np.vstack([p14[tr], np.ones(len(tr))]).T
        coef = np.linalg.lstsq(A, yt[tr], rcond=None)[0]
        final[vk] = coef[0]*p14[vk] + coef[1]
        coefs.append(coef)
    r2_c = r2_score(yt, final)
    a, b = np.mean(coefs, axis=0)
    print(f"  {tt:<4} | {r2_p14:.4f}  | {r2_c:.4f}  | {a:.4f} | {b:+.4f}")
