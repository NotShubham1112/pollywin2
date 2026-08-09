"""FINAL SYNTHESIS of findings.

State of P14 (LB 0.883):
- Per-target Ridge blend of GBM trio + MT-GNN (PI1M pretrained).
- Honest OOF mean R^2 = 0.877.
- Per-target OOF: tg 0.901, egc 0.907, egb 0.926, eps 0.756, nc 0.839, ei 0.811, eea 0.907.

What we tried in this analysis:
1. Sib Ridge over all 7 targets as 3rd arm: +0.002 nested-CV mean R^2 (ebg +0.004, ei +0.010, nc +0.003; tg/egc/eea flat or negative).
2. Conservative physics blend (eps = a*nc^2 + b): on eps, +0.05 to +0.09 R^2 on sib rows in TRAIN.
3. Distribution-shift calibration (NeurIPS 1st place Tg += 0.5644 * std): no OOF gain (residuals already ~0).
4. Exact twin replacement (test rows with TRUE train labels): only 2 test rows match; negligible.
5. Backbone twin (strip [*]): 837/4940 (17%) test rows have cross-target twin in train.
6. Physics imputation recipe coverage: eps/nc have 62% test coverage, egb 28%, others <1%.

KEY BREAKTHROUGH: Conservative physics imputation on eps (and possibly nc) rows with sib.
- eps OOF R^2: 0.767 baseline -> 0.857 with phys blend (alpha=0.30) on sib rows
- 95/153 = 62% of test eps rows benefit.
- eps is one of the two weakest targets. Mean R^2 gain per target = +0.06 if transferred.
- Equal-weight per target: (0.06 * 153 / 4940) = 0.0019 mean R^2 contribution to overall metric.

Combined with sib Ridge for egb (+0.004), ei (+0.010), nc (+0.003) on their full train sets
(but test-time transfer depends on whether the Ridge weights generalize):
Total potential gain: +0.005 to +0.010 mean R^2.

This would move LB from 0.883 to 0.890-0.895 if it transfers.

Implementation plan:
- Generate P14 baseline test predictions (already have).
- Apply conservative sib Ridge blend per target (alpha from nested CV).
- Apply conservative physics imputation blend on eps (and ideally nc) for sib-covered rows.
- Combine and submit.

Risk: same as v16, but with smaller alpha and only on rows where the auxiliary arm is computed from real data.

Let me build the FINAL submission and measure OOF.
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

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
SMALL5 = ["eea","egb","eps","nc","ei"]

train_pivot = train.dropna(subset=["target"]).pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="first")

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

npz = np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)
oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
oof_mt  = np.asarray(npz["oof_mt"], dtype=float)
y       = np.asarray(npz["y_train"], dtype=float)
t_arr   = np.asarray(npz["target_type_train"])
test_gbm = np.asarray(npz["test_gbm"], dtype=float)
test_mt  = np.asarray(npz["test_mt"], dtype=float)
test_tt  = np.asarray(npz["target_type_test"])

# Build P14 baseline (per-target Ridge over GBM, MT)
print("=== Building P14 baseline (per-target Ridge) ===")
ALPHAS = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]

def fold_safe_blend(M, y, g, alphas=ALPHAS, n_splits=5):
    M = np.where(np.isfinite(M), M, np.nanmean(M, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, besta = -np.inf, alphas[0]
    for a in alphas:
        o = np.zeros(len(y))
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best: best, besta = r, a
    oof = np.zeros(len(y))
    coefs = []
    for tr, vk in cv:
        lr = Ridge(alpha=besta).fit(M[tr], y[tr])
        oof[vk] = lr.predict(M[vk])
        coefs.append(lr.coef_)
    return oof, besta, np.mean(coefs, axis=0)

p14_oof = np.zeros(len(train))
best_a_p14 = {}
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    M = np.column_stack([oof_gbm[idx], oof_mt[idx]])
    yt = y[idx]; g = train.iloc[idx]["smiles"].values  # group by SMILES
    oof, ba, _ = fold_safe_blend(M, yt, g)
    p14_oof[idx] = oof
    best_a_p14[tt] = ba

# P14 baseline R^2 per target
print("P14 baseline R^2 per target:")
arr_p14 = []
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    r2 = r2_score(y[idx], p14_oof[idx])
    arr_p14.append((tt, r2, len(idx)))
    print(f"  {tt}: R2={r2:.4f}  n={len(idx)}  alpha={best_a_p14[tt]}")

# Now build the FINAL blend: P14 + conservative sib + conservative phys on eps
print("\n=== FINAL blend: P14 + sib Ridge (alpha tuned) + phys on eps ===")
final_oof = p14_oof.copy()
chosen_alpha_sib = {}
chosen_alpha_phys = {}

# 1) Sib Ridge per target, conservative alpha
ALPHA_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    keep = [k for k in range(7) if k != j]
    Xsib = train_sib[idx][:, keep]
    yt = y[idx]; g = train.iloc[idx]["smiles"].values
    cv = list(GroupKFold(n_splits=5).split(Xsib, yt, g))
    o_sib = np.zeros(len(idx))
    for tr, vk in cv:
        Xf = Xsib[tr].copy()
        cm = np.nanmean(Xf, axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        Xf = np.where(np.isfinite(Xf), Xf, cm)
        Xv = np.where(np.isfinite(Xsib[vk].copy()), Xsib[vk].copy(), cm)
        o_sib[vk] = Ridge(alpha=1.0).fit(Xf, yt[tr]).predict(Xv)
    # nested CV to choose alpha
    final = np.zeros(len(idx))
    chosen = []
    for outer_tr, outer_vk in cv:
        best_a, best_r2 = 0.0, -np.inf
        for a in ALPHA_GRID:
            sib_f = np.where(np.isfinite(o_sib[outer_tr]), o_sib[outer_tr], yt[outer_tr].mean())
            r = r2_score(yt[outer_tr], (1-a)*p14_oof[idx][outer_tr] + a*sib_f)
            if r > best_r2: best_r2, best_a = r, a
        sib_f_vk = np.where(np.isfinite(o_sib[outer_vk]), o_sib[outer_vk], yt[outer_vk].mean())
        final[outer_vk] = (1-best_a)*p14_oof[idx][outer_vk] + best_a*sib_f_vk
        chosen.append(best_a)
    alpha_final = float(np.mean(chosen))
    chosen_alpha_sib[tt] = alpha_final
    final_oof[idx] = final
    print(f"  {tt}: alpha_sib={alpha_final:.3f}, R2={r2_score(y[idx], final):.4f}")

# 2) Physics imputation on eps: blend (1-a)*P14 + a*phys where phys is computable
print("\n=== Physics on eps (a*nc^2 + b fit on train pairs) ===")
mask_tr = np.isfinite(train_pivot["nc"]) & np.isfinite(train_pivot["eps"])
nc_v = train_pivot.loc[mask_tr, "nc"].values
eps_v = train_pivot.loc[mask_tr, "eps"].values
A_phys = np.linalg.lstsq(np.column_stack([nc_v**2, np.ones_like(nc_v)]), eps_v, rcond=None)[0]

# Compute phys for eps rows in train (using sib)
j_eps = TARGETS.index("eps")
nc_idx = TARGETS.index("nc")
train_eps_idx = np.where(t_arr == "eps")[0]
phys_eps = np.full(len(train_eps_idx), np.nan)
mask = np.isfinite(train_sib[train_eps_idx, nc_idx])
phys_eps[mask] = A_phys[0]*train_sib[train_eps_idx, nc_idx][mask]**2 + A_phys[1]

# Tune alpha_phys for eps
ALPHAS_PHYS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
best_a, best_r2 = 0.0, -np.inf
for a in ALPHAS_PHYS:
    p14_eps = p14_oof[train_eps_idx]
    blend = np.where(np.isfinite(phys_eps), (1-a)*p14_eps + a*phys_eps, p14_eps)
    r2 = r2_score(y[train_eps_idx], blend)
    if r2 > best_r2:
        best_r2, best_a = r2, a
chosen_alpha_phys["eps"] = best_a
print(f"  eps: alpha_phys={best_a:.3f}, R2={best_r2:.4f}")
# Apply
final_eps_blend = np.where(np.isfinite(phys_eps), (1-best_a)*p14_oof[train_eps_idx] + best_a*phys_eps, p14_oof[train_eps_idx])
final_oof[train_eps_idx] = final_eps_blend

# Final per-target R^2
print("\n=== FINAL blend per-target R^2 ===")
arr_final = []
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    r2 = r2_score(y[idx], final_oof[idx])
    arr_final.append((tt, r2, len(idx)))
    print(f"  {tt}: R2={r2:.4f}  delta_vs_P14={r2-arr_p14[j][1]:+.4f}")

# Equal-weight mean R^2
print(f"\nP14 mean R^2 (equal weight): {np.mean([a[1] for a in arr_p14]):.4f}")
print(f"FINAL mean R^2 (equal weight): {np.mean([a[1] for a in arr_final]):.4f}")
print(f"Delta: {np.mean([a[1] for a in arr_final]) - np.mean([a[1] for a in arr_p14]):+.4f}")
