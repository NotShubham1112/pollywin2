"""Generate the FINAL submission with all improvements:
1. P14 baseline (per-target Ridge over GBM, MT).
2. Conservative sib Ridge blend (alpha tuned per target).
3. Conservative physics imputation on eps (alpha=0.50 on sib-covered rows).

Output: submission_v17_final.csv
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

# Build P14 baseline per target
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
    for tr, vk in cv:
        lr = Ridge(alpha=besta).fit(M[tr], y[tr])
        oof[vk] = lr.predict(M[vk])
    return oof, besta

p14_oof = np.zeros(len(train))
test_pred_p14 = np.zeros(len(test))
best_a_p14 = {}
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    M = np.column_stack([oof_gbm[idx], oof_mt[idx]])
    yt = y[idx]; g = train.iloc[idx]["smiles"].values
    oof, ba = fold_safe_blend(M, yt, g)
    p14_oof[idx] = oof
    best_a_p14[tt] = ba
    # Train on full data, predict test
    lr = Ridge(alpha=ba).fit(M, yt)
    idx_te = np.where(test_tt == tt)[0]
    M_te = np.column_stack([test_gbm[idx_te], test_mt[idx_te]])
    test_pred_p14[idx_te] = lr.predict(M_te)

# 1) Conservative sib Ridge blend (alpha tuned per target)
ALPHA_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
oof_sib = np.zeros(len(train))
test_pred_sib = np.zeros(len(test))
chosen_alpha_sib = {}
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
    chosen = []
    for outer_tr, outer_vk in cv:
        best_a, best_r2 = 0.0, -np.inf
        for a in ALPHA_GRID:
            sib_f = np.where(np.isfinite(o_sib[outer_tr]), o_sib[outer_tr], yt[outer_tr].mean())
            r = r2_score(yt[outer_tr], (1-a)*p14_oof[idx][outer_tr] + a*sib_f)
            if r > best_r2: best_r2, best_a = r, a
        chosen.append(best_a)
    alpha_final = float(np.mean(chosen))
    chosen_alpha_sib[tt] = alpha_final
    # Refit Ridge on FULL train, predict test
    cm = np.nanmean(Xsib, axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
    Xtr_imp = np.where(np.isfinite(Xsib), Xsib, cm)
    lr = Ridge(alpha=1.0).fit(Xtr_imp, yt)
    idx_te = np.where(test_tt == tt)[0]
    Xte = test_sib[idx_te][:, keep]
    Xte_imp = np.where(np.isfinite(Xte), Xte, cm)
    sib_pred_te = lr.predict(Xte_imp)
    oof_sib[idx] = o_sib
    test_pred_sib[idx_te] = sib_pred_te
    print(f"  {tt}: alpha_sib={alpha_final:.3f}, test_pred mean={sib_pred_te.mean():.3f}")

# 2) Physics imputation on eps
mask_tr = np.isfinite(train_pivot["nc"]) & np.isfinite(train_pivot["eps"])
nc_v = train_pivot.loc[mask_tr, "nc"].values
eps_v = train_pivot.loc[mask_tr, "eps"].values
A_phys = np.linalg.lstsq(np.column_stack([nc_v**2, np.ones_like(nc_v)]), eps_v, rcond=None)[0]
print(f"\n  eps = {A_phys[0]:.4f} * nc^2 + {A_phys[1]:.4f}")

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
    if r2 > best_r2: best_r2, best_a = r2, a
print(f"  eps: alpha_phys={best_a:.3f}, R2={best_r2:.4f}")

# Apply phys to test eps
test_eps_idx = np.where(test_tt == "eps")[0]
phys_te_eps = np.full(len(test_eps_idx), np.nan)
mask = np.isfinite(test_sib[test_eps_idx, nc_idx])
phys_te_eps[mask] = A_phys[0]*test_sib[test_eps_idx, nc_idx][mask]**2 + A_phys[1]
print(f"  test eps rows with phys: {mask.sum()}/{len(test_eps_idx)}")

# Generate final predictions
final_test = test_pred_p14.copy()
# Apply sib Ridge blend per target
for j, tt in enumerate(TARGETS):
    idx_te = np.where(test_tt == tt)[0]
    a = chosen_alpha_sib[tt]
    final_test[idx_te] = (1-a)*test_pred_p14[idx_te] + a*test_pred_sib[idx_te]
# Apply phys on eps
final_test[test_eps_idx] = np.where(
    np.isfinite(phys_te_eps),
    (1-best_a)*final_test[test_eps_idx] + best_a*phys_te_eps,
    final_test[test_eps_idx]
)

# Sanity check: per-target mean shift
print("\n=== Per-target prediction shift (P14 -> FINAL) ===")
for tt in TARGETS:
    idx_te = np.where(test_tt == tt)[0]
    print(f"  {tt}: P14 mean={test_pred_p14[idx_te].mean():.3f} std={test_pred_p14[idx_te].std():.3f} | FINAL mean={final_test[idx_te].mean():.3f} std={final_test[idx_te].std():.3f}")

# Save submission
sub = pd.DataFrame({"id": test["id"].values, "target": final_test})
sub.to_csv(os.path.join(WORK, "vault", "submission_v17_final.csv"), index=False)
print(f"\nWrote submission_v17_final.csv (n={len(sub)})")

# Also compute final OOF R^2 for verification
print("\n=== Final OOF R^2 verification ===")
final_oof = p14_oof.copy()
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    a = chosen_alpha_sib[tt]
    final_oof[idx] = (1-a)*p14_oof[idx] + a*oof_sib[idx]
# Apply eps phys
final_eps_blend = np.where(np.isfinite(phys_eps), (1-best_a)*final_oof[train_eps_idx] + best_a*phys_eps, final_oof[train_eps_idx])
final_oof[train_eps_idx] = final_eps_blend

arr_final = []
for tt in TARGETS:
    idx = np.where(t_arr == tt)[0]
    r2 = r2_score(y[idx], final_oof[idx])
    arr_final.append((tt, r2, len(idx)))
    print(f"  {tt}: R2={r2:.4f}")

print(f"\nFINAL mean R^2 (equal weight per target): {np.mean([a[1] for a in arr_final]):.4f}")
print(f"P14 mean R^2 (equal weight per target):  0.8642")
print(f"Delta: {np.mean([a[1] for a in arr_final]) - 0.8642:+.4f}")
