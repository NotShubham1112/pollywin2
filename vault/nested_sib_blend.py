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

sib_te = np.full((len(test), 7), np.nan)
for j, tt in enumerate(TARGETS):
    sib_te[:, j] = piv[tt].reindex(test["canon"].values).values

# Nested-CV per-target conservative sib blend
ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
print("=== Nested-CV per-target conservative sib blend ===")
results = []
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    keep = [k for k in range(7) if k != j]
    Xsib = sib[idx][:, keep]
    yt = y_arr[idx]; g = groups[idx]
    p14 = 0.5 * oof_gbm[idx] + 0.5 * oof_mt[idx]
    # Pre-compute sib Ridge OOF on the SAME folds
    cv = list(GroupKFold(n_splits=5).split(Xsib, yt, g))
    o_sib = np.zeros(len(idx))
    for tr, vk in cv:
        Xf = Xsib[tr].copy()
        cm = np.nanmean(Xf, axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        Xf = np.where(np.isfinite(Xf), Xf, cm)
        Xv = np.where(np.isfinite(Xsib[vk].copy()), Xsib[vk].copy(), cm)
        o_sib[vk] = Ridge(alpha=1.0).fit(Xf, yt[tr]).predict(Xv)
    # Nested: for each held-out fold, tune alpha on OTHER folds
    final = np.zeros(len(idx))
    chosen = []
    for outer_tr, outer_vk in cv:
        best_a, best_r2 = 0.0, -np.inf
        for a in ALPHAS:
            sib_f = np.where(np.isfinite(o_sib[outer_tr]), o_sib[outer_tr], yt[outer_tr].mean())
            r = r2_score(yt[outer_tr], (1 - a) * p14[outer_tr] + a * sib_f)
            if r > best_r2: best_r2, best_a = r, a
        sib_f_vk = np.where(np.isfinite(o_sib[outer_vk]), o_sib[outer_vk], yt[outer_vk].mean())
        final[outer_vk] = (1 - best_a) * p14[outer_vk] + best_a * sib_f_vk
        chosen.append(best_a)
    r2_p = r2_score(yt, p14); r2_c = r2_score(yt, final)
    results.append((tt, r2_p, r2_c, chosen))
    print(f"  {tt:<4} | P14={r2_p:.4f} | nested={r2_c:.4f} delta={r2_c-r2_p:+.4f} | alphas={[round(a,3) for a in chosen]}")

arr = np.array([(r[1], r[2]) for r in results])
print(f"\nEqual-weight per target:")
print(f"  P14:               {arr[:,0].mean():.4f}")
print(f"  Conservative sib:  {arr[:,1].mean():.4f}")
print(f"  Delta:             {(arr[:,1]-arr[:,0]).mean():+.4f}")

# Generate test predictions for the conservative blend
print("\n=== Generating test predictions under nested-CV-tuned conservative sib blend ===")
test_pred = np.zeros(len(test))
for j, tt in enumerate(TARGETS):
    idx = np.where(t_arr == tt)[0]
    keep = [k for k in range(7) if k != j]
    Xsib = sib[idx][:, keep]
    yt = y_arr[idx]; g = groups[idx]
    p14 = 0.5 * oof_gbm[idx] + 0.5 * oof_mt[idx]
    cv = list(GroupKFold(n_splits=5).split(Xsib, yt, g))
    o_sib = np.zeros(len(idx))
    for tr, vk in cv:
        Xf = Xsib[tr].copy()
        cm = np.nanmean(Xf, axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        Xf = np.where(np.isfinite(Xf), Xf, cm)
        Xv = np.where(np.isfinite(Xsib[vk].copy()), Xsib[vk].copy(), cm)
        o_sib[vk] = Ridge(alpha=1.0).fit(Xf, yt[tr]).predict(Xv)
    # Choose alpha = mean of per-fold chosen alphas
    chosen_alpha = []
    for outer_tr, outer_vk in cv:
        best_a, best_r2 = 0.0, -np.inf
        for a in ALPHAS:
            sib_f = np.where(np.isfinite(o_sib[outer_tr]), o_sib[outer_tr], yt[outer_tr].mean())
            r = r2_score(yt[outer_tr], (1 - a) * p14[outer_tr] + a * sib_f)
            if r > best_r2: best_r2, best_a = r, a
        chosen_alpha.append(best_a)
    alpha_final = float(np.mean(chosen_alpha))
    # Train final sib Ridge on FULL data
    cm = np.nanmean(Xsib, axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
    Xtr_imp = np.where(np.isfinite(Xsib), Xsib, cm)
    lr = Ridge(alpha=1.0).fit(Xtr_imp, yt)
    idx_te = np.where(test_tt == tt)[0]
    Xte = sib_te[idx_te][:, keep]
    Xte_imp = np.where(np.isfinite(Xte), Xte, cm)
    sib_te_pred = lr.predict(Xte_imp)
    p14_te = 0.5 * test_gbm[idx_te] + 0.5 * test_mt[idx_te]
    pred = (1 - alpha_final) * p14_te + alpha_final * sib_te_pred
    test_pred[idx_te] = pred
    print(f"  {tt}: alpha={alpha_final:.3f}, test_pred mean={pred.mean():.3f} std={pred.std():.3f}")

sub = pd.DataFrame({"id": test["id"].values, "target": test_pred})
sub.to_csv(os.path.join(WORK, "vault", "submission_conservative_sib_nested.csv"), index=False)
print("\nWrote submission_conservative_sib_nested.csv")
