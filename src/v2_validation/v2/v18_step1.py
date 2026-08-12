"""v18 Gate A (OOF) + Gate B (3 synthetic private-slice stratifications).

Decision rule (binary hard gate):
    out[row] = P14[row]
    if has_sib[t, row]:                       # >=1 finite other-target sibling
        out[row] = (1-a_sib[t])*P14[row] + a_sib[t]*sibRidge[row]
    if t=='eps' and has_nc and 1.55<=nc<=2.80:
        out[row] = (1-0.20)*out[row] + 0.20*(a*nc^2+b)
    if t=='egb' and has_egc:
        out[row] = (1-0.15)*out[row] + 0.15*(a*egc+b)
  Rows without sibling => P14 exactly.

Gates:
  A: each-target R2 >= P14 - 0.002 ; mean >= 0.8740
  B: 3 holdouts (random / by_target / by_sibcov), each v18 >= v14 - 0.003
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
GI = {t: i for i, t in enumerate(TARGETS)}
GLOBAL_FOLDS = 5
SEED = 42

ALPHA_SIB = {"eea": 0.025, "egb": 0.035, "egc": 0.0, "ei": 0.050, "eps": 0.080, "nc": 0.100, "tg": 0.0}
A_PHYS_EPS = 0.20
A_PHYS_EGB = 0.15
NC_LO, NC_HI = 1.55, 2.80
GATE_A_TOL = 0.002
GATE_A_MEAN = 0.8740
GATE_B_TOL = 0.003

NZX = "D:/Parth/ploywin r2/vault/kernel-v17-sib-phys/out/blend_oof_test.npz"
TR = "D:/Parth/ploywin r2/official_dataset/train.csv"
TE = "D:/Parth/ploywin r2/official_dataset/test.csv"

z = np.load(NZX, allow_pickle=True)
gbm, mt = z["oof_gbm"].astype(np.float64), z["oof_mt"].astype(np.float64)
tgbm, tmt = z["test_gbm"].astype(np.float64), z["test_mt"].astype(np.float64)
tt, yy, gg = z["t_all"], z["y_all"].astype(np.float64), z["g_all"]

tr = pd.read_csv(TR)
test = pd.read_csv(TE)

# ------------------------------------------------------------------ helpers
def blend_oof(g, m, ttype, y, grp, folds=GLOBAL_FOLDS):
    out = np.zeros(len(y))
    for t in TARGETS:
        i = np.where(ttype == t)[0]
        M = np.column_stack([g[i], m[i]])
        yv = y[i]
        cv = list(GroupKFold(n_splits=folds).split(M, yv, grp[i]))
        for tri, vk in cv:
            out[i[vk]] = Ridge(alpha=1.0).fit(M[tri], yv[tri]).predict(M[vk])
    return out


def r2s(pred, ttype, y):
    return {t: r2_score(y[ttype == t], pred[ttype == t]) for t in TARGETS}


def build_piv(df, cols=("smiles", "target_type", "target")):
    return df[list(cols)].dropna(subset=["target"]).pivot_table(
        index="smiles", columns="target_type", values="target", aggfunc="first")


def sib_mat(piv, smiles):
    out = np.full((len(smiles), 7), np.nan)
    for i, s in enumerate(smiles):
        if s in piv.index:
            row = piv.loc[s]
            for j, t in enumerate(TARGETS):
                if t in row.index and pd.notna(row[t]):
                    out[i, j] = row[t]
    return out


def phys_coeffs(piv):
    co = {}
    m = np.isfinite(piv["nc"]) & np.isfinite(piv["eps"])
    if m.sum() >= 5:
        A = np.column_stack([piv.loc[m, "nc"].astype(float) ** 2, np.ones(m.sum())])
        co["eps"] = np.linalg.lstsq(A, piv.loc[m, "eps"].astype(float), rcond=None)[0]
    m = np.isfinite(piv["egc"]) & np.isfinite(piv["egb"])
    if m.sum() >= 5:
        A = np.column_stack([piv.loc[m, "egc"].astype(float), np.ones(m.sum())])
        co["egb"] = np.linalg.lstsq(A, piv.loc[m, "egb"].astype(float), rcond=None)[0]
    return co


def v18_eval(sib, ttype, p, y, piv, pc):
    """Apply binary-gate rule. sib per-row (n,7). p = P14 preds. Returns candidate."""

    out = p.astype(np.float64).copy()
    # sib-Ridge legs
    for t in TARGETS:
        a = A_SIB[t]
        if a <= 0:
            continue
        i = np.where(ttype == t)[0]
        j = GI[t]
        keep = [k for k in range(7) if k != j]
        X = sib[i][:, keep]
        has = np.isfinite(X).sum(1) >= 1
        if has.sum() == 0:
            continue
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        lr = Ridge(alpha=1.0).fit(np.where(np.isfinite(X), X, cm), y[i])
        sibp = lr.predict(np.where(np.isfinite(X), X, cm))
        out[i[org]] = out[i[has]]

    # phys legs
    if "eps" in pc:
        i = np.where(ttype == "eps")[0]
        nc = sib[i, GI["nc"]]
        m = np.isfinite(nc) & (nc >= NC_LO) & (nc <= NC_HI)
        if m.sum():
            aa, bb = pc["eps"]
            phys = aa * nc[m] ** 2 + bb
            out[i[m]] = (1 - A_PHYS_EPS) * out[i[m]] + A_PHYS_EPS * phys
    if "egb" in pc:
        i = np.where(ttype == "egb")[0]
        egc = sib[i, GI["egc"]]
        m = np.isfinite(egc)
        if m.sum():
            aa, bb = pc["egb"]
            phys = aa * egc[m] + bb
            out[i[m]] = (1 - A_PHYS_EGB) * out[i[m]] + A_PHYS_EGB * phys
    return out


print("defs ok")