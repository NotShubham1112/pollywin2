"""v18 gate validation (honest): Gate A OOF (GroupKFold 5) + Gate B (3 synthetic
private-slice 20% holdouts). For every split the sibling Ridge, phys coefs, and
the per-target gbm/mt blend are fit ONLY on the train fold; the val fold gets
fresh predictions. Binary hard rule: rows without a real sibling for the target
stay on P14 exactly.

Gate A to ship: OOF mean R2 >= 0.8740 and every target >= P14-0.002.
Gate B to ship: for all 3 holdouts, R2(v18) >= R2(v14) - 0.003.
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
SEED = 1230

A_SIB = {"eea": 0.025, "egb": 0.035, "egc": 0.0, "ei": 0.050, "eps": 0.080, "nc": 0.100, "tg": 0.0}
A_PHYS_EPS, A_PHYS_EGB = 0.20, 0.15
NC_LO, NC_HI = 1.55, 2.80
GATE_A_MEAN = 0.8740
GATE_A_TOL = 0.002
GATE_B_TOL = 0.003

NX = r"D:\Parth\ploywin r2\vault\kernel-v17-sib-phys\out\blend_oof_test.npz"
TR = r"D:\Parth\ploywin r2\official_dataset\train.csv"

z = np.load(NX, allow_pickle=True)
gbm = z["oof_gbm"].astype(np.float64)
mtm = z["oof_mt"].astype(np.float64)
tt = np.asarray(list(z["t_all"]))
yy = z["y_all"].astype(np.float64)
grp = z["g_all"]

tr = pd.read_csv(TR)
assert len(tr) == len(yy)
assert np.array_equal(tr["target_type"].values, tt)
smiles = tr["smiles"].values


def r2s(pred, ttype, yhs, idx):
    return {t: r2_score(yhs[idx][ttype[idx] == t], pred[idx][ttype[idx] == t])
            for t in TARGETS}


def build_pivot(rows):
    df = pd.DataFrame({
        "smiles": smiles[rows],
        "target": yy[rows],
        "target_type": tt[rows],
    }).dropna(subset=["target"])
    return df.pivot_table(index="smiles", columns="target_type",
                          values="target", aggfunc="first")


def sib_mat(piv, sl):
    out = np.full((len(sl), 7), np.nan)
    if len(piv) == 0:
        return out
    for i, s in enumerate(sl):
        if s in piv.index:
            row = piv.loc[s]
            for j, t in enumerate(TARGETS):
                if t in row.index and pd.notna(row[t]):
                    out[i, j] = row[t]
    return out


# Full-train pivot provides the sibling/phys FEATURES (factual co-target values
# of each molecule, exactly as the v17 kernel uses them). Only the *fits*
# (blend, sibling Ridge, phys coefs) are restricted to the train fold, so no
# val label ever enters a fit. GroupKFold on smiles then still shows the effect.


def sib_matrix_all_rows():
    piv = build_pivot(np.arange(len(yy)))
    return sib_mat(piv, smiles)


SIB_ALL = sib_matrix_all_rows()


def phys_coefs_df(sibm, rows):
    """Physical coefs (A*x^2+B for nc->eps, A*x+B for egc->egb) fit ONLY on the
    given train rows' lattice rows."""
    c = {}
    mm = np.isfinite(sibm[rows, GI["nc"]]) & np.isfinite(sibm[rows, GI["eps"]]) & (tt[rows] == "eps")
    if mm.sum() >= 5:
        x = sibm[rows[mm], GI["nc"]]
        y = sibm[rows[mm], GI["eps"]]
        c["eps"] = np.linalg.lstsq(np.column_stack([x**2, np.ones(len(x))]), y, rcond=None)[0]
    mm = np.isfinite(sibm[rows, GI["egc"]]) & np.isfinite(sibm[rows, GI["egb"]]) & (tt[rows] == "egb")
    if mm.sum() >= 5:
        x = sibm[rows[mm], GI["egc"]]
        y = sibm[rows[mm], GI["egb"]]
        c["egb"] = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    return c


def eval_split(train_rows, val_rows, full_sib):
    """Return (v14_pred, v18_pred) for val_rows. Features (sibling + phys coefs)
    come from the full sibling lattice; all model fits happen on train_rows only
    (Val target labels NEVER enter any fit)."""
    sib_tr = full_sib[np.sort(train_rows)]
    sib_va = full_sib[np.sort(val_rows)]

    # v14: per-target Ridge on gbm/mtm fit on train, predict val
    p14 = np.zeros(len(val_rows))
    for t in TARGETS:
        trt = np.where(tt[train_rows] == t)[0]
        vjt = np.where(tt[val_rows] == t)[0]
        if len(trt) == 0 or len(vjt) == 0:
            continue
        Mi = np.column_stack([gbm[train_rows][trt], mtm[train_rows][trt]])
        m = Ridge(alpha=1.0).fit(Mi, yy[train_rows][trt])
        Mb = np.column_stack([gbm[val_rows][vjt], mtm[val_rows][vjt]])
        p14[vjt] = m.predict(Mb)

    # phys coefs: fit on train_rows' sibling rows only
    pc = phys_coefs_df(full_sib, train_rows)
    # v18 = copy then sibling arms where real sibling exists
    v18 = p14.copy()
    va_tt = tt[val_rows]
    for t in TARGETS:
        a = A_SIB[t]
        if a <= 0:
            continue
        vt = np.where(va_tt == t)[0]
        if len(vt) == 0:
            continue
        keep = [k for k in range(7) if k != GI[t]]
        X = sib_va[vt][:, keep]
        has = np.isfinite(X).sum(1) >= 1
        if has.sum() == 0:
            continue
        trt = np.where(tt[train_rows] == t)[0]
        Xtr = sib_tr[trt][:, keep]
        htr = np.isfinite(Xtr).sum(1) >= 1
        if htr.sum() == 0:
            continue
        cm = np.nanmean(Xtr[htr], axis=0)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        Xtr_f = np.where(np.isfinite(Xtr), Xtr, cm)
        mdl = Ridge(alpha=1.0).fit(Xtr_f[htr], yy[train_rows][trt][htr])
        Xf = np.where(np.isfinite(X[has]), X[has], cm)
        sp = mdl.predict(Xf)
        v18[vt[has]] = (1 - a) * v18[vt[has]] + a * sp

    # phys legs
    vt = np.where(va_tt == "eps")[0]
    if len(vt) and "eps" in pc:
        nc = sib_va[vt, GI["nc"]]
        mk = np.isfinite(nc) & (nc >= NC_LO) & (nc <= NC_HI)
        if mk.sum():
            a2, b2 = pc["eps"]
            v18[vt[mk]] = (1 - A_PHYS_EPS) * v18[vt[mk]] + A_PHYS_EPS * (a2 * nc[mk]**2 + b2)
    vt = np.where(va_tt == "egb")[0]
    if len(vt) and "egb" in pc:
        ec = sib_va[vt, GI["egc"]]
        mk = np.isfinite(ec)
        if mk.sum():
            a2, b2 = pc["egb"]
            v18[vt[mk]] = (1 - A_PHYS_EGB) * v18[vt[mk]] + A_PHYS_EGB * (a2 * ec[mk] + b2)
    return p14, v18


# ---------- Gate A : 5-fold GroupKFold (group by smiles) ----------
folds = list(GroupKFold(n_splits=5).split(np.arange(len(yy)), yy, grp))
p14a = np.zeros(len(yy))
v18a = np.zeros(len(yy))
for tri, vai in folds:
    p, v = eval_split(np.sort(tri), np.sort(vai), SIB_ALL)
    p14a[vai] = p
    v18a[vai] = v

a14 = {t: r2_score(yy[tt == t], p14a[tt == t]) for t in TARGETS}
a18 = {t: r2_score(yy[tt == t], v18a[tt == t]) for t in TARGETS}
print("== Gate A (GroupKFold-5, honest) ==")
print("P14 :", " ".join(f"{t}={v:.4f}" for t, v in a14.items()),
      f"mean={np.mean(list(a14.values())):.4f}")
print("v18 :", " ".join(f"{t}={v:.4f}" for t, v in a18.items()),
      f"mean={np.mean(list(a18.values())):.4f}")
print("delt:", " ".join(f"{a18[t] - a14[t]:+.4f}" for t in TARGETS))
gateA_mean = np.mean(list(a18.values())) >= GATE_A_MEAN
gateA_tgt = all(a18[t] >= a14[t] - GATE_A_TOL for t in TARGETS)
print(f"Gate A mean>={GATE_A_MEAN}: {gateA_mean} | all>=-{GATE_A_TOL}: {gateA_tgt}")

# ---------- Gate B : three 20% holdouts ----------
def make_splits():
    n = len(tr)
    ic = np.arange(n)
    tr_r, va_r = train_test_split(ic, test_size=0.2, random_state=SEED, stratify=tt)
    yield "random-stratified", np.sort(tr_r), np.sort(va_r)

    piv_all = build_pivot(ic)
    sv = sib_mat(piv_all, smiles)
    cov = np.isfinite(sv).sum(1)
    order = np.argsort(cov, kind="stable")
    va = np.sort(order[:int(0.2 * n)])
    tr_n = np.setdiff1d(ic, va)
    yield "low-sib-cov", np.sort(tr_n), np.sort(va)

    rare = np.isin(tt, ["ei", "eps", "nc"])
    idx_common = np.where(~rare)[0]
    vax = idx_common[:int(0.2 * n)] if idx_common.size >= int(0.2 * n) else idx_common
    tr_x = np.setdiff1d(ic, vax)
    shuf = np.random.RandomState(SEED + 1).permutation(len(idx_common))
    vax = np.sort(idx_common[shuf[:int(0.2 * n)]])
    tr_x = np.setdiff1d(ic, vax)
    yield "low-rare-target", np.sort(tr_x), np.sort(vax)


print("\n== Gate B (3 holdouts) ==")
gateB_ok = True
for nm, tri, vai in make_splits():
    p, v = eval_split(np.sort(tri), np.sort(vai), SIB_ALL)
    r14 = r2_score(yy[vai], p)
    r18 = r2_score(yy[vai], v)
    ok = r18 >= r14 - GATE_B_TOL
    gateB_ok = gateB_ok and ok
    print(f"{nm:20s} n={len(vai):5d} P14={r14:.4f} v18={r18:.4f} "
          f"delta={r18-r14:+.4f} OK={ok}")

print("\n== FINAL ==")
ship = gateA_mean and gateA_tgt and gateB_ok
print("Gate A:", "PASS" if (gateA_mean and gateA_tgt) else "FAIL",
      "| Gate B:", "PASS" if gateB_ok else "FAIL",
      "|", "SHIP v18 to LB" if ship else "=> freeze P14 (LB 0.883)")