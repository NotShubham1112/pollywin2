"""Build submission_v18.csv from the v17-sib-phys kernel's P14 test predictions.

Rule (hard coverage gate, binary):
   pred = P14  (exact base, the submitted v14 test preds)
   for each target with alpha_sib>0:
       row has >=1 finite OTHER-target sibling  ->  pred = (1-a)*P14 + a*sibRidge
       else                                    ->  pred stays P14  (HARD GATE)
   physics eps: if nc sibling finite and in [1.55,2.80]:
       pred = (1-0.20)*pred + 0.20*(A*nc^2+B)          (A,B fit on train pairs)
   physics egb: if egc sibling finite:
       pred = (1-0.15)*pred + 0.15*(A*egc+B)
Output = exact v14 rows for rows that touch nothing; identical to v14 submission
wherever no real sibling exists (this is what v17 lacked: it blended everywhere).
"""
import numpy as np
import pandas as pd

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
GI = {t: i for i, t in enumerate(TARGETS)}
NC_LO, NC_HI = 1.55, 2.80

# halved alphas (v18 spec)
A_SIB = {"eea": 0.025, "egb": 0.035, "egc": 0.0, "ei": 0.050,
         "eps": 0.080, "nc": 0.100, "tg": 0.0}
A_PHYS_EPS, A_PHYS_EGB = 0.20, 0.15

OUT = r"D:\Parth\ploywin r2\vault\kernel-v17-sib-phys\out"
TR = r"D:\Parth\ploywin r2\official_dataset\train.csv"
TE = r"D:\Parth\ploywin r2\official_dataset\test.csv"

tr = pd.read_csv(TR)
te = pd.read_csv(TE)
p14 = pd.read_csv(rf"{OUT}\submission_v14.csv").set_index("id")["target"]
assert (p14.index.values == te["id"].values).all()

# sibling lattice from train pivot (same - as v17 kernel)
pivot = tr.pivot_table(index="smiles", columns="target_type",
                       values="target", aggfunc="first")

def sibs(df):
    s = np.full((len(df), 7), np.nan)
    for i, sm in enumerate(df["smiles"].values):
        if sm in pivot.index:
            row = pivot.loc[sm]
            for j, tt in enumerate(TARGETS):
                if tt in row.index and pd.notna(row[tt]):
                    s[i, j] = row[tt]
    return s

sib_tr = sibs(tr)
sib_te = sibs(te)

# physics coefs from train pairs
m = np.isfinite(pivot["nc"].astype(float)) & np.isfinite(pivot["eps"].astype(float))
A_eps, B_eps = np.linalg.lstsq(
    np.column_stack([pivot.loc[m, "nc"].astype(float).values**2,
                     np.ones(m.sum())]),
    pivot.loc[m, "eps"].astype(float).values, rcond=None)[0]
m = np.isfinite(pivot["egc"].astype(float)) & np.isfinite(pivot["egb"].astype(float))
A_egb, B_egb = np.linalg.lstsq(
    np.column_stack([pivot.loc[m, "egc"].astype(float).values,
                     np.ones(m.sum())]),
    pivot.loc[m, "egb"].astype(float).values, rcond=None)[0]
print(f"phys eps = {A_eps:.4f}*nc^2 + {B_eps:.4f}   (n={m.sum()})")
print(f"phys egb = {A_egb:.4f}*egc + {B_egb:.4f}   (n={m.sum()})")

from sklearn.linear_model import Ridge
te_tt = te["target_type"].values
pred = p14.values.copy()
stats = {}

for t in TARGETS:
    a = A_SIB[t]
    if a <= 0:
        stats[t] = "alpha=0"
        continue
    j = GI[t]
    keep = [k for k in range(7) if k != j]
    te_keep = te_tt == t
    Xte = sib_te[te_keep][:, keep]
    has = np.isfinite(Xte).sum(1) >= 1
    if has.sum() == 0:
        stats[t] = f"alpha={a} no-sib-test"
        continue
    # fit sibling Ridge on ALL train rows of this target
    tri = np.where(tr["target_type"].values == t)[0]
    Xtr = sib_tr[tri][:, keep]
    cm = np.nanmean(Xtr, axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
    Xtr_f = np.where(np.isfinite(Xtr), Xtr, cm)
    lr = Ridge(alpha=1.0).fit(Xtr_f, tr["target"].values[tri].astype(np.float64))
    sibp = lr.predict(np.where(np.isfinite(Xte[has]), Xte[has], cm))
    sub = pred[te_keep]
    sub[has] = (1 - a) * sub[has] + a * sibp
    pred[te_keep] = sub
    stats[t] = f"alpha={a} applied={has.sum()}"

# physics eps leg
te_eps = te_tt == "eps"
nc = sib_te[te_eps, GI["nc"]]
mk = np.isfinite(nc) & (nc >= NC_LO) & (nc <= NC_HI)
if mk.sum():
    sub = pred[te_eps]
    sub[mk] = (1 - A_PHYS_EPS) * sub[mk] + A_PHYS_EPS * (A_eps * nc[mk]**2 + B_eps)
    pred[te_eps] = sub
    stats["phys_eps"] = f"a={A_PHYS_EPS} applied={mk.sum()}"

# physics egb leg
te_egb = te_tt == "egb"
ec = sib_te[te_egb, GI["egc"]]
mk = np.isfinite(ec)
if mk.sum():
    sub = pred[te_egb]
    sub[mk] = (1 - A_PHYS_EGB) * sub[mk] + A_PHYS_EGB * (A_egb * ec[mk] + B_egb)
    pred[te_egb] = sub
    stats["phys_egb"] = f"a={A_PHYS_EGB} applied={mk.sum()}"

print("applied:", stats)
print("max |v18 - v14| =", np.abs(pred - p14.values).max())

sub = pd.DataFrame({"id": te["id"].values, "target": pred})
sub.to_csv(rf"{OUT}\submission_v18.csv", index=False)
print("wrote", rf"{OUT}\submission_v18.csv", "rows", len(sub),
      "NaN", sub["target"].isna().sum())