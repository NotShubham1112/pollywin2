"""Evaluate Ridge blend OOF from the superblend cache."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

OUT = r"D:\Parth\ploywin r2\vault\pipeline_out_pretrain"
npz = np.load(OUT + "/superblend_oof.npz", allow_pickle=True)
oof_gbm = npz["oof_gbm"]
oof_mt = npz["oof_mt"]
y = npz["y_train"].astype(np.float64)
tt = npz["target_type_train"].astype(str)

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
idx = {t: np.where(tt == t)[0] for t in TARGETS}

ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
rows = []
for t in TARGETS:
    ix = idx[t]
    yt = y[ix]
    Mx = np.column_stack([oof_gbm[ix], oof_mt[ix]])
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(Mx, yt))
    oof_r2 = {}
    for a in ALPHA_GRID:
        o = np.zeros(len(ix))
        for trk, vk in cv:
            o[vk] = Ridge(alpha=a).fit(Mx[trk], yt[trk]).predict(Mx[vk])
        oof_r2[a] = r2_score(yt, o)
    a_best = max(oof_r2, key=oof_r2.get)
    oof = np.zeros(len(ix))
    for trk, vk in cv:
        lr = Ridge(alpha=a_best); lr.fit(Mx[trk], yt[trk])
        oof[vk] = lr.predict(Mx[vk])
    blend = r2_score(yt, oof)
    gbm = r2_score(yt, oof_gbm[ix])
    gnn = r2_score(yt, oof_mt[ix])
    rows.append(dict(target=t, alpha=a_best, blend=blend, GBM=gbm, GNN=gnn))

df = pd.DataFrame(rows).set_index("target")
print(df.to_string())
print()
mean_blend = df["blend"].mean()
print("mean blend=%.4f | GBM=%.4f | GNN=%.4f" % (mean_blend, df["GBM"].mean(), df["GNN"].mean()))
print("P14 baseline blend=0.8769")
print("delta vs P14=%+.4f" % (mean_blend - 0.8769))
if mean_blend - 0.8769 >= 0.002:
    print("GATE: PASS")
else:
    print("GATE: FAIL")
