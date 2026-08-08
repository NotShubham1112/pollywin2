#!/usr/bin/env python
"""Compare v13 vs v14 (P14) blend_oof_test.npz offline.

Uses the exact evaluation the kernel used: per-target Criterion A corr(GBM,GNN),
Criterion B blend OOF R2 via per-target Ridge(alpha grid). P14 passes if:
  A) corr drops from 0.915-0.969 toward 0.88-0.92, OR
  B) blend OOF gain >= +0.005 vs v13.
"""
import numpy as np, pandas as pd, os

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

def load(path):
    d = np.load(path, allow_pickle=True)
    return d

def analyze(d, label):
    oof_g = d["oof_gbm"].astype(np.float64)
    oof_m = d["oof_mt"].astype(np.float64)
    y = d["y_all"].astype(np.float64)
    g = np.char.decode(d["g_all"]) if d["g_all"].dtype.kind == "S" else d["g_all"].astype(str)
    t = np.char.decode(d["t_all"]) if d["t_all"].dtype.kind == "S" else d["t_all"].astype(str)
    rows = []
    corrs = {}
    for tg in TARGETS:
        idx = np.where(t == tg)[0]
        if len(idx) < 5:
            continue
        corr = np.corrcoef(oof_g[idx], oof_m[idx])[0, 1]
        corrs[tg] = corr
        yt = y[idx]
        Mc = np.column_stack([oof_g[idx], oof_m[idx]])
        gidx = g[idx]
        cv = list(GroupKFold(n_splits=min(5, len(np.unique(gidx)))).split(Mc, yt, gidx))
        best, besta = -np.inf, 1.0
        for a in ALPHA_GRID:
            o = np.zeros(len(idx))
            for trk, vk in cv:
                o[vk] = Ridge(alpha=a).fit(Mc[trk], yt[trk]).predict(Mc[vk])
            r = r2_score(yt, o)
            if r > best:
                best, besta = r, a
        rows.append(dict(target=tg, corr=corr, blend=best, best_alpha=besta,
                         gbm=r2_score(yt, oof_g[idx]), gnn=r2_score(yt, oof_m[idx])))
    df = pd.DataFrame(rows).set_index("target")
    print(f"=== {label} ===")
    print(df.round(4).to_string())
    print(f"mean blend={df['blend'].mean():.4f} gbm={df['gbm'].mean():.4f} gnn={df['gnn'].mean():.4f}")
    print(f"mean corr={np.mean(list(corrs.values())):.4f} (per-target {corrs})\n")
    return df, corrs

v13 = load(r"vault/kernel-v13-multiseed-full/out/blend_oof_test.npz")
v14 = load(r"vault/kernel-v14-p1m/out/blend_oof_test.npz")

df13, c13 = analyze(v13, "v13 baseline")
df14, c14 = analyze(v14, "v14 P14 (full PI1M pretrain)")

print("=== DELTA v14 - v13 ===")
merged = pd.concat({"v13": df13, "v14": df14}, axis=1)
for tg in TARGETS:
    if tg not in c13 or tg not in c14:
        continue
    print(f"{tg:<4} corr {c13[tg]:.3f}->{c14[tg]:.3f}  blend {df13.loc[tg,'blend']:.4f}->{df14.loc[tg,'blend']:.4f} "
          f"({df14.loc[tg,'blend']-df13.loc[tg,'blend']:+.4f})")
print(f"\nblend OOF gain (mean): {df14['blend'].mean()-df13['blend'].mean():+.4f}")
print(f"corr delta (mean): {np.mean(list(c14.values()))-np.mean(list(c13.values())):+.4f}")

ok_a = np.mean(list(c14.values())) <= 0.92
ok_b = df14['blend'].mean() - df13['blend'].mean() >= 0.005
print("\nP14 verdict:", "PASS A (corr dropped)" if ok_a else "A not met (corr still high)",
      "|", "PASS B (blend +0.005)" if ok_b else "B not met (blend gain < .005)")
print("P14 overall:", "PASS -> submit" if (ok_a or ok_b) else "FAIL -> freeze pretraining line")