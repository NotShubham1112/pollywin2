"""
Experiment 3: Specialist per-target blend (GBM vs retrieval) on fold-consistent
OOF predictions (level-2). Reuses the exact base matrix, folds, and retrieval
pipeline from Experiment 2 so config A (GBM-only OOF mean) is a true baseline.

Per target, per grouped fold:
  - GBM OOF pred (base matrix)
  - retrieval w5sq OOF pred per pool (morgan2, morgan3)
  - level-2: Ridge stacker fit on fold-train, scored on fold-val
  - oracle floor: best single specialist of {GBM, m2, m3} per target (the
    routing ceiling we chase)

Reported per target + means.
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

WORK = r"D:\Parth\ploywin r2"
USE_SMOKE = os.environ.get("SMOKE", "0") == "1"
GLOBAL_FOLDS = 2 if USE_SMOKE else 5
KNN = 11

trf = pd.read_pickle(os.path.join(WORK, "r2_train_feat.pkl"))
tef = pd.read_pickle(os.path.join(WORK, "r2_test_feat.pkl"))
TARGETS = sorted(trf["target_type"].unique())
Y = trf["target"].values.astype(np.float64)
T = trf["target_type"].values
G = trf["canon"].values
idx_of = {t: np.where(T == t)[0] for t in TARGETS}
print("targets:", TARGETS, "|| rows", len(trf), "folds", GLOBAL_FOLDS, flush=True)

FEAT_COLS = [c for c in trf.columns if c not in
             ("target_type", "target", "canon", "inchikey", "smiles", "mol")]
desc_tr = trf[FEAT_COLS].to_numpy(np.float32)

from rdkit import Chem
from rdkit.Chem import AllChem
FP_CACHE = os.path.join(WORK, "vault", "fp_cache.npz")
data = np.load(FP_CACHE, allow_pickle=True)
tr_fp = data["tr_fp"].item()
te_fp = data["te_fp"].item()
print("loaded fingerprint cache", flush=True)

Xtr0 = np.hstack([desc_tr, tr_fp["morgan2"], tr_fp["morgan3"]]).astype(np.float32)
FP_TR = {p: tr_fp[p].astype(np.float32) for p in ("morgan2", "morgan3")}
POOLS = ("morgan2", "morgan3")

lgb_params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                  random_state=42, verbose=-1, n_jobs=8)


def retr_w5sq(Xfit, yfit, Xq):
    nn = NearestNeighbors(n_neighbors=KNN, metric="jaccard").fit(Xfit)
    d, nbr = nn.kneighbors(Xq, KNN)
    sim = 1.0 - np.clip(d, 0, 1.0)
    v = yfit[nbr]
    return (v * sim ** 2).sum(1) / ((sim ** 2).sum(1) + 1e-9)


spl_all = {}
for t in TARGETS:
    idx = idx_of[t]
    spl_all[t] = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Xtr0[idx], Y[idx], G[idx]))


def oof_preds(t):
    idx = idx_of[t]
    m = len(idx)
    yt = Y[idx]
    oof_g = np.zeros(m)
    oof_r = {p: np.zeros(m) for p in POOLS}
    for trk, vk in spl_all[t]:
        g = lgb.LGBMRegressor(**lgb_params)
        g.fit(Xtr0[idx][trk], yt[trk])
        oof_g[vk] = g.predict(Xtr0[idx][vk])
        for p in POOLS:
            oof_r[p][vk] = retr_w5sq(FP_TR[p][idx][trk], yt[trk], FP_TR[p][idx][vk])
    return oof_g, oof_r


print("\n=== per-target specialist OOF (GBM vs retrieval w5sq) ===", flush=True)
rows = []
t0 = time.time()
for t in TARGETS:
    idx = idx_of[t]
    yt = Y[idx]
    oof_g, oof_r = oof_preds(t)
    L2 = np.column_stack([oof_g, oof_r["morgan2"], oof_r["morgan3"]])
    oof_l2 = np.zeros(len(idx))
    for trk, vk in spl_all[t]:
        lr = Ridge(alpha=1.0)
        lr.fit(L2[trk], yt[trk])
        oof_l2[vk] = lr.predict(L2[vk])
    r_g = r2_score(yt, oof_g)
    r_m2 = r2_score(yt, oof_r["morgan2"])
    r_m3 = r2_score(yt, oof_r["morgan3"])
    r_l2 = r2_score(yt, oof_l2)
    floor = max(r_g, r_m2, r_m3)
    rows.append(dict(target=t, GBM=r_g, m2=r_m2, m3=r_m3, stack=r_l2, floor=floor))
    print(f"  {t:<4} n={len(idx):>4} GBM={r_g:.4f} m2={r_m2:.4f} m3={r_m3:.4f} "
          f"stack={r_l2:.4f} floor={floor:.4f}", flush=True)

df = pd.DataFrame(rows).set_index("target")
print("\n=== summary ===", flush=True)
print("  means:", {c: round(df[c].mean(), 4) for c in ("GBM", "m2", "m3", "stack", "floor")}, flush=True)

outdir = os.path.join(WORK, "vault", "pipeline_out_pretrain_smoke" if USE_SMOKE else "pipeline_out_pretrain")
os.makedirs(outdir, exist_ok=True)
out_path = os.path.join(outdir, "exp3_blend.csv")
df.round(4).to_csv(out_path, index=True)
print("wrote", out_path, flush=True)
print("DONE", flush=True)