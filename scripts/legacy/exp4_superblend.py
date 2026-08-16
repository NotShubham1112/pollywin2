"""
Experiment 4: GBM + MT-GNN + Retrieval level-2 super-blend.

Loads fold-consistent OOF predictions persisted by mt_gnn_v2.py
(superblend_oof.npz): oof_gbm (GBM trio ridge-stack), oof_mt (MT-GNN), plus
test preds. Recomputes retrieval w5sq OOF with the SAME per-target GroupKFold
grouping (canon -> one fold) so all three specialist OOF arrays are on an
identical leak-safe footing.

Runs the ablation matrix the user specified:
  GBM only | GNN only | Retr only | GBM+GNN | GBM+Retr | GNN+Retr | all
per target via level-2 Ridge, then reports per-target + mean R2.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

WORK = r"D:\Parth\ploywin r2"
USE_SMOKE = os.environ.get("SMOKE", "0") == "1"
GLOBAL_FOLDS = 2 if USE_SMOKE else 5
KNN = 11

outdir = os.path.join(WORK, "vault", "pipeline_out_pretrain_smoke" if USE_SMOKE else "pipeline_out_pretrain")
NPZ = os.path.join(outdir, "superblend_oof.npz")

trf = pd.read_pickle(os.path.join(WORK, "r2_train_feat.pkl"))
tef = pd.read_pickle(os.path.join(WORK, "r2_test_feat.pkl"))
TARGETS = sorted(trf["target_type"].unique())
Y = trf["target"].values.astype(np.float64)
T = trf["target_type"].values
G = trf["canon"].values
idx_of = {t: np.where(T == t)[0] for t in TARGETS}
print("targets:", TARGETS, "|| rows", len(trf), "folds", GLOBAL_FOLDS, flush=True)

d = np.load(NPZ, allow_pickle=True)
oof_gbm = d["oof_gbm"].astype(np.float64)
oof_mt = d["oof_mt"].astype(np.float64)
print("loaded OOF: gbm", oof_gbm.shape, "mt", oof_mt.shape, flush=True)
assert not np.isnan(oof_gbm).any() and not np.isnan(oof_mt).any()

from rdkit import Chem
from rdkit.Chem import AllChem
FP_CACHE = os.path.join(WORK, "vault", "fp_cache.npz")
fpd = np.load(FP_CACHE, allow_pickle=True)
tr_fp = fpd["tr_fp"].item()
print("loaded fingerprint cache", flush=True)
FP_TR = {p: tr_fp[p].astype(np.float32) for p in ("morgan2", "morgan3")}


def retr_w5sq(Xfit, yfit, Xq):
    nn = NearestNeighbors(n_neighbors=KNN, metric="jaccard").fit(Xfit)
    dd, nbr = nn.kneighbors(Xq, KNN)
    sim = 1.0 - np.clip(dd, 0, 1.0)
    v = yfit[nbr]
    return (v * sim ** 2).sum(1) / ((sim ** 2).sum(1) + 1e-9)


def retr_oof(pool):
    """Fold-consistent retrieval OOF aligned to global train rows."""
    oof = np.full(len(trf), np.nan, dtype=np.float64)
    for t in TARGETS:
        idx = idx_of[t]
        Xt = FP_TR[pool][idx]; yt = Y[idx]; gt = G[idx]
        for trk, vk in GroupKFold(n_splits=GLOBAL_FOLDS).split(Xt, yt, gt):
            oof[idx[vk]] = retr_w5sq(Xt[trk], yt[trk], Xt[vk])
    assert not np.isnan(oof).any()
    return oof


print("\ncomputing fold-consistent retrieval OOF...", flush=True)
t0 = time.time()
oof_r2 = retr_oof("morgan2")
oof_r3 = retr_oof("morgan3")
print(f"done {time.time()-t0:.0f}s", flush=True)


def l2_blend(names):
    """Level-2 Ridge stack over given specialist OOF columns, fold-consistent.
    Returns dict target -> oof preds, plus per-target R2 dict."""
    cols = [np.zeros(len(trf)) for _ in names]
    for i, nm in enumerate(names):
        if nm == "gbm":
            cols[i] = oof_gbm
        elif nm == "mt":
            cols[i] = oof_mt
        elif nm == "r2":
            cols[i] = oof_r2
        elif nm == "r3":
            cols[i] = oof_r3
    R = {}
    PRED = {}
    for t in TARGETS:
        idx = idx_of[t]
        M = np.column_stack([c[idx] for c in cols])
        oof = np.zeros(len(idx))
        for trk, vk in GroupKFold(n_splits=GLOBAL_FOLDS).split(M, Y[idx], G[idx]):
            lr = Ridge(alpha=1.0)
            lr.fit(M[trk], Y[idx][trk])
            oof[vk] = lr.predict(M[vk])
        R[t] = r2_score(Y[idx], oof)
        PRED[t] = oof
    return R, PRED


CONFIGS = [
    ("GBM only", ["gbm"]),
    ("GNN only", ["mt"]),
    ("Retr only", ["r2", "r3"]),
    ("GBM+GNN", ["gbm", "mt"]),
    ("GBM+Retr", ["gbm", "r2", "r3"]),
    ("GNN+Retr", ["mt", "r2", "r3"]),
    ("GBM+GNN+Retr", ["gbm", "mt", "r2", "r3"]),
]

print("\n=== ablation matrix (level-2 ridge, per target) ===", flush=True)
summary = {}
for name, cols in CONFIGS:
    R, _ = l2_blend(cols)
    summary[name] = R
    print(f"\n  {name:<16} mean={np.mean(list(R.values())):.4f}", flush=True)
    for t in TARGETS:
        print(f"      {t:<4} {R[t]:.4f}", flush=True)

print("\n=== summary table ===", flush=True)
names = [c[0] for c in CONFIGS]
df = pd.DataFrame({name: [summary[name][t] for t in TARGETS] for name in names},
                  index=TARGETS)
df.loc["mean"] = df.mean()
print(df.round(4), flush=True)

# also report the simple weighted average of the three final specialists (no L2)
w = np.zeros(len(trf))
for t in TARGETS:
    idx = idx_of[t]
    w[idx] = (oof_gbm[idx] + oof_mt[idx] + oof_r2[idx] + oof_r3[idx]) / 4.0
print("\n  simple-avg of 4 specialists mean R2: %.4f" % np.mean(
    [r2_score(Y[idx_of[t]], w[idx_of[t]]) for t in TARGETS]), flush=True)

out_path = os.path.join(outdir, "exp4_superblend.csv")
df.round(4).to_csv(out_path, index=True)
print("wrote", out_path, flush=True)
print("DONE", flush=True)