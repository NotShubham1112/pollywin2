"""
v13 PRODUCTION: GBM + MT-GNN level-2 blend -> submission.csv.

Freezes the winning architecture from Exp 4:
  Level 0: GBM trio ridge-stack OOF + MT-GNN OOF  (persisted by mt_gnn_v2.py)
  Level 1: per-target Ridge(alpha=1.0) on [GBM, GNN] OOF, fold-consistent.

Sanity checks (per target):
  1. corr(GBM, GNN) - want meaningful disagreement (< 0.99).
  2. ridge coefficients (mean over folds) - small targets should lean GNN.
  3. per-target R2 vs GBM-only / GNN-only.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

WORK = r"D:\Parth\ploywin r2"
USE_SMOKE = os.environ.get("SMOKE", "0") == "1"
GLOBAL_FOLDS = 2 if USE_SMOKE else 5
ALPHA = 1.0

outdir = os.path.join(WORK, "vault", "pipeline_out_pretrain_smoke" if USE_SMOKE else "pipeline_out_pretrain")
NPZ = os.path.join(outdir, "superblend_oof.npz")

trf = pd.read_pickle(os.path.join(WORK, "r2_train_feat.pkl"))
tef = pd.read_pickle(os.path.join(WORK, "r2_test_feat.pkl"))
TARGETS = sorted(trf["target_type"].unique())
Y = trf["target"].values.astype(np.float64)
T = trf["target_type"].values
G = trf["canon"].values
idx_of = {t: np.where(T == t)[0] for t in TARGETS}
print("targets:", TARGETS, "|| rows", len(trf), "test", len(tef),
      "folds", GLOBAL_FOLDS, flush=True)

d = np.load(NPZ, allow_pickle=True)
oof_gbm = d["oof_gbm"].astype(np.float64)
oof_mt = d["oof_mt"].astype(np.float64)
test_gbm = d["test_gbm"].astype(np.float64)
test_mt = d["test_mt"].astype(np.float64)
assert not np.isnan(oof_gbm).any() and not np.isnan(oof_mt).any()
print("loaded OOF/test: gbm/mt", flush=True)

# ---- sanity check 1: correlation per target ----
print("\n=== Check 1: corr(GBM, GNN) per target ===", flush=True)
for t in TARGETS:
    idx = idx_of[t]
    c = np.corrcoef(oof_gbm[idx], oof_mt[idx])[0, 1]
    print(f"  {t:<4} corr={c:.4f}", flush=True)

# ---- level-2 per-target ridge, fold-consistent ----
print("\n=== building v13 blend (per-target Ridge) ===", flush=True)
rows = []
coefs = {t: [] for t in TARGETS}
final_te = np.zeros(len(tef))
for t in TARGETS:
    idx = idx_of[t]
    yt = Y[idx]
    M = np.column_stack([oof_gbm[idx], oof_mt[idx]])
    Mte = np.column_stack([test_gbm, test_mt])
    oof = np.zeros(len(idx))
    te_pred = np.zeros(len(tef))
    for trk, vk in GroupKFold(n_splits=GLOBAL_FOLDS).split(M, yt, G[idx]):
        lr = Ridge(alpha=ALPHA)
        lr.fit(M[trk], yt[trk])
        oof[vk] = lr.predict(M[vk])
        te_pred += lr.predict(Mte) / GLOBAL_FOLDS
        coefs[t].append(lr.coef_.tolist())
    m_te = (tef["target_type"] == t).values
    final_te[m_te] = te_pred[m_te]
    r_blend = r2_score(yt, oof)
    r_g = r2_score(yt, oof_gbm[idx])
    r_m = r2_score(yt, oof_mt[idx])
    cb = np.mean(coefs[t], axis=0)
    rows.append(dict(target=t, blend=r_blend, GBM=r_g, GNN=r_m,
                     w_GBM=cb[0], w_GNN=cb[1]))
    print(f"  {t:<4} blend={r_blend:.4f} GBM={r_g:.4f} GNN={r_m:.4f} "
          f"w_GBM={cb[0]:.3f} w_GNN={cb[1]:.3f}", flush=True)

df = pd.DataFrame(rows).set_index("target")
print("\n=== summary ===", flush=True)
print("  mean blend=%.4f | GBM=%.4f | GNN=%.4f | delta-vs-GBM %+.4f" % (
    df["blend"].mean(), df["GBM"].mean(), df["GNN"].mean(),
    df["blend"].mean() - df["GBM"].mean()), flush=True)

# ---- sanity check 2/3: weights ----
print("\n=== Check 2: blend weights (small targets should lean GNN) ===", flush=True)
for t in TARGETS:
    cb = np.mean(coefs[t], axis=0)
    print(f"  {t:<4} w_GBM={cb[0]:.3f} w_GNN={cb[1]:.3f} "
          f"GNN_share={cb[1]/(cb.sum()):.2f}", flush=True)

sub = pd.DataFrame({"id": tef["id"].values, "target": final_te})
sub_path = os.path.join(outdir, "submission_v13.csv")
sub.to_csv(sub_path, index=False)
print("\nwrote", sub_path, flush=True)
print("  rows", len(sub), "| NaN", sub["target"].isna().sum(),
      "| range [%.2f, %.2f]" % (sub["target"].min(), sub["target"].max()), flush=True)

df.round(4).to_csv(os.path.join(outdir, "v13_blend_report.csv"), index=True)
print("wrote", os.path.join(outdir, "v13_blend_report.csv"), flush=True)
print("DONE", flush=True)