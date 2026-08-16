"""
Experiment 2: Does appending retrieval features lift the GBM stack over the
base-matrix baseline?

Methodology (leak-safe):
  - Per target, per grouped-CV fold k: fit NN on fold-train fingerprints.
      * val rows: neighbors from fold-train pool only (no leak).
      * train rows: neighbors from fold-train pool, SELF EXCLUDED (drop the
        first neighbor, which is the row itself) -> clean leave-one-out-ish
        retrieval features that see no own label and no val labels.
  - Config A: LGBM on base matrix only (desc + morgan2 + morgan3).
    Config B: LGBM on base matrix + retrieval scheme feats, both computed with
    the identical fold-train pool inside the same fold.
  - The only difference between A and B is the appended columns -> any delta is
    attributable to retrieval features.
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

WORK = r"D:\Parth\ploywin r2"
USE_SMOKE = os.environ.get("SMOKE", "0") == "1"
GLOBAL_FOLDS = 2 if USE_SMOKE else 5
KNN = 11
SELF = 1  # drop self for train queries

trf = pd.read_pickle(os.path.join(WORK, "r2_train_feat.pkl"))
tef = pd.read_pickle(os.path.join(WORK, "r2_test_feat.pkl"))
TARGETS = sorted(trf["target_type"].unique())
Y = trf["target"].values.astype(np.float64)
T = trf["target_type"].values
G = trf["canon"].values
TARGET_TYPE_TE = tef["target_type"].values
idx_of = {t: np.where(T == t)[0] for t in TARGETS}
print("targets:", TARGETS, "|| rows", len(trf), "folds", GLOBAL_FOLDS, flush=True)

FEAT_COLS = [c for c in trf.columns if c not in
             ("target_type", "target", "canon", "inchikey", "smiles", "mol")]
desc_tr = trf[FEAT_COLS].to_numpy(np.float32)
desc_te = tef[FEAT_COLS].to_numpy(np.float32)
print("descriptor block", desc_tr.shape, "n_feat", desc_tr.shape[1], flush=True)

from rdkit import Chem
from rdkit.Chem import AllChem
FP_CACHE = os.path.join(WORK, "vault", "fp_cache.npz")


def fp_morgan(sm_list, rad):
    m = np.zeros((len(sm_list), 2048), dtype=np.uint8)
    for i, s in enumerate(sm_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        m[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
            mol, rad, nBits=2048).ToBitString().encode(), 'u1') - ord('0')
    return m


POOLS = ["morgan2", "morgan3"]
tr_fp = {}
te_fp = {}
if os.path.exists(FP_CACHE):
    data = np.load(FP_CACHE, allow_pickle=True)
    tr_fp = data["tr_fp"].item()
    te_fp = data["te_fp"].item()
    print("loaded fingerprint cache", flush=True)
else:
    for p, rad in (("morgan2", 2), ("morgan3", 3)):
        tr_fp[p] = fp_morgan(trf["smiles"].values, rad)
        te_fp[p] = fp_morgan(tef["smiles"].values, rad)
    np.savez(FP_CACHE, tr_fp=tr_fp, te_fp=te_fp)

Xtr0 = np.hstack([desc_tr, tr_fp["morgan2"], tr_fp["morgan3"]]).astype(np.float32)
print("full base matrix", Xtr0.shape, "n_feat", Xtr0.shape[1], flush=True)

SCHEMES = ["top5", "top10", "w5", "w10", "w5sq", "w10sq"]


def scheme_feats(sim, nbr_y):
    out = {}
    out["top5"] = nbr_y[:, :5].mean(1)
    out["top10"] = nbr_y[:, :11].mean(1)
    out["w5"] = (nbr_y[:, :5] * sim[:, :5]).sum(1) / (sim[:, :5].sum(1) + 1e-9)
    out["w10"] = (nbr_y[:, :11] * sim[:, :11]).sum(1) / (sim[:, :11].sum(1) + 1e-9)
    out["w5sq"] = (nbr_y[:, :5] * sim[:, :5] ** 2).sum(1) / ((sim[:, :5] ** 2).sum(1) + 1e-9)
    out["w10sq"] = (nbr_y[:, :11] * sim[:, :11] ** 2).sum(1) / ((sim[:, :11] ** 2).sum(1) + 1e-9)
    return out


def retrieval_feats_for_fold(Xfit, yfit, Xq, yq, query_in_fit):
    """scheme feats for query rows; neighbors from Xfit/yfit pool.
    query_in_fit=True -> drop self-neighbor (first col)."""
    nq = len(Xq)
    nn = NearestNeighbors(n_neighbors=KNN + (1 if query_in_fit else 0),
                          metric="jaccard").fit(Xfit)
    d, nbr = nn.kneighbors(Xq, KNN + (1 if query_in_fit else 0))
    if query_in_fit:
        d = d[:, 1:]; nbr = nbr[:, 1:]
    sim = 1.0 - np.clip(d, 0, 1.0)
    sf = scheme_feats(sim, yfit[nbr])
    return np.column_stack([sf[sm] for sm in SCHEMES])


def grouped_cv(Xa, ya, ga, Xfp, use_retrieval):
    """OOF preds for one target. Xa = base-matrix rows (target rows only).
    Xfp = {pool: (fp rows of the target)}. use_retrieval appends fold-local
    retrieval feats computed from the fold-train pool."""
    n = len(ya)
    oof = np.zeros(n)
    splits = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Xa, ya, ga))
    for trk, vk in splits:
        Xtr_base = Xa[trk]
        Xva_base = Xa[vk]
        if use_retrieval:
            tr_feats = np.hstack([retrieval_feats_for_fold(
                Xfp[p][trk], ya[trk], Xfp[p][trk], ya[trk], True) for p in POOLS])
            va_feats = np.hstack([retrieval_feats_for_fold(
                Xfp[p][trk], ya[trk], Xfp[p][vk], ya[vk], False) for p in POOLS])
            Xtr = np.hstack([Xtr_base, tr_feats])
            Xva = np.hstack([Xva_base, va_feats])
        else:
            Xtr, Xva = Xtr_base, Xva_base
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(Xtr, ya[trk])
        oof[vk] = m.predict(Xva)
    return oof


lgb_params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                  random_state=42, verbose=-1, n_jobs=8)

# fp blocks aligned to target row order (already global index order)
FP_TR = {p: tr_fp[p].astype(np.float32) for p in POOLS}

print("\n=== config A: base matrix only ===", flush=True)
res_base = {}
for t in TARGETS:
    idx = idx_of[t]
    fp_t = {p: FP_TR[p][idx] for p in POOLS}
    pred = grouped_cv(Xtr0[idx], Y[idx], G[idx], fp_t, False)
    res_base[t] = r2_score(Y[idx], pred)
    print(f"  {t:<4} n={len(idx):>4} R2={res_base[t]:.4f}", flush=True)
print("  mean %.4f" % np.mean(list(res_base.values())), flush=True)

print("\n=== config B: base matrix + retrieval feats ===", flush=True)
res_retr = {}
for t in TARGETS:
    idx = idx_of[t]
    fp_t = {p: FP_TR[p][idx] for p in POOLS}
    pred = grouped_cv(Xtr0[idx], Y[idx], G[idx], fp_t, True)
    res_retr[t] = r2_score(Y[idx], pred)
    print(f"  {t:<4} n={len(idx):>4} base={res_base[t]:.4f} +retr={res_retr[t]:.4f} delta={res_retr[t]-res_base[t]:+.4f}",
          flush=True)
print("  mean base %.4f | +retr %.4f | delta %+.4f" % (
    np.mean(list(res_base.values())), np.mean(list(res_retr.values())),
    np.mean(list(res_retr.values())) - np.mean(list(res_base.values()))), flush=True)

outdir = os.path.join(WORK, "vault", "pipeline_out_pretrain_smoke" if USE_SMOKE else "pipeline_out_pretrain")
os.makedirs(outdir, exist_ok=True)
pd.DataFrame({"target": TARGETS,
              "base": [res_base[t] for t in TARGETS],
              "retr": [res_retr[t] for t in TARGETS]}).round(4).to_csv(
    os.path.join(outdir, "exp2_retrieval_feats.csv"), index=False)
print("DONE", flush=True)