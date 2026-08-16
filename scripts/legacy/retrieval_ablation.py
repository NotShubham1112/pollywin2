"""
Experiment 1: Retrieval-only ablation (per-target, leak-safe).

For each target t and fingerprint pool (Morgan r2 / r3, AtomPair, MACCS):
  - Canonical GroupKFold over target-t rows -> train/val fold masks (a canon
    never straddles a fold, so val-row targets never leak into the neighbor pool).
  - OOF: neighbors are taken from the *within-fold train part* of target t only
    (every neighbor has a measured value for t), fit k=11 NN, use top-k.
  - Test: pool = ALL target-t train rows (entire train; test rows are not in the
    pool so this is leak-free).
  - Schemes per row: top1, top3, top5, top10 means; sim-weighted w3/w5/w10;
    sim^2-weighted w3sq/w5sq/w10sq; plus max_sim, mean_sim.

Answers: does retrieval alone already explain EPS/Nc/Ei/Eea? Which pool+scheme
is strongest per target?
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

WORK = r"D:\Parth\ploywin r2"
FP_CACHE = os.path.join(WORK, "vault", "fp_cache.npz")
USE_SMOKE = os.environ.get("SMOKE", "0") == "1"
GLOBAL_FOLDS = 2 if USE_SMOKE else 5

trf = pd.read_pickle(os.path.join(WORK, "r2_train_feat.pkl"))
tef = pd.read_pickle(os.path.join(WORK, "r2_test_feat.pkl"))
TARGETS = sorted(trf["target_type"].unique())
Y = trf["target"].values.astype(np.float64)
T = trf["target_type"].values
G = trf["canon"].values
TARGET_TYPE_TE = tef["target_type"].values
N_TR = len(trf); N_TE = len(tef)
idx_of = {t: np.where(T == t)[0] for t in TARGETS}
print("targets:", TARGETS, "|| train", N_TR, "test", N_TE, "folds", GLOBAL_FOLDS, flush=True)


def fp_smiles(sm_list):
    morgan2 = np.zeros((len(sm_list), 2048), dtype=np.uint8)
    morgan3 = np.zeros((len(sm_list), 2048), dtype=np.uint8)
    maccs = np.zeros((len(sm_list), 2048), dtype=np.uint8)
    for i, s in enumerate(sm_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        morgan2[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
            m, 2, nBits=2048).ToBitString().encode(), 'u1') - ord('0')
        morgan3[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
            m, 3, nBits=2048).ToBitString().encode(), 'u1') - ord('0')
        bts = MACCSkeys.GenMACCSKeys(m).ToBitString()
        maccs[i, :len(bts)] = np.frombuffer(bts.encode(), 'u1') - ord('0')
    return {"morgan2": morgan2, "morgan3": morgan3, "maccs": maccs}


if os.path.exists(FP_CACHE):
    data = np.load(FP_CACHE, allow_pickle=True)
    tr_fp = data["tr_fp"].item(); te_fp = data["te_fp"].item()
    print("loaded fingerprint cache", flush=True)
else:
    t0 = time.time()
    tr_fp = fp_smiles(trf["smiles"].values)
    te_fp = fp_smiles(tef["smiles"].values)
    np.savez(FP_CACHE, tr_fp=tr_fp, te_fp=te_fp)
    print(f"built fingerprints ({time.time()-t0:.0f}s) cached", flush=True)

SCHEMES = ["top1", "top3", "top5", "top10", "w3", "w5", "w10", "w3sq", "w5sq", "w10sq"]


def scheme_preds(sim, nbr_y):
    """Return dict(scheme -> ndarray(len(rows))) of retrieval predictions."""
    out = {sm: np.zeros(sim.shape[0]) for sm in SCHEMES}
    s, v = sim, nbr_y
    out["top1"] = v[:, 0]
    out["top3"] = v[:, :3].mean(1)
    out["top5"] = v[:, :5].mean(1)
    out["top10"] = v[:, :11].mean(1)
    out["w3"] = (v[:, :3] * s[:, :3]).sum(1) / (s[:, :3].sum(1) + 1e-9)
    out["w5"] = (v[:, :5] * s[:, :5]).sum(1) / (s[:, :5].sum(1) + 1e-9)
    out["w10"] = (v[:, :11] * s[:, :11]).sum(1) / (s[:, :11].sum(1) + 1e-9)
    out["w3sq"] = (v[:, :3] * s[:, :3] ** 2).sum(1) / ((s[:, :3] ** 2).sum(1) + 1e-9)
    out["w5sq"] = (v[:, :5] * s[:, :5] ** 2).sum(1) / ((s[:, :5] ** 2).sum(1) + 1e-9)
    out["w10sq"] = (v[:, :11] * s[:, :11] ** 2).sum(1) / ((s[:, :11] ** 2).sum(1) + 1e-9)
    return out


print("\ncomputing leak-safe retrieval OOF (may take a few minutes)...", flush=True)
rows_report = []
for pool in ("morgan2", "morgan3", "maccs"):
    Ftr, Fte = tr_fp[pool], te_fp[pool]
    # test retrieval: pool = full target rows (covered by a filter-by-target below)
    test_te = {t: {sm: np.zeros(N_TE) for sm in SCHEMES} for t in TARGETS}
    oof_te = {t: {sm: np.zeros(len(idx_of[t])) for sm in SCHEMES} for t in TARGETS}

    for t in TARGETS:
        idx = idx_of[t]
        Xt, yt, gt = Ftr[idx], Y[idx], G[idx]
        # --- leak-safe OOF (neighbors only from within-fold train) ---
        for tr_kk, va_kk in GroupKFold(n_splits=GLOBAL_FOLDS).split(Xt, yt, gt):
            nn = NearestNeighbors(n_neighbors=11, metric="jaccard").fit(Xt[tr_kk])
            d, nbr = nn.kneighbors(Xt[va_kk], 11)
            sim = 1.0 - d
            nbr_y = yt[tr_kk][nbr]
            schemes = scheme_preds(sim, nbr_y)
            for sm in SCHEMES:
                oof_te[t][sm][va_kk] = schemes[sm]
        # --- test predictions: pool = ALL target rows of t ---
        nn = NearestNeighbors(n_neighbors=11, metric="jaccard").fit(Xt)
        d, nbr = nn.kneighbors(Fte, 11)
        sim = 1.0 - d
        nbr_y = yt[nbr]
        schemes = scheme_preds(sim, nbr_y)
        for sm in SCHEMES:
            test_te[t][sm] = schemes[sm]

    # report per target (per-pool best-of-scheme) + save to csv
    print(f"\n== pool {pool} ==", flush=True)
    for t in TARGETS:
        idx = idx_of[t]
        yt = Y[idx]
        best = None
        for sm in SCHEMES:
            r = r2_score(yt, oof_te[t][sm])
            if best is None or r > best[1]:
                best = (sm, r)
        # also report the retrieval-only mean across targets for the best scheme
        rows_report.append((pool, t, *[r2_score(Y[idx_of[t]], oof_te[t][sm]) for sm in SCHEMES]))
        print(f"  {t:<4} n={len(idx):>4}  best={best[0]:<6} R2={best[1]:.4f}", flush=True)

cols = ["pool", "target"] + SCHEMES
df = pd.DataFrame(rows_report, columns=cols)
os.makedirs(os.path.join(WORK, "vault", "pipeline_out_pretrain" if not USE_SMOKE else "pipeline_out_pretrain_smoke"), exist_ok=True)
out_path = os.path.join(WORK, "vault",
                        "pipeline_out_pretrain_smoke" if USE_SMOKE else "pipeline_out_pretrain",
                        "retrieval_ablation.csv")
df.round(4).to_csv(out_path, index=False)
print("\nwrote", out_path)

# quick per-target best-pool summary
print("\n=== best pool per target (retrieval-only OOF R2) ===", flush=True)
for t in TARGETS:
    sub = df[df.target == t]
    best_row = sub.loc[sub[SCHEMES].max(axis=1).idxmax()]
    best_sm = SCHEMES[int(np.argmax(sub[SCHEMES].values.max(axis=0)))]
    print(f"  {t:<4} best={best_row['pool']}/{best_sm} R2={best_row[best_sm]:.4f}", flush=True)
print("DONE", flush=True)