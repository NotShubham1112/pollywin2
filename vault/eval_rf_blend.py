"""RF blend arm evaluation v2: train RF OOF PER-TARGET (matching GBM pipeline),
blend with existing GBM + MT-GNN OOF via per-target Ridge."""
import os, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors, Crippen, GraphDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

WORK = r"D:\Parth\ploywin r2"
OUT_CACHE = os.path.join(WORK, "vault", "kernel-v14-p1m", "out", "blend_oof_test.npz")
npz = np.load(OUT_CACHE, allow_pickle=True)
oof_gbm = npz["oof_gbm"]
oof_mt = npz["oof_mt"]
y = npz["y_all"].astype(np.float64)
t_all = npz["t_all"].astype(str)
g_all = npz["g_all"].astype(str)

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
idx = {t: np.where(t_all == t)[0] for t in TARGETS}
print("Target counts:", {t: len(idx[t]) for t in TARGETS})

# --- Compute original P14 features (22 descriptors + Morgan r=2 2048 + MACCS 167) ---
print("Loading train CSVs...", flush=True)
tr = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))

def feats(m):
    if m is None:
        return [np.nan] * 22
    return [
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
        Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
        Descriptors.RingCount(m), Descriptors.NumAromaticRings(m),
        Descriptors.NumAliphaticRings(m), Descriptors.NumSaturatedRings(m),
        Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumHeavyAtoms(m),
        Descriptors.NumHeteroatoms(m), Descriptors.FractionCSP3(m),
        Crippen.MolMR(m), rdMolDescriptors.CalcNumBridgeheadAtoms(m),
        rdMolDescriptors.CalcNumSpiroAtoms(m),
        rdMolDescriptors.CalcNumAromaticAtoms(m) if hasattr(rdMolDescriptors, "CalcNumAromaticAtoms") else Descriptors.NumAromaticRings(m),
        GraphDescriptors.BalabanJ(m), GraphDescriptors.Ipc(m),
        rdMolDescriptors.CalcNumLipinskiHBA(m), rdMolDescriptors.CalcNumLipinskiHBD(m),
        rdMolDescriptors.CalcNumAtomStereoCenters(m),
    ]

FNAMES = ["MolWt", "LogP", "TPSA", "HDon", "HAccep", "RingCnt", "AroRing", "AliRing", "SatRing",
          "RotB", "HeavyAt", "HeteroAt", "FracCSP3", "MR", "Bridge", "Spiro", "AroAt",
          "BalabanJ", "Ipc", "LipHBA", "LipHBD", "Stereo"]

print("Computing descriptors...", flush=True)
tr_f = np.array(tr["smiles"].map(lambda s: feats(Chem.MolFromSmiles(s))).tolist())
tr[FNAMES] = tr_f

# Compute fingerprints for train
print("Computing fingerprints...", flush=True)
t0 = time.time()
morgan_tr = np.zeros((len(tr), 2048), dtype=np.float32)
maccs_tr = np.zeros((len(tr), 167), dtype=np.float32)
for i, s in enumerate(tr["smiles"]):
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    morgan_tr[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
        m, 2, nBits=2048).ToBitString().encode(), "u1") - ord("0")
    maccs_tr[i] = np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(),
                                 "u1") - ord("0")
print(f"Fingerprints computed in {time.time()-t0:.0f}s", flush=True)

# Build full feature matrix (same as P14)
FEAT_COLS = FNAMES
F32_MAX = np.finfo(np.float32).max
D_tr = np.clip(tr[FEAT_COLS].values, -F32_MAX, F32_MAX)
for j in range(D_tr.shape[1]):
    col = D_tr[:, j]
    med = np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
    col[~np.isfinite(col)] = med
X = np.hstack([D_tr, morgan_tr, maccs_tr]).astype(np.float32)
Xs = StandardScaler().fit(X).transform(X).astype(np.float32)
print("Feature matrix:", X.shape, "(expected 7409 x 2237)")

# Canonical SMILES for GroupKFold
from rdkit.Chem import MolToSmiles
canon_map = {}
G = []
for s in tr["smiles"]:
    if s not in canon_map:
        m = Chem.MolFromSmiles(s)
        canon_map[s] = MolToSmiles(m) if m else s
    G.append(canon_map[s])
G = np.array(G)

# --- Train RF OOF PER-TARGET with GroupKFold(5) ---
T = tr["target_type"].values
idx_of_target_local = {t: np.where(T == t)[0] for t in TARGETS}

N_FOLDS = 5
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=5,
    max_features=0.5,
    n_jobs=-1,
    random_state=42,
)

print(f"\nTraining PER-TARGET RF OOF ({N_FOLDS} folds)...", flush=True)
rf_oof_global = np.zeros(len(X), dtype=np.float64)
rf_oof = {}  # per-target OOF, aligned with idx arrays
for t in TARGETS:
    ix = idx_of_target_local[t]
    Xt, yt, gt = Xs[ix], y[ix], G[ix]
    rf_oof_t = np.zeros(len(ix), dtype=np.float64)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xt, yt, gt)):
        t0 = time.time()
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(Xt[tr_idx], yt[tr_idx])
        pred = rf.predict(Xt[va_idx])
        rf_oof_t[va_idx] = pred
        r2 = r2_score(yt[va_idx], pred)
        print(f"  {t} fold {fold}: R2={r2:.4f} ({time.time()-t0:.0f}s)", flush=True)
    rf_oof[t] = rf_oof_t
    rf_oof_global[ix] = rf_oof_t
    rf_r2 = r2_score(yt, rf_oof_t)
    print(f"  {t} full RF R2={rf_r2:.4f}", flush=True)

# --- Blend: GBM + MT-GNN + RF via per-target Ridge ---
print("\n=== Per-target Ridge blend (3-arm: GBM, MT-GNN, RF) ===")
SMALL_FIVE = ["egb", "eps", "nc", "ei", "eea"]
ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
rows = []
for t in TARGETS:
    ix = idx[t]
    yt = y[ix]
    Mx = np.column_stack([oof_gbm[ix], oof_mt[ix], rf_oof[t]])
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
    rf_r2 = r2_score(yt, rf_oof[t])
    # 2-arm blend (GBM + GNN) for comparison
    Mx2 = np.column_stack([oof_gbm[ix], oof_mt[ix]])
    oof2 = np.zeros(len(ix))
    for trk, vk in cv:
        oof2[vk] = Ridge(alpha=a_best).fit(Mx2[trk], yt[trk]).predict(Mx2[vk])
    blend2 = r2_score(yt, oof2)
    rows.append(dict(target=t, alpha=a_best, blend3=blend, blend2=blend2,
                     GBM=gbm, GNN=gnn, RF=rf_r2,
                     in_small5=t in SMALL_FIVE))

df = pd.DataFrame(rows).set_index("target")
print(df.to_string(float_format="%.4f"))
print()

mean3 = df["blend3"].mean()
mean2 = df["blend2"].mean()
print(f"Mean 3-arm blend: {mean3:.4f}")
print(f"Mean 2-arm blend (P14): {mean2:.4f}")
print(f"Mean GBM: {df['GBM'].mean():.4f} | GNN: {df['GNN'].mean():.4f} | RF: {df['RF'].mean():.4f}")
print()

# Gate checks
print("=== GATE CHECKS ===")
print(f"1. 3-arm blend >= P14 (0.8769): {mean3:.4f} (delta {mean3 - 0.8769:+.4f})")
print(f"2. 3-arm blend >= 2-arm blend: {mean3:.4f} vs {mean2:.4f} (delta {mean3 - mean2:+.4f})")

# RF correlation with GBM per target
print("\n3. RF-GBM correlation per target:")
for t in TARGETS:
    ix = idx[t]
    corr = np.corrcoef(rf_oof[t], oof_gbm[ix])[0, 1]
    print(f"   {t}: corr(RF, GBM) = {corr:.4f}")

# Small-five specific
sf = df[df["in_small5"]]
print(f"\n4. Small-five mean: 3-arm={sf['blend3'].mean():.4f} vs 2-arm={sf['blend2'].mean():.4f} (delta {sf['blend3'].mean() - sf['blend2'].mean():+.4f})")
print(f"5. Small-five RF R2: {sf['RF'].mean():.4f}")
print(f"6. No target regression: ", end="")
regressed = []
for t in TARGETS:
    best_single = max(df.loc[t, "GBM"], df.loc[t, "GNN"], df.loc[t, "RF"])
    if df.loc[t, "blend3"] < best_single - 0.001:
        regressed.append(t)
if regressed:
    print(f"FAIL - regressed: {regressed}")
else:
    print("PASS")

overall_delta = mean3 - 0.8769
print(f"\n=== VERDICT: delta={overall_delta:+.4f} ===")
if overall_delta >= 0.001:
    print("GATE: PASS - integrate RF blend into notebook")
else:
    print("GATE: FAIL - do not integrate")
