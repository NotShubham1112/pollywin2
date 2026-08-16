import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import warnings
warnings.filterwarnings("ignore")
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, rdFreeSASA, Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import GraphDescriptors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

D = "C:/Users/shubh/.cache/kagglehub/competitions/ppp-round-2/"
tr = pd.read_csv(D + "train.csv")
te = pd.read_csv(D + "test.csv")
PI = pd.read_csv(D + "PI1M.csv")["SMILES"].astype(str)

def canonical(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None, None
    try:
        c = Chem.MolToSmiles(m)
        ik = Chem.MolToInchiKey(m)
    except Exception:
        return Chem.MolToSmiles(m), None, None
    return c, ik, m

print(f"train {tr.shape} test {te.shape} PI1M {len(PI)}")

t0 = time.time()
tpl = tr["smiles"].map(canonical)
tr["canon"], tr["inchikey"], _ = zip(*tpl)
te_tpl = te["smiles"].map(canonical)
te["canon"], te["inchikey"], _ = zip(*te_tpl)

# fast descriptor set (no SASA - slow)
def feats(m):
    if m is None:
        return [np.nan]*22
    return [
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
        Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
        Descriptors.RingCount(m), Descriptors.NumAromaticRings(m),
        Descriptors.NumAliphaticRings(m), Descriptors.NumSaturatedRings(m),
        Descriptors.NumRotatableBonds(m), rdMolDescriptors.CalcNumHeavyAtoms(m),
        Descriptors.NumHeteroatoms(m), Descriptors.FractionCSP3(m),
        Crippen.MolMR(m), rdMolDescriptors.CalcNumBridgeheadAtoms(m),
        rdMolDescriptors.CalcNumSpiroAtoms(m),
        rdMolDescriptors.CalcNumAromaticAtoms(m) if hasattr(rdMolDescriptors, 'CalcNumAromaticAtoms') else Descriptors.NumAromaticRings(m),
        GraphDescriptors.BalabanJ(m), GraphDescriptors.Ipc(m),
        rdMolDescriptors.CalcNumLipinskiHBA(m), rdMolDescriptors.CalcNumLipinskiHBD(m),
        rdMolDescriptors.CalcNumAtomStereoCenters(m),
    ]

FNAMES = ["MolWt","LogP","TPSA","HDon","HAccep","RingCnt","AroRing","AliRing","SatRing",
          "RotB","HeavyAt","HeteroAt","FracCSP3","MR","Bridge","Spiro","AroAt",
          "BalabanJ","Ipc","LipHBA","LipHBD","Stereo"]

tr_f = np.array(tr["smiles"].map(lambda s: feats(Chem.MolFromSmiles(s))).tolist())
te_f = np.array(te["smiles"].map(lambda s: feats(Chem.MolFromSmiles(s))).tolist())
tr[FNAMES] = tr_f
te[FNAMES] = te_f
print(f"descriptors done {time.time()-t0:.1f}s")

# --- twin/leakage stats (canonical SMILES as group key; InChIKey fails on * dummy atoms) ---
print("\n=== LEAKAGE ===")
print("train dup smiles:", tr.smiles.duplicated().sum())
train_canon = set(tr["canon"].dropna())
test_canon = set(te["canon"].dropna())
print("test rows w/ canonical twin in train:", te["canon"].isin(train_canon).sum(), "/", len(te))
print("test unique canonical present in train:", len(test_canon & train_canon), "/", len(test_canon))
print("train rows w/ canonical dup:", tr.duplicated("canon").sum())

# PI1M retrieval feasibility - sample 100k to estimate (full 1M ~5min)
pi_sample = PI.sample(100000, random_state=42)
def _c(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None
t0 = time.time()
pi_canon = set([k for k in map(_c, pi_sample.tolist()) if k])
print(f"PI1M sample ({len(pi_sample)}) canon computed in {time.time()-t0:.0f}s, unique={len(pi_canon)}")
print("\n=== RETRIEVAL (PI1M sample) ===")
train_c = set(tr["canon"].dropna())
test_c = set(te["canon"].dropna())
print("PI1M-sample overlap w/ train:", len(train_c & pi_canon), "/", len(train_c))
print("PI1M-sample overlap w/ test:", len(test_c & pi_canon), "/", len(test_c))
for tt in tr["target_type"].unique():
    sub = tr[tr["target_type"] == tt]
    cov = sub["canon"].isin(pi_canon).mean()
    print(f"  {tt}: train rows={len(sub)} PI1M-twin-coverage={cov:.2f}")

# --- grouped 5-fold CV per target (group by canonical SMILES to prevent twin leakage) ---
print("\n=== LIGHTGBM GROUPED CV (real numbers) ===")
def clean(X, y, grp=None):
    ok = ~np.isnan(X).any(1)
    if grp is None:
        return X[ok], y[ok]
    return X[ok], y[ok], grp[ok]

lgb_params = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                  random_state=42, verbose=-1, n_jobs=8)

results = {}
for tt in tr["target_type"].unique():
    sub = tr[tr["target_type"] == tt].copy()
    X = sub[FNAMES].to_numpy()
    y = sub["target"].to_numpy()
    grp = sub["canon"].fillna(sub["smiles"]).to_numpy()
    X, y, grp = clean(X, y, grp)
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr_i, va_i in gkf.split(X, y, grp):
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X[tr_i], y[tr_i], eval_set=[(X[va_i], y[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_i] = m.predict(X[va_i])
    results[tt] = dict(r2=r2_score(y, oof), n=len(y), n_unique=len(np.unique(grp)))
    print(f"  {tt:>3}: R2={r2_score(y, oof):.4f}  n={len(y):5d}  unique={len(np.unique(grp)):4d}")

mean_r2 = np.mean([v["r2"] for v in results.values()])
print(f"\nMEAN R2 over 7 targets: {mean_r2:.4f}")
print(f"rows-weighted R2: {np.average([v['r2'] for v in results.values()], weights=[v['n'] for v in results.values()]):.4f}")

tr.to_pickle("r2_train_feat.pkl")
te.to_pickle("r2_test_feat.pkl")
print("saved r2_train_feat.pkl / r2_test_feat.pkl")
