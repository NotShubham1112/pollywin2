import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os, time, sys
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb

OUT = os.path.join("vault", "figures")
train = pd.read_csv("official_dataset/train.csv")
DESC_NAMES = [d[0] for d in Descriptors.descList]

def parse(smiles):
    m = Chem.MolFromSmiles(smiles.replace("*", "[*]"))
    return m if m is not None else Chem.MolFromSmiles(smiles.replace("*", "C"))

def calc_desc(smiles):
    m = parse(smiles)
    return None if m is None else Descriptors.CalcMolDescriptors(m)

def morgan_fp(smiles, radius=2, nbits=2048):
    m = parse(smiles)
    if m is None: return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.array(gen.GetFingerprint(m), dtype=np.float32)

def maccs_fp(smiles):
    m = parse(smiles)
    if m is None: return None
    return np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)

def get_features(df):
    descs = df["smiles"].apply(calc_desc)
    Xd = pd.DataFrame(list(descs.values), columns=DESC_NAMES).replace([np.inf, -np.inf], np.nan)
    Xd = Xd.fillna(Xd.median()).replace([np.inf, -np.inf], np.nan)
    Xd = Xd.drop(columns=[c for c in Xd.columns if Xd[c].nunique() <= 1])
    for c in Xd.columns:
        lo, hi = Xd[c].quantile(0.001), Xd[c].quantile(0.999)
        Xd[c] = Xd[c].clip(lo, hi)
    Xd = Xd.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X = Xd.values.astype(np.float64)
    fp = np.stack(df["smiles"].apply(morgan_fp).values)
    mc = np.stack(df["smiles"].apply(maccs_fp).values)
    X = np.hstack([X, fp, mc])
    return np.nan_to_num(X)

POLY_FEATS = ["aromatic_ratio", "hetero_ratio", "ring_count", "ring_ratio", "nF", "nSi", "nS",
              "nN", "nO", "nCl", "rot_bonds", "frac_sp3", "donors", "acceptors",
              "polarizability", "logP", "n_conj_rings"]
def poly_feats_df(df):
    sys.path.insert(0, os.path.abspath("vault/figures"))
    from phase2_chemistry import polymer_features
    P = df["smiles"].apply(polymer_features)
    return pd.DataFrame(P.tolist())[POLY_FEATS].fillna(0).values.astype(np.float64)

def bench(df, label):
    y = df["target"].values.astype(np.float64)
    X = np.hstack([get_features(df), poly_feats_df(df)])
    assert np.isfinite(X).all(), "non-finite X"
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingRegressor(max_iter=250, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31,
                                      subsample=0.9, colsample_bytree=0.8, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                                    subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=0),
    }
    print(f"\n======== {label} (n={len(df)})  X={X.shape} ========")
    res = []
    kf = KFold(5, shuffle=True, random_state=42)
    for name, mdl in models.items():
        t0 = time.time(); oof = np.zeros(len(y))
        for tr, va in kf.split(X):
            mdl.fit(X[tr], y[tr]); oof[va] = mdl.predict(X[va])
        rmse = root_mean_squared_error(y, oof); tt = time.time() - t0
        res.append({"model": name, "rmse": round(rmse, 4), "seconds": round(tt, 1), "target": label})
        print(f"  {name:14s} RMSE={rmse:8.4f}  ({tt:.0f}s)")
    return res

allres = []
for tt, df in [("tg", train[train.target_type == "tg"]),
               ("egc", train[train.target_type == "egc"]),
               ("small5", train[train.target_type.isin(["egb", "eps", "nc", "ei", "eea"])])]:
    allres += bench(df, tt)

pd.DataFrame(allres).to_csv(os.path.join(OUT, "model_benchmark.csv"), index=False)
print("\nSaved model_benchmark.csv")
