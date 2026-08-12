import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
GLOBAL_FOLDS = 5

ALPHA_SIB = {"eea": 0.025, "egb": 0.035, "egc": 0.000, "ei": 0.050, "eps": 0.080, "nc": 0.100, "tg": 0.000}
ALPHA_PHYS = {"eps": 0.20, "egb": 0.15}
PHYS_EPS_A = 1.040; PHYS_EPS_B = 0.616
PHYS_EPS_NC_RANGE = (1.55, 2.80)
PHYS_EGB_A = 1.022; PHYS_EGB_B = -0.108

NZX = r"D:\Parth\ploywin r2\vault\kernel-v17-sib-phys\out\blend_oof_test.npz"
TRAIN_CSV = r"D:\Parth\ploywin r2\official_dataset\train.csv"
TEST_CSV = r"D:\Parth\ploywin r2\official_dataset\test.csv"

# ---- load cache (P14 arms per row, aligned to train.csv/test.csv order) ----
z = np.load(NZX, allow_pickle=True)
oof_gbm, oof_mt = z["oof_gbm"], z["oof_mt"]
tgbm, tmt = z["test_gbm"], z["test_mt"]
t_all, y_all, g_all = z["t_all"], z["y_all"].astype(np.float64), z["g_all"]

tr = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
assert len(tr) == len(oof_gbm) and (tr["target_type"].values == np.array([s for s in t_all])).all()

# ---- P14 per-target Ridge blend (the shipped baseline) ----
def p14_blend_pred(gbm, mt, ttype, y_te=None, foldshare=True, idxs=None):
    """per-target Ridge(alpha=1.0) blend, GroupKFold(5) OOF / full-fit test."""
    preds = np.zeros(len(gbm))
    for t in TARGETS:
        m = ttype == t
        idxs = np.where(m)[0]
        Mx = np.column_stack([gbm[idxs], mt[idxs]])
        if fold:
            cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Mx, y_te, fold[idxs]))
            for tr, vk in cv:
                preds[idxs[vk]] = Ridge(alpha=1.0).fit(Mx[tr], y_te[vk]).predict(Mx[vk])
    return preds

def oof_p14(gbm, mt, ttype, yt, grp):
    preds = np.zeros(len(yt))
    for t in TARGETS:
        idxs = np.where(ttype == t)[0]
        Mx = np.column_stack([gbm[idxs], mt[idxs]])
        cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Mx, yt[idxs], grp[idxs]))
        for tr, vk in cv:
            preds[idxs[vk]] = Ridge(alpha=1.0).fit(Mx[tr], yt[idxs[tr]]).predict(Mx[vk])
    return preds

p14_oof = oof_p14(oof_gbm, oof_mt, t_all, y_all, g_all)
per_t = {t: r2_score(y_all[np.where(t_all == t)[0]], p14_oof[np.where(t_all == t)[0]]) for t in TARGETS}
print("P14 (reproduced) per-target R2:", {t: f"{perf[v]:.4f}" for v in per_t})
print("P14 mean:", f"{np.mean(list(per_t.values())):.4f}")