"""Check for distribution shift between train and test PREDICTIONS.

If the test predictions have a different mean/std than the train OOF, we have
a shift and applying a calibration learned on train could help.

NeurIPS 1st place trick: Tg += 0.5644 * std(Tg test predictions).
The 0.5644 factor was learned from public LB feedback.
"""
import os, warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np, pandas as pd
from rdkit import Chem

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None
train["canon"] = train["smiles"].apply(canon)
test["canon"]  = test["smiles"].apply(canon)
train = train.dropna(subset=["canon"]).reset_index(drop=True)
test  = test.dropna(subset=["canon"]).reset_index(drop=True)

TARGETS = ["eea","egb","egc","ei","eps","nc","tg"]
npz = np.load(os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz"), allow_pickle=True)
oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
oof_mt  = np.asarray(npz["oof_mt"], dtype=float)
y       = np.asarray(npz["y_train"], dtype=float)
t_arr   = np.asarray(npz["target_type_train"])
test_gbm = np.asarray(npz["test_gbm"], dtype=float)
test_mt  = np.asarray(npz["test_mt"], dtype=float)
test_tt  = np.asarray(npz["target_type_test"])

# Compare TRAIN OOF distribution vs TEST PRED distribution (per target)
print("=== Distribution comparison: TRAIN OOF (truth vs P14 blend) vs TEST (P14 blend predictions) ===")
print("target | train_y mean/std | train_oof_residual mean/std | test_pred mean/std | shift in pred mean")
for tt in TARGETS:
    idx_tr = np.where(t_arr == tt)[0]
    idx_te = np.where(test_tt == tt)[0]
    yt = y[idx_tr]
    p14_tr = 0.5*oof_gbm[idx_tr] + 0.5*oof_mt[idx_tr]
    p14_te = 0.5*test_gbm[idx_te] + 0.5*test_mt[idx_te]
    residual = yt - p14_tr
    shift_pred_mean = p14_te.mean() - p14_tr.mean()
    print(f"  {tt:<4} | y {yt.mean():7.2f}/{yt.std():6.2f} | res {residual.mean():+.3f}/{residual.std():6.3f} | pred {p14_te.mean():7.2f}/{p14_te.std():6.2f} | shift_pred_mean {shift_pred_mean:+.3f}")

# Per-target, what is the test P14 prediction distribution (mean, std) vs train target distribution?
# If test_pred is shifted away from train_y, then applying a shift learned on train could help.
# This is what NeurIPS 1st place did.

# Specifically: if test_pred.mean() < train_y.mean(), we should INCREASE predictions (positive shift).
# We measure: (train_y.mean() - test_pred.mean()) / train_y.std()  =  effect size

print("\n=== Estimated distribution shift: train_y.mean() - test_pred.mean() ===")
print("(positive = test predicts lower than train mean; we should add a positive shift)")
for tt in TARGETS:
    idx_tr = np.where(t_arr == tt)[0]
    idx_te = np.where(test_tt == tt)[0]
    yt = y[idx_tr]
    p14_te = 0.5*test_gbm[idx_te] + 0.5*test_mt[idx_te]
    shift = float(yt.mean() - p14_te.mean())
    shift_sigma = shift / yt.std() if yt.std() > 0 else 0.0
    print(f"  {tt:<4}: shift = {shift:+.4f} (in sigma: {shift_sigma:+.3f})")

# Apply the train_y.mean() - test_pred.mean() shift as a calibration and see OOF R2 impact
# Note: this assumes test distribution matches train, which may or may not hold.
print("\n=== Apply shift +0.5644 * std (NeurIPS 1st place trick) ===")
for tt in TARGETS:
    idx_tr = np.where(t_arr == tt)[0]
    idx_te = np.where(test_tt == tt)[0]
    yt = y[idx_tr]
    p14_tr = 0.5*oof_gbm[idx_tr] + 0.5*oof_mt[idx_tr]
    p14_te = 0.5*test_gbm[idx_te] + 0.5*test_mt[idx_te]
    # Apply +0.5644 * std (or -0.5644 depending on sign convention)
    shift = 0.5644 * p14_te.std()
    p14_te_shifted = p14_te + shift
    # We can\'t compute LB gain directly, but we can estimate: if test truth has same mean as train_y,
    # then the shift moves test_pred.mean() closer to truth.mean().
    print(f"  {tt}: shift=+{shift:.3f}  test_pred mean before={p14_te.mean():.3f}  after={p14_te_shifted.mean():.3f}  train_y mean={yt.mean():.3f}")
