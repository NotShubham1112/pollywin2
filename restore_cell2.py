import json

CELL2 = '''import os, sys, random, json, time, hashlib, pickle, warnings
warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
random.seed(SEED)
np.random.seed(SEED)

N_FOLDS = 5
MAX_ESTIMATORS = 3000          # Kaggle-scale budget; early stopping typically halts well before this
EARLY_STOPPING_ROUNDS = 150
EARLY_STOP_HOLDOUT_FRAC = 0.15

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def detect_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

GPU_AVAILABLE = detect_gpu()
LGB_PARAMS_EXTRA = {}
XGB_TREE_METHOD = "hist"
CB_TASK_TYPE = "CPU"  # CPU-only for bit-reproducible results across runs (CatBoost GPU is nondeterministic)

print("="*70)
print("SECTION 1 \u2014 CONFIGURATION")
print("="*70)
print(f"SEED = {SEED}")
print(f"N_FOLDS = {N_FOLDS}")
print(f"GPU available: {GPU_AVAILABLE}")
print(f"numpy version: {np.__version__}")
print(f"lightgbm version: {lgb.__version__}")
print(f"xgboost version: {xgb.__version__}")
print(f"catboost version: {cb.__version__}")
'''

def cell_source(nb, idx):
    return ''.join(nb['cells'][idx]['source'])

def set_cell(nb, idx, src):
    nb['cells'][idx]['source'] = [src]

p = r'd:\Parth\ploywin r2\polymer_prediction_notebook.ipynb'
nb = json.load(open(p, encoding='utf-8'))

# cell 2 must be the CONFIG cell
print('cell2 was:', repr(cell_source(nb, 2)[:60]))
set_cell(nb, 2, CELL2)
print('cell2 now len:', len(CELL2))

json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1)
print('saved', p)
