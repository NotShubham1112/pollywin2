"""Analyze why we're stuck at 0.883 and what could break through.

P14 baseline: 0.883 public LB.
- Conservative sib Ridge nested-CV gain: +0.002 mean R^2 (ebg, ei, nc).
- v16 attempt (decoder arms): regressed on LB.

Hypotheses for breakthrough:
1. Per-target calibration shift (NeurIPS 1st place trick). Apply a learned multiplier to each target.
2. Snapshot ensemble of GNN seeds (42/999/2025) - but per postmortem, already tried, no gain.
3. Train an additional GNN architecture (e.g., AttentiveFP, GIN with virtual nodes) as a 3rd arm.
4. Use PI1M for semi-supervised pretraining of a regression head for the small targets.
5. Augment features with cross-target predictions: for each row, predict ALL 7 targets, use the
   predicted small-five as features for the big targets (and vice versa).
6. Multi-output Gaussian Process / NGBoost for calibrated uncertainty-aware blend.
7. Test-time augmentation: predict on multiple SMILES variants (non-canonical), median.

Test (7) is the cleanest, lowest-risk, fastest. Let's see if TTA can help.
"""
import os, warnings, sys
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

WORK = r"D:\Parth\ploywin r2"
train = pd.read_csv(os.path.join(WORK, "official_dataset", "train.csv"))
test  = pd.read_csv(os.path.join(WORK, "official_dataset", "test.csv"))

# Generate N=5 non-canonical SMILES per row and see if RDKit produces the same molecule
def n_random_smiles(s, n=5, seed=0):
    m = Chem.MolFromSmiles(s)
    if m is None: return []
    rng = np.random.default_rng(seed)
    out = set()
    for i in range(n*3):
        try:
            cs = Chem.MolToSmiles(m, canonical=False, doRandom=True, isomericSmiles=True)
            if cs not in out:
                out.add(cs)
                if len(out) >= n: break
        except: pass
    return list(out)

# Count how many train rows have distinct non-canonical SMILES (i.e., SMILES ambiguity)
n_unique = 0
total = 0
for i, s in enumerate(train["smiles"].head(200)):
    variants = n_random_smiles(s, n=5, seed=i)
    total += 1
    if len(variants) > 1:
        n_unique += 1
print(f"Of {total} train rows sampled: {n_unique} have >1 distinct non-canonical SMILES ({(100*n_unique/total):.1f}%)")

# TTA could only help if models are SMILES-variant sensitive (e.g., GNN with order-dependent message passing).
# For RDKit descriptors (order-invariant), TTA is a no-op. For Morgan FP (canonical by default), also a no-op.
# Only for GNN with non-canonical input does TTA matter.
print("\nTTA applicability: only GNNs benefit. P14's GNN (mt_gnn_v2) likely uses canonical SMILES.")
print("TTA on canonical SMILES = no effect. TTA on RDKit FP/desc = no effect.")
print("TTA on a non-canonical-input GNN would help, but we don't have one.")
