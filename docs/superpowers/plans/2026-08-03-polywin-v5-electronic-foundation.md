# PolyWin R2 — v5 Electronic Foundation Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the v5 pipeline in `build_pipeline_nb.py` so every base-model OOF is honest (fold-wise), Tg is isolated, an Electronic Foundation Network (EFN) with 10 aux physics heads and per-head target masking replaces the leaky `train_multitask` and the degenerate GNN, and a level-2 meta with reliability + cross-target OOF features produces the final predictions — all with per-model artifacts persisted to `WORK/`.

**Architecture:** All models share one global `GLOBAL_FOLDS`-fold canonical GroupKFold partition (`folds.csv` persisted, never regenerated). GBMs run per target as before. The EFN (encoder `1153→512→256→128`, 6 real electronic heads + 10 aux heads, per-target inverse-σ weighting, per-head presence masks) and a dedicated single-target `TgNN` (`256→128→64`) each train one model per fold and average test predictions. Level-1.5 Ridge stacks the target's own base-model OOFs; level-2 Ridge meta adds reliability features plus cross-target level-1.5 OOF features for the electronic cluster only (fold-safe because folds are global and canon-keyed). Tg receives/contributes no cross features.

**Tech Stack:** Python, `build_pipeline_nb.py` (nbformat generator), RDKit, scikit-learn, LightGBM, CatBoost, XGBoost, PyTorch (cu121 GPU bootstrap, CPU fallback).

## Global Constraints

- `GLOBAL_FOLDS = 10` for production, `5` when `SMOKE=True` (env `POLYWIN_SMOKE=1`). **Every model** (GBMs, EFN, tgnn, level-1.5, level-2) uses the identical global fold partition. When an existing `WORK/folds.csv` is found, load it and set `GLOBAL_FOLDS = max(fold)+1`; never regenerate.
- Fold persistence file: `WORK/folds.csv` (columns `canon`, `target_type`, `fold`).
- EFN width fixed: `512/256/128`. Do not increase (7406 rows).
- Aux tasks: the 10 Mol-derived scores in spec §4.3, defined for **all** train rows, standardized **per fold** (mean/std fit on fold-train rows only), drop any with near-zero global variance. Aux used at train time only.
- Missing target labels are **never imputed**; per-head presence masks only.
- Tg isolation: Tg excluded from the EFN trunk; `CROSS_MAP["tg"] = []` unconditionally; `BASE_MODELS_TG = ["lgb","cat","xgb","hgb","tgnn"]`.
- `BASE_MODELS_ELEC = ["lgb","cat","xgb","hgb","efn"]`.
- Ridge `alpha=10.0` for both level-1.5 and level-2 meta.
- GNN removed from the runtime path and from `BASE_MODELS`; code preserved in generator history / archived note.
- Loss: total = per-target inverse-σ weighted real MSE + `λ_aux·aux MSE` with `λ_aux = 0.3`.
- Submission uses level-2 `FINAL_TE`; physics bounds: egc/egb/ei ≥ 0, eps ≥ 1, nc ∈ [1,3], tg/eea unconstrained.
- Fig-08 (cross-target correlation) computed in-notebook from `Y` over shared canonical molecules — no dependence on `vault/figures`.
- Artifact persistence: `folds.csv`, `oof_lgb.parquet`, `oof_cat.parquet`, `oof_xgb.parquet`, `oof_hgb.parquet`, `oof_efn.parquet`, `oof_tgnn.parquet`, `l15_ridge.parquet`, `final_meta.parquet`.
- GPU bootstrap best-effort only: repair = `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121` → hard re-import (drop `torch`/`torch.*` from `sys.modules`) → re-probe. CPU fallback must always work.
- Smoke runs write to `vault/pipeline_out_smoke` (local only) so a 5-fold `folds.csv` never pollutes the production `vault/pipeline_out`.

---

### Task 1: GPU bootstrap + smoke config in cell 1

**Files:**
- Modify: `build_pipeline_nb.py:37-78` (the first `P("""import os, sys...""")` cell)

**Interfaces:**
- Consumes: nothing.
- Produces: `ensure_torch_cuda()` (returns a usable `torch` module), `get_torch_device()` (unchanged behavior), globals `SMOKE`, `GLOBAL_FOLDS`, `EFN_EPOCHS`, `TGNN_EPOCHS`, `WORK` (switches to `vault/pipeline_out_smoke` when smoke + local). All later tasks reference `GLOBAL_FOLDS`, `EFN_EPOCHS`, `TGNN_EPOCHS`, `SMOKE`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_nb.py`:

```python
"""Tests for the v5 notebook generator (build_pipeline_nb.py).

Each test rebuilds the notebook from the generator and asserts on the combined
source of all code cells. Run with `python tests/test_pipeline_nb.py` or pytest.
"""
import pathlib, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "AISEHack_Round2_Pipeline.ipynb"
GEN = REPO / "build_pipeline_nb.py"

def _build():
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True)
    nb = nbformat.read(str(NB), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    return code, md

def test_cell1_gpu_bootstrap():
    code, _ = _build()
    assert "def ensure_torch_cuda" in code
    assert "GLOBAL_FOLDS = 5 if SMOKE else 10" in code
    assert "pipeline_out_smoke" in code

if __name__ == "__main__":
    import traceback
    failed = 0
    for _n, _fn in list(globals().items()):
        if _n.startswith("test_") and callable(_fn):
            try:
                _fn(); print("PASS", _n)
            except Exception as _e:
                failed += 1
                print("FAIL", _n, "->", _e)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_cell1_gpu_bootstrap -> 'def ensure_torch_cuda' not in code` (assertion error).

- [ ] **Step 3: Implement the new cell 1**

Replace in `build_pipeline_nb.py` the entire existing first code cell (lines 37–78, from `P("""import os, sys, gc...` through the trailing `""")`) with:

```python
P("""import os, sys, gc, time, json, warnings, random
import subprocess, importlib.util
def ensure_pkg(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print("installing", pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", pkg])
for _p, _n in [("rdkit", "rdkit"), ("catboost", "catboost"),
               ("lightgbm", "lightgbm"), ("xgboost", "xgboost")]:
    ensure_pkg(_p, _n)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
np.random.seed(42); random.seed(42)
import torch

def ensure_torch_cuda():
    \"\"\"Best-effort repair of a broken torch/CUDA pairing; always falls back to CPU.\"\"\"
    import torch as _t0
    if not _t0.cuda.is_available():
        return _t0
    try:
        _x = _t0.zeros(1, device="cuda"); _x = _x + 1
        _t0.cuda.synchronize(); del _x
        return _t0
    except Exception as _e:
        print("CUDA probe failed:", str(_e)[:120], "-> attempting torch cu121 repair")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--index-url", "https://download.pytorch.org/whl/cu121",
                               "torch==2.2.2"])
    except Exception as _e:
        print("torch repair install failed:", str(_e)[:120])
    for _m in list(sys.modules):
        if _m == "torch" or _m.startswith("torch."):
            del sys.modules[_m]
    import torch
    if torch.cuda.is_available():
        try:
            _x = torch.zeros(1, device="cuda"); _x = _x + 1
            torch.cuda.synchronize(); del _x
            print("torch repaired -> CUDA OK:", torch.__version__)
            return torch
        except Exception as _e:
            print("CUDA still failing after repair -> CPU:", str(_e)[:120])
    else:
        print("no CUDA after repair -> CPU")
    return torch

torch = ensure_torch_cuda()

def get_torch_device():
    if torch.cuda.is_available():
        try:
            _t = torch.zeros(1, device="cuda"); _t = _t + 1
            torch.cuda.synchronize(); del _t
            return torch.device("cuda")
        except Exception as _e:
            print("CUDA probe failed -> using CPU:", str(_e)[:120])
    return torch.device("cpu")

ON_KAGGLE = os.path.exists("/kaggle")
if ON_KAGGLE:
    WORK = "/kaggle/working"
else:
    WORK = os.path.join("vault", "pipeline_out")
SMOKE = os.environ.get("POLYWIN_SMOKE", "0") == "1"
if SMOKE and not ON_KAGGLE:
    WORK = os.path.join("vault", "pipeline_out_smoke")
os.makedirs(WORK, exist_ok=True)
FIG = os.path.join(WORK, "figures"); os.makedirs(FIG, exist_ok=True)
GLOBAL_FOLDS = 5 if SMOKE else 10
EFN_EPOCHS = 15 if SMOKE else 40
TGNN_EPOCHS = 15 if SMOKE else 40
print("ON_KAGGLE =", ON_KAGGLE)
print("WORK =", WORK)
print("SMOKE =", SMOKE, "| GLOBAL_FOLDS =", GLOBAL_FOLDS)
print("torch =", torch.__version__, "| cuda =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_cell1_gpu_bootstrap` (first line of output; the placeholder line does not raise).

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: GPU bootstrap + smoke config in cell 1"
```

---

### Task 2: Global fold persistence (never regenerate)

**Files:**
- Modify: `build_pipeline_nb.py:143-150` (the GroupKFold block inside the canonicalization cell)

**Interfaces:**
- Consumes: `GLOBAL_FOLDS`, `WORK` (Task 1); `dedup` (defined earlier in same cell).
- Produces: global `folds` array (length `len(dedup)`, values 0..`GLOBAL_FOLDS-1`), `dedup["fold"]` column, `WORK/folds.csv`. Later tasks (retrieval, get_splits, EFN, tgnn, stacks) read `folds`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_cell3_folds_persistence():
    code, _ = _build()
    assert "FOLDS_CSV" in code
    assert "loaded folds.csv" in code
    assert "dedup[\"fold\"] = folds" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_cell3_folds_persistence -> 'FOLDS_CSV' not in code`

- [ ] **Step 3: Implement**

Replace in `build_pipeline_nb.py`:

```python
# ---- GroupKFold on canonical polymer ----
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=10)
folds = np.zeros(len(dedup), dtype=int)
for i, (_, va) in enumerate(gkf.split(dedup, groups=dedup["canon"])):
    folds[va] = i
dedup["fold"] = folds
print(dedup.groupby(["target_type","fold"]).size().unstack(fill_value=0).to_string())
```

with:

```python
# ---- GroupKFold on canonical polymer (persisted; never regenerate) ----
from sklearn.model_selection import GroupKFold
FOLDS_CSV = os.path.join(WORK, "folds.csv")
if os.path.exists(FOLDS_CSV):
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy()
    GLOBAL_FOLDS = int(folds.max()) + 1
    print("loaded folds.csv with", len(folds), "rows -> GLOBAL_FOLDS =", GLOBAL_FOLDS)
else:
    gkf = GroupKFold(n_splits=GLOBAL_FOLDS)
    folds = np.zeros(len(dedup), dtype=int)
    for i, (_, va) in enumerate(gkf.split(dedup, groups=dedup["canon"])):
        folds[va] = i
    pd.DataFrame({"canon": dedup["canon"].values,
                  "target_type": dedup["target_type"].values,
                  "fold": folds}).to_csv(FOLDS_CSV, index=False)
    print("wrote folds.csv", FOLDS_CSV)
dedup["fold"] = folds
print(dedup.groupby(["target_type","fold"]).size().unstack(fill_value=0).to_string())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_cell3_folds_persistence`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: persist global folds, never regenerate"
```

---

### Task 3: Auxiliary physics scores cell

**Files:**
- Modify: `build_pipeline_nb.py` — insert a new markdown + code cell between the cleaning cell (ends line 276) and the retrieval markdown (line 278)

**Interfaces:**
- Consumes: `parse_mol` (cell 3), `Chem`, `rdMolDescriptors`, `time`; `dedup["smiles"]`, `test["smiles"]`.
- Produces: `AUX_TASKS` (list of kept task names), `aux_tr` (7406×k), `aux_te` (4940×k). Task 5 (EFN) consumes `aux_tr`, `aux_te`, `AUX_TASKS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_aux_physics_cell():
    code, _ = _build()
    assert "AUX_TASKS" in code
    assert "def aux_physics_scores" in code
    assert "Chem.AtomHasConjugatedBond" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_aux_physics_cell -> 'AUX_TASKS' not in code`

- [ ] **Step 3: Implement**

Insert after the `Xte.to_pickle(...)` block (line 274–276) and before the `M("""## Layer 3 — Retrieval Memory...""")` cell:

```python
M("""## Auxiliary physics tasks (10 scores)

Chemistry-derived scores computed from RDKit Mol objects for **all** training rows (no missing
labels). Used at train time only to give the shared EFN encoder dense supervision. They are NOT
descriptor columns, so the model cannot copy them trivially.""")
P("""AUX_TASKS = ["aromaticity_score","conjugation_score","sulfur_score","electronegativity_score",
              "polarity_score","ring_density_score","flexibility_score","halogen_density",
              "hbond_capacity","heteroatom_fraction"]

PAULING = {"C":2.55,"N":3.04,"O":3.44,"F":3.98,"S":2.58,"Si":1.90,"Cl":3.16,"P":2.19,"Br":2.96,"I":2.66}

def aux_physics_scores(smiles_list):
    rows = []
    for smi in smiles_list:
        m = parse_mol(smi)
        if m is None:
            rows.append(np.zeros(len(AUX_TASKS), dtype=np.float32)); continue
        atoms = list(m.GetAtoms())
        heavy = max(m.GetNumHeavyAtoms(), 1)
        arom = sum(1 for a in atoms if a.GetIsAromatic())
        conj = sum(1 for a in atoms if Chem.AtomHasConjugatedBond(a))
        nS = sum(1 for a in atoms if a.GetSymbol()=="S")
        en = np.mean([PAULING.get(a.GetSymbol(), 2.5) for a in atoms])
        tpsa = rdMolDescriptors.CalcTPSA(m)
        ri = m.GetRingInfo()
        ring_atoms = len({a for ring in ri.AtomRings() for a in ring})
        rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        nF = sum(1 for a in atoms if a.GetSymbol()=="F")
        nCl = sum(1 for a in atoms if a.GetSymbol()=="Cl")
        nBr = sum(1 for a in atoms if a.GetSymbol()=="Br")
        nI = sum(1 for a in atoms if a.GetSymbol()=="I")
        hbd = rdMolDescriptors.CalcNumHBD(m); hba = rdMolDescriptors.CalcNumHBA(m)
        nC = sum(1 for a in atoms if a.GetSymbol()=="C")
        nHeavy = m.GetNumHeavyAtoms()
        rows.append(np.array([
            arom/heavy, conj/heavy, nS/heavy, en, tpsa/heavy,
            ring_atoms/heavy, rot/heavy, (nF+nCl+nBr+nI)/heavy,
            (hbd+hba)/heavy, (nHeavy-nC)/heavy,
        ], dtype=np.float32))
    return np.stack(rows)

print("Computing auxiliary physics scores...")
t0 = time.time()
aux_tr = aux_physics_scores(dedup["smiles"].tolist())
aux_te = aux_physics_scores(test["smiles"].tolist())
aux_all = np.vstack([aux_tr, aux_te])
keep_aux = [j for j in range(aux_all.shape[1]) if np.nanstd(aux_all[:, j]) > 1e-8]
aux_all = aux_all[:, keep_aux]
AUX_TASKS = [AUX_TASKS[j] for j in keep_aux]
aux_tr = aux_all[:len(dedup)]; aux_te = aux_all[len(dedup):]
print(f"aux scores {aux_tr.shape} {aux_te.shape}; kept {len(AUX_TASKS)} tasks")
for j, name in enumerate(AUX_TASKS):
    v = aux_tr[:, j]
    print(f"  {name:24s} mean={np.nanmean(v):.4f} std={np.nanstd(v):.4f}")""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_aux_physics_cell`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: add 10 aux physics scores (all rows, Mol-derived)"
```

---

### Task 4: Validation harness — electronic targets + artifact helper

**Files:**
- Modify: `build_pipeline_nb.py:358-387` (the validation-harness `P(...)` cell)

**Interfaces:**
- Consumes: `TARGETS`, `Y`, `dedup`, `test`, `WORK`, `rmse_metric`.
- Produces: `ELECTRONIC_TARGETS` (list), `save_oof_artifact(name, oof_map, te_map)`. Tasks 5, 6, 9 (EFN/tgnn/GBM cells) call `save_oof_artifact`; Task 8's stacking uses `ELECTRONIC_TARGETS` implicitly via store keys; `oof_*.parquet` names derive from `name`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_harness_electronic_targets_and_artifact_helper():
    code, _ = _build()
    assert "ELECTRONIC_TARGETS = [\"egc\",\"egb\",\"eps\",\"nc\",\"ei\",\"eea\"]" in code
    assert "def save_oof_artifact" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_harness_electronic_targets_and_artifact_helper -> 'ELECTRONIC_TARGETS ...' not in code`

- [ ] **Step 3: Implement**

Replace in `build_pipeline_nb.py`:

```python
# sanity: how many test rows per type
print(test["target_type"].value_counts().to_string())""")
```

with:

```python
ELECTRONIC_TARGETS = ["egc","egb","eps","nc","ei","eea"]

def save_oof_artifact(name, oof_map, te_map):
    \"\"\"Persist per-target OOF + test predictions for one base model as parquet.\"\"\"
    parts = []
    for tt in TARGETS:
        m_tr = (dedup["target_type"] == tt).values
        m_te = (test["target_type"] == tt).values
        if tt in oof_map:
            parts.append(pd.DataFrame({
                "target": tt,
                "dedup_index": np.where(m_tr)[0],
                "oof": np.asarray(oof_map[tt]),
                "test_pred": np.asarray(te_map[tt])[m_te],
            }))
    if not parts:
        return
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(WORK, f"oof_{name}.parquet"), index=False)
    print("saved oof_" + name + ".parquet")

# sanity: how many test rows per type
print(test["target_type"].value_counts().to_string())""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_harness_electronic_targets_and_artifact_helper`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: ELECTRONIC_TARGETS + save_oof_artifact helper"
```

---

### Task 5: Electronic Foundation Network (replaces `train_multitask`)

**Files:**
- Modify: `build_pipeline_nb.py:24-25` (intro markdown layer list) and `build_pipeline_nb.py:442-522` (the entire Layer 5 mtnn markdown + code cells)

**Interfaces:**
- Consumes: `ELECTRONIC_TARGETS`, `AUX_TASKS`, `aux_tr`, `aux_te`, `Y`, `dedup`, `folds`, `GLOBAL_FOLDS`, `EFN_EPOCHS`, `Xtr`, `Xte`, `save_oof_artifact`, `record`, `get_torch_device`.
- Produces: `EFN` class, `_fit_efn_fold(...)`, `efn_fit_predict(...)`, global `Xtr_s`, `Xte_s` (standardized, shared with tgnn), `real_y` (n×6 NaN-masked electronic target matrix), `efn_oof`, `efn_te`, OOF records under keys `("efn_" + tt, tt)`, and `WORK/oof_efn.parquet`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_efn_replaces_mtnn():
    code, _ = _build()
    assert "class EFN" in code
    assert "def efn_fit_predict" in code
    assert "save_oof_artifact(\"efn\"" in code
    assert "train_multitask" not in code
    assert "MultiTaskNN" not in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_efn_replaces_mtnn -> 'class EFN' not in code`

- [ ] **Step 3: Implement**

Edit A — intro markdown. Replace in `build_pipeline_nb.py`:

```python
5. **Multi-task NN** — shared trunk + per-target heads (PyTorch, GPU)
6. **GNN branch** — pure-PyTorch GIN message passing on the polymer graph
```

with:

```python
5. **Electronic Foundation Network** — shared encoder + 6 real electronic heads + 10 aux physics heads (PyTorch, GPU)
6. **Tg isolation** — dedicated single-target Tg NN (no shared trunk, no cross features)
```

Edit B — replace the entire Layer 5 block (markdown cell `M("""## Layer 5 — Multi-task NN...` plus the `P("""import torch.nn as nn...` code cell through `torch.save(mt_model.state_dict(), os.path.join(WORK, "mtnn.pt"))""")`) with:

```python
M("""## Layer 5 — Electronic Foundation Network (EFN)

Shared encoder `1153 -> 512 -> 256 -> 128` (BN + SiLU + Dropout 0.3) produces a polymer-state
vector. 6 real electronic heads (egc, egb, eps, nc, ei, eea) + 10 aux physics heads supervise the
encoder on **all** rows. Per-target inverse-sigma MSE weighting + per-head presence masking
(missing labels are never imputed). **Tg is excluded entirely from this trunk.** Honest OOF: one
model per global fold; test predictions averaged across fold models.""")
P("""import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class EFN(nn.Module):
    def __init__(self, n_in, hidden=512, latent=128, n_aux=len(AUX_TASKS)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, latent), nn.SiLU(),
        )
        self.real_heads = nn.ModuleDict({t: nn.Linear(latent, 1) for t in ELECTRONIC_TARGETS})
        self.aux_heads = nn.ModuleList([nn.Linear(latent, 1) for _ in range(n_aux)])
    def forward(self, x):
        z = self.enc(x)
        real = {t: h(z) for t, h in self.real_heads.items()}
        aux = [h(z) for h in self.aux_heads]
        return real, aux

def _fit_efn_fold(tr_idx, X_s, real_y, aux_all, aux_w, epochs, bs, lr, wd, lam_aux, dev):
    torch.manual_seed(42)
    model = EFN(X_s.shape[1], n_aux=len(AUX_TASKS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X_s[tr_idx], dtype=torch.float32, device=dev)
    real_t = torch.tensor(real_y[tr_idx], dtype=torch.float32, device=dev)
    mu = np.nanmean(aux_all[tr_idx], axis=0)
    sd = np.nanstd(aux_all[tr_idx], axis=0); sd[sd < 1e-8] = 1.0
    aux_t = torch.tensor(np.clip((aux_all[tr_idx] - mu) / sd, -5.0, 5.0), dtype=torch.float32, device=dev)
    sig = {}
    for j, t in enumerate(ELECTRONIC_TARGETS):
        v = real_t[:, j].cpu().numpy(); v = v[~np.isnan(v)]
        sig[t] = float(np.std(v)) if len(v) > 1 and np.std(v) > 1e-6 else 1.0
    n = len(tr_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            bi = perm[i:i+bs]
            opt.zero_grad()
            rp, ap = model(Xt[bi])
            rb, ab = real_t[bi], aux_t[bi]
            loss = torch.tensor(0.0, device=dev); n_real = 0
            for j, t in enumerate(ELECTRONIC_TARGETS):
                mm = ~torch.isnan(rb[:, j])
                if mm.any():
                    loss = loss + (1.0/sig[t]) * F.mse_loss(rp[t].squeeze()[mm], rb[mm, j]); n_real += 1
            if n_real > 0:
                loss = loss / n_real
            aux_loss = torch.tensor(0.0, device=dev)
            for j, h in enumerate(model.aux_heads):
                aux_loss = aux_loss + aux_w[j] * F.mse_loss(ap[j].squeeze(), ab[:, j])
            aux_loss = aux_loss / max(len(model.aux_heads), 1)
            loss = loss + lam_aux * aux_loss
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 10 == 0:
            print(f"    efn ep {ep+1}/{epochs} loss {tot/max(nb,1):.4f}")
    return model

def efn_fit_predict(Xtr_s, Xte_s, real_y, aux_tr, aux_te, dedup_, folds, epochs, bs=256,
                    lr=1e-3, wd=1e-4, lam_aux=0.3):
    dev = get_torch_device()
    aux_w = 1.0 / np.maximum(np.nanstd(aux_tr, axis=0), 1e-6)
    tgt_sub = {t: np.where((dedup_["target_type"] == t).values)[0] for t in ELECTRONIC_TARGETS}
    pos_in_sub = {}
    for t, idx in tgt_sub.items():
        p = np.full(len(dedup_), -1, dtype=int); p[idx] = np.arange(len(idx)); pos_in_sub[t] = p
    oof = {t: np.full(len(idx), np.nan) for t, idx in tgt_sub.items()}
    te = {t: np.zeros(len(Xte_s)) for t in ELECTRONIC_TARGETS}
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=dev)
    for f in range(GLOBAL_FOLDS):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        model = _fit_efn_fold(tr_idx, Xtr_s, real_y, aux_tr, aux_w, epochs, bs, lr, wd, lam_aux, dev)
        model.eval()
        with torch.no_grad():
            r_te, _ = model(Xte_t)
            for j, t in enumerate(ELECTRONIC_TARGETS):
                te[t] += r_te[t].cpu().numpy().ravel() / GLOBAL_FOLDS
            r_va, _ = model(torch.tensor(Xtr_s[va_idx], dtype=torch.float32, device=dev))
            for j, t in enumerate(ELECTRONIC_TARGETS):
                m_ok = ~np.isnan(real_y[va_idx, j])
                if m_ok.any():
                    oof[t][pos_in_sub[t][va_idx[m_ok]]] = r_va[t].cpu().numpy().ravel()[m_ok]
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, te

# ---- standardize inputs for NN (global, shared by EFN + tgnn) ----
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xs = sc.fit_transform(pd.concat([Xtr, Xte], axis=0).values)
Xtr_s, Xte_s = Xs[:len(dedup)], Xs[len(dedup):]

# ---- real target matrix (NaN where a target is absent for a row) ----
real_y = np.full((len(dedup), len(ELECTRONIC_TARGETS)), np.nan)
for j, t in enumerate(ELECTRONIC_TARGETS):
    mm = (dedup["target_type"] == t).values
    real_y[mm, j] = Y[mm]

print("Training Electronic Foundation Network...")
t0 = time.time()
efn_oof, efn_te = efn_fit_predict(Xtr_s, Xte_s, real_y, aux_tr, aux_te, dedup, folds,
                                  epochs=EFN_EPOCHS)
print(f"EFN done in {time.time()-t0:.0f}s")
for tt in ELECTRONIC_TARGETS:
    m = (dedup["target_type"] == tt).values
    r = record("efn_" + tt, tt, efn_oof[tt], efn_te[tt])
    print(f"  efn {tt}: RMSE={r:.4f}")
save_oof_artifact("efn", efn_oof, efn_te)""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_efn_replaces_mtnn`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: EFN with aux heads + per-head masks replaces leaky mtnn"
```

---

### Task 6: Tg isolation — dedicated `TgNN`

**Files:**
- Modify: `build_pipeline_nb.py` — insert a new markdown + code cell immediately after the EFN cell (after `save_oof_artifact("efn", efn_oof, efn_te)""")` and before the Layer 6 GNN markdown

**Interfaces:**
- Consumes: `Xtr_s`, `Xte_s`, `Y`, `dedup`, `folds`, `GLOBAL_FOLDS`, `TGNN_EPOCHS`, `nn`, `F`, `save_oof_artifact`, `record`.
- Produces: `TgNN` class, `tgnn_fit_predict(...)`, `tg_oof`, `tg_te`, record `("tgnn_tg", "tg")`, `WORK/oof_tgnn.parquet`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_tgnn_cell():
    code, _ = _build()
    assert "class TgNN" in code
    assert "def tgnn_fit_predict" in code
    assert "save_oof_artifact(\"tgnn\"" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_tgnn_cell -> 'class TgNN' not in code`

- [ ] **Step 3: Implement**

Insert after the EFN cell and before `M("""## Layer 6 — GNN branch...`)`:

```python
M("""## Layer 5b — Tg isolation (dedicated single-target NN)

Tg is statistically disconnected from the electronic targets (shared-polymer overlap < 5%), so it
gets its own small MLP `256 -> 128 -> 64` and its own stack. No shared trunk, no cross-target
features to or from tg.""")
P("""class TgNN(nn.Module):
    def __init__(self, n_in, hidden=256, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, latent), nn.SiLU(),
            nn.Linear(latent, 1),
        )
    def forward(self, x):
        return self.net(x)

def tgnn_fit_predict(Xtr_s, Xte_s, Y, dedup_, folds, epochs, bs=128, lr=1e-3, wd=1e-4):
    dev = get_torch_device()
    m = (dedup_["target_type"] == "tg").values
    idx = np.where(m)[0]
    oof = np.full(m.sum(), np.nan)
    te_pred = np.zeros(len(Xte_s))
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=dev)
    for f in range(GLOBAL_FOLDS):
        tr_l = np.where(folds[idx] != f)[0]; va_l = np.where(folds[idx] == f)[0]
        tr_idx, va_idx = idx[tr_l], idx[va_l]
        torch.manual_seed(42)
        model = TgNN(Xtr_s.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xt = torch.tensor(Xtr_s[tr_idx], dtype=torch.float32, device=dev)
        yt = torch.tensor(Y[tr_idx], dtype=torch.float32, device=dev).view(-1, 1)
        n = len(tr_idx)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                bi = perm[i:i+bs]
                opt.zero_grad()
                loss = F.mse_loss(model(Xt[bi]), yt[bi])
                loss.backward(); opt.step()
            sched.step()
            if (ep+1) % 10 == 0:
                print(f"    tgnn ep {ep+1}/{epochs} loss {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            oof[va_l] = model(torch.tensor(Xtr_s[va_idx], dtype=torch.float32, device=dev)).cpu().numpy().ravel()
            te_pred += model(Xte_t).cpu().numpy().ravel() / GLOBAL_FOLDS
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model, oof, te_pred

print("Training Tg NN (isolated)...")
t0 = time.time()
tg_model, tg_oof, tg_te = tgnn_fit_predict(Xtr_s, Xte_s, Y, dedup, folds, epochs=TGNN_EPOCHS)
print(f"TgNN done in {time.time()-t0:.0f}s")
r = record("tgnn_tg", "tg", tg_oof, tg_te)
print(f"  tgnn tg: RMSE={r:.4f}")
save_oof_artifact("tgnn", {"tg": tg_oof}, {"tg": tg_te})""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_tgnn_cell`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: dedicated single-target TgNN (tg isolation)"
```

---

### Task 7: Remove GNN from runtime path

> **Controller note (Task 6 review, applied):** The Layer 6 heading was renumbered in Task 6 — the TgNN cell is now `## Layer 6 — Tg isolation (dedicated single-target TgNN)` and the GNN cell is now `## Layer 7 — GNN branch (pure-PyTorch GIN message passing)`. This task's replacement anchor is therefore `M("""## Layer 7 — GNN branch...`)` and the line range is refreshed below. This deviation was user-approved.

**Files:**
- Modify: `build_pipeline_nb.py` (the entire Layer 7 GNN markdown + code cells)

**Interfaces:**
- Consumes: nothing new; removes `PolymerGNN`, `GINConv`, `train_gnn`, `gnn_model`, and the `gnn.pt` save.
- Produces: an archived note cell. Downstream tasks (8, 10, 11) must not reference `"gnn"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_gnn_removed():
    code, _ = _build()
    assert "class GINConv" not in code
    assert "PolymerGNN" not in code
    assert "GNN branch archived" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_gnn_removed -> 'GNN branch archived' not in code`

- [ ] **Step 3: Implement**

Replace in `build_pipeline_nb.py` the entire Layer 7 block — the `M("""## Layer 7 — GNN branch...`)` markdown cell and the full `P("""class GINConv...`)` code cell ending at `torch.save(gnn_model.state_dict(), os.path.join(WORK, "gnn.pt"))""")` — with:

```python
M("""## Layer 7 — GNN (archived, not in runtime path)

The pure-PyTorch GIN branch (v4) degenerated on Kaggle's non-GPU runtime (RMSE 158–324). It is
removed from `BASE_MODELS` and the runtime path. The v4 implementation is preserved in this
generator's git history for future experiments.""")
P("""# GNN branch archived in v5 (see spec section 4.8); code kept in build_pipeline_nb.py git history.
print("GNN branch archived in v5 (see spec 4.8).")""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_gnn_removed`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: remove GNN from runtime path"
```

---

### Task 8: Level-1.5 Ridge stack + level-2 meta (reliability + cross-target) + persistence

**Files:**
- Modify: `build_pipeline_nb.py:699-761` (the entire Layer 9 stacking markdown + code cells)

**Interfaces:**
- Consumes: `oof_store`, `test_store`, `get_splits`, `TARGETS`, `Y`, `dedup`, `test`, `WORK`, `rmse_metric`, `LEADERBOARD` (unchanged, still produced by the GBM cell), records from Tasks 5/6.
- Produces: `BASE_MODELS_ELEC`, `BASE_MODELS_TG`, `base_models_for(tt)`, `store_key(b, tt)` (unified `(f"{b}_{tt}", tt)` for all models), `L15_OOF`, `L15_TE`, `CROSS_MAP`, `reliability_features(...)`, `cross_oof_features(...)`, `cross_te_features(...)`, `FINAL_OOF`, `FINAL_TE`, `WORK/l15_ridge.parquet`, `WORK/final_meta.parquet`, `WORK/final_leaderboard.csv`. Task 10 (figures) and Task 11 (submission) consume `FINAL_OOF`/`FINAL_TE`/`L15_OOF`/`base_models_for`/`store_key`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_stacking_levels():
    code, _ = _build()
    assert "BASE_MODELS_ELEC" in code
    assert "BASE_MODELS_TG" in code
    assert "CROSS_MAP" in code
    assert "FINAL_OOF" in code
    assert "l15_ridge.parquet" in code
    assert "final_meta.parquet" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_stacking_levels -> 'BASE_MODELS_ELEC' not in code`

- [ ] **Step 3: Implement**

Replace the entire existing stacking block (`M("""## Layer 9 — Stacking (Ridge / ElasticNet / CatBoost meta-model)...`)` through the `pd.DataFrame(summary).to_csv(os.path.join(WORK, "final_leaderboard.csv"), index=False)""")`) with:

```python
M("""## Layer 9 — Stacking (level-1.5 Ridge + level-2 meta with reliability + cross-target features)

Level-1 base models per target: electronic = `lgb, cat, xgb, hgb, efn`; tg = `lgb, cat, xgb, hgb, tgnn`.
Level-1.5: per-target Ridge stack on the target's own base-model OOFs (as v4).
Level-2 (electronic cluster only): per-target Ridge meta on own base OOFs + reliability features
(mean/std/max/min disagreement) + cross-target level-1.5 stack OOFs for correlated targets
(fold-safe because all targets share one global canonical fold partition). Tg gets no cross features.
Level-2 output = final predictions.""")
P("""from sklearn.linear_model import Ridge

BASE_MODELS_ELEC = ["lgb","cat","xgb","hgb","efn"]
BASE_MODELS_TG = ["lgb","cat","xgb","hgb","tgnn"]

def base_models_for(tt):
    return BASE_MODELS_TG if tt == "tg" else BASE_MODELS_ELEC

def store_key(b, tt):
    return (b + "_" + tt, tt)

def build_stack_features(oof_store, tt, models):
    feats, cols = [], []
    for b in models:
        k = store_key(b, tt)
        if k in oof_store:
            feats.append(oof_store[k]); cols.append(k)
    if len(feats) == 0:
        return None, None
    return np.column_stack(feats), cols

# ---- level 1.5: per-target Ridge on own base OOFs ----
L15_OOF = {}; L15_TE = {}
print("Level-1.5 Ridge stack (per target, own base OOFs)...")
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    m, idx, splits = get_splits(tt)
    Z, cols = build_stack_features(oof_store, tt, base_models_for(tt))
    if Z is None:
        print(f"  {tt}: no base features"); continue
    Zte = np.column_stack([test_store[c] for c in cols])
    pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
    oof = np.zeros(m.sum()); te_pred = np.zeros(len(Zte))
    for tr, va in splits:
        tr_l, va_l = pos[tr], pos[va]
        sr = StandardScaler().fit(Z[tr_l]); Ztr_s = sr.transform(Z[tr_l]); Zva_s = sr.transform(Z[va_l])
        meta = Ridge(alpha=10.0); meta.fit(Ztr_s, Y[idx][tr_l])
        oof[va_l] = meta.predict(Zva_s)
        te_pred += meta.predict(sr.transform(Zte)) / len(splits)
    L15_OOF[tt] = oof; L15_TE[tt] = te_pred
    r = rmse_metric(Y[m], oof)
    print(f"  l15 {tt}: RMSE={r:.4f}  (cols={cols})")

# ---- level 2: reliability + cross-target OOF features ----
CROSS_MAP = {
    "eps": ["nc","egc","egb","eea"],
    "nc": ["eps","egb","egc","ei"],
    "egc": ["egb","eea","nc","eps","ei"],
    "egb": ["egc","nc","eea","eps","ei"],
    "ei": ["egc","egb","nc"],
    "eea": ["egc","egb","eps"],
    "tg": [],
}

def reliability_features(tt, models):
    Z, cols = build_stack_features(oof_store, tt, models)
    if Z is None:
        return None, None
    feats = np.column_stack([Z.mean(1), Z.std(1), Z.max(1), Z.min(1)])
    return feats, ["rel_mean", "rel_std", "rel_max", "rel_min"]

def cross_oof_features(tt):
    feats, cols = [], []
    m_tt = (dedup["target_type"] == tt).values
    for ct in CROSS_MAP[tt]:
        m_ct = (dedup["target_type"] == ct).values
        c2o = dict(zip(dedup.loc[m_ct, "canon"].values, L15_OOF[ct]))
        vals = np.array([c2o.get(c, np.nan) for c in dedup.loc[m_tt, "canon"].values], dtype=np.float32)
        miss = np.isnan(vals).astype(np.float32)
        vals = np.nan_to_num(vals, nan=float(np.nanmean(L15_OOF[ct])))
        feats += [vals, miss]; cols += [f"cross_{ct}", f"cross_{ct}_miss"]
    if not feats:
        return None, None
    return np.column_stack(feats), cols

def cross_te_features(tt):
    feats, cols = [], []
    for ct in CROSS_MAP[tt]:
        feats.append(np.asarray(L15_TE[ct], dtype=np.float32))
        cols.append(f"cross_{ct}")
    if not feats:
        return None, None
    return np.column_stack(feats), cols

FINAL_OOF = {}; FINAL_TE = {}
print("\\nLevel-2 meta (own base + reliability + cross-target OOF)...")
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    m, idx, splits = get_splits(tt)
    Z1, c1 = build_stack_features(oof_store, tt, base_models_for(tt))
    if Z1 is None:
        print(f"  {tt}: no base features"); continue
    Zrel, crel = reliability_features(tt, base_models_for(tt))
    Zcr, ccr = cross_oof_features(tt)
    Z2 = np.column_stack([Z1, Zrel] + ([Zcr] if Zcr is not None else []))
    cols = c1 + crel + (ccr or [])
    Zte1 = np.column_stack([test_store[c] for c in c1])
    Zte_rel = np.column_stack([Zte1.mean(1), Zte1.std(1), Zte1.max(1), Zte1.min(1)])
    Zte_cr, _ = cross_te_features(tt)
    Zte2 = np.column_stack([Zte1, Zte_rel] + ([Zte_cr] if Zte_cr is not None else []))
    pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
    oof = np.zeros(m.sum()); te_pred = np.zeros(len(Zte2))
    for tr, va in splits:
        tr_l, va_l = pos[tr], pos[va]
        sr = StandardScaler().fit(Z2[tr_l]); Z2tr = sr.transform(Z2[tr_l]); Z2va = sr.transform(Z2[va_l])
        meta = Ridge(alpha=10.0); meta.fit(Z2tr, Y[idx][tr_l])
        oof[va_l] = meta.predict(Z2va)
        te_pred += meta.predict(sr.transform(Zte2)) / len(splits)
    FINAL_OOF[tt] = oof; FINAL_TE[tt] = te_pred
    r = rmse_metric(Y[m], oof)
    print(f"  final {tt}: RMSE={r:.4f}  (n_feats={len(cols)})")

# ---- persistence ----
l15_parts, fin_parts = [], []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    m_te = (test["target_type"] == tt).values
    if tt in L15_OOF:
        l15_parts.append(pd.DataFrame({"target": tt, "dedup_index": np.where(m)[0],
                                       "l15_oof": L15_OOF[tt], "l15_test": np.asarray(L15_TE[tt])[m_te]}))
    if tt in FINAL_OOF:
        fin_parts.append(pd.DataFrame({"target": tt, "dedup_index": np.where(m)[0],
                                       "final_oof": FINAL_OOF[tt], "final_test": np.asarray(FINAL_TE[tt])[m_te]}))
pd.concat(l15_parts).to_parquet(os.path.join(WORK, "l15_ridge.parquet"), index=False)
pd.concat(fin_parts).to_parquet(os.path.join(WORK, "final_meta.parquet"), index=False)
print("saved l15_ridge.parquet, final_meta.parquet")

# ---- final per-target RMSE summary ----
print("\\n==== FINAL LEADERBOARD (honest OOF RMSE) ====")
summary = []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    row = {"target": tt}
    for b in base_models_for(tt):
        k = store_key(b, tt)
        if k in oof_store:
            row[b] = round(rmse_metric(Y[m], oof_store[k]), 4)
    if tt in L15_OOF: row["l15"] = round(rmse_metric(Y[m], L15_OOF[tt]), 4)
    if tt in FINAL_OOF: row["final"] = round(rmse_metric(Y[m], FINAL_OOF[tt]), 4)
    summary.append(row)
    print(row)
pd.DataFrame(summary).to_csv(os.path.join(WORK, "final_leaderboard.csv"), index=False)""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_stacking_levels`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: level-1.5 stack + level-2 meta with reliability and cross-target OOF features"
```

---

### Task 9: GBM cell — collect + persist per-model artifacts

**Files:**
- Modify: `build_pipeline_nb.py:426-440` (the GBM training loop, inside the Layer 4 code cell)

**Interfaces:**
- Consumes: `record`, `save_oof_artifact` (Task 4), `gbm_fit_predict`, model factories (unchanged).
- Produces: `model_oof`/`model_te` dicts; `WORK/oof_lgb.parquet`, `oof_cat.parquet`, `oof_xgb.parquet`, `oof_hgb.parquet`; `LEADERBOARD` (unchanged contract for Task 8/11).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_gbm_artifact_save():
    code, _ = _build()
    assert "model_oof" in code
    assert "save_oof_artifact(name, model_oof[name], model_te[name])" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_gbm_artifact_save -> 'model_oof' not in code`

- [ ] **Step 3: Implement**

Replace in `build_pipeline_nb.py`:

```python
LEADERBOARD = {}
print("Training GBM experts...")
for tt in TARGETS:
    m, idx, splits = get_splits(tt)
    leader = {}
    for name, mk in [("lgb", make_lgb), ("cat", make_cat), ("xgb", make_xgb), ("hgb", make_hgb)]:
        t0 = time.time()
        oof, tep = gbm_fit_predict(tt, mk, Xtr, Xte)
        r = record(name + "_" + tt, tt, oof, tep)
        leader[name] = r
        print(f"  {tt} {name}: RMSE={r:.4f} ({time.time()-t0:.0f}s)")
    LEADERBOARD[tt] = leader
    best = min(leader, key=leader.get)
    print(f"  -> best for {tt}: {best} RMSE={leader[best]:.4f}")
pd.DataFrame(LEADERBOARD).round(4).to_csv(os.path.join(WORK, "leaderboard_gbm.csv"))""")
```

with:

```python
LEADERBOARD = {}
model_oof = {n: {} for n in ("lgb", "cat", "xgb", "hgb")}
model_te = {n: {} for n in ("lgb", "cat", "xgb", "hgb")}
print("Training GBM experts...")
for tt in TARGETS:
    m, idx, splits = get_splits(tt)
    leader = {}
    for name, mk in [("lgb", make_lgb), ("cat", make_cat), ("xgb", make_xgb), ("hgb", make_hgb)]:
        t0 = time.time()
        oof, tep = gbm_fit_predict(tt, mk, Xtr, Xte)
        r = record(name + "_" + tt, tt, oof, tep)
        leader[name] = r
        model_oof[name][tt] = oof; model_te[name][tt] = tep
        print(f"  {tt} {name}: RMSE={r:.4f} ({time.time()-t0:.0f}s)")
    LEADERBOARD[tt] = leader
    best = min(leader, key=leader.get)
    print(f"  -> best for {tt}: {best} RMSE={leader[best]:.4f}")
for name in ("lgb", "cat", "xgb", "hgb"):
    save_oof_artifact(name, model_oof[name], model_te[name])
pd.DataFrame(LEADERBOARD).round(4).to_csv(os.path.join(WORK, "leaderboard_gbm.csv"))""")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_gbm_artifact_save`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: persist GBM OOF artifacts (oof_lgb/cat/xgb/hgb.parquet)"
```

---

### Task 10: Judge figures — Fig-08 in-notebook + model-name/level-2 updates

**Files:**
- Modify: `build_pipeline_nb.py:816-830` (Fig-04 loop), `832-843` (Fig-05), `845-852` (Fig-06), `864-870` (Fig-08), `872-882` (Fig-09) — all inside the figures cell

**Interfaces:**
- Consumes: `base_models_for`, `store_key`, `L15_OOF`, `FINAL_OOF`, `dedup`, `Y`, `test`, `WORK`, `spearmanr` (imported earlier in the figures cell), `savefig`.
- Produces: updated `04_model_comparison.png`, `05_pred_vs_actual.png`, `06_residuals.png`, `08_cross_target_corr.png` + `cross_target_corr.csv`, `09_stack_improvement.png`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_fig08_in_notebook():
    code, _ = _build()
    assert "cross_target_corr.csv" in code
    assert "vault/figures" not in code
    assert "FINAL_OOF.get" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_fig08_in_notebook -> 'cross_target_corr.csv' not in code`

- [ ] **Step 3: Implement**

Edit A — Fig-04. Replace:

```python
# ---- Fig 4: model comparison ----
rows = []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    for b in BASE_MODELS:
        k = store_key(b, tt)
        if k in oof_store:
            rows.append({"target": tt, "model": b, "rmse": rmse_metric(Y[m], oof_store[k])})
    if tt in STACKED_OOF:
        rows.append({"target": tt, "model": "stack", "rmse": rmse_metric(Y[m], STACKED_OOF[tt])})
mc = pd.DataFrame(rows)
```

with:

```python
# ---- Fig 4: model comparison ----
rows = []
for tt in TARGETS:
    m = (dedup["target_type"] == tt).values
    for b in base_models_for(tt):
        k = store_key(b, tt)
        if k in oof_store:
            rows.append({"target": tt, "model": b, "rmse": rmse_metric(Y[m], oof_store[k])})
    if tt in L15_OOF:
        rows.append({"target": tt, "model": "l15", "rmse": rmse_metric(Y[m], L15_OOF[tt])})
    if tt in FINAL_OOF:
        rows.append({"target": tt, "model": "final", "rmse": rmse_metric(Y[m], FINAL_OOF[tt])})
mc = pd.DataFrame(rows)
```

Edit B — Fig-05 and Fig-06 prediction source. In both, replace:

```python
    yp = STACKED_OOF.get(tt, np.zeros(m.sum()))
```

with:

```python
    yp = FINAL_OOF.get(tt, np.zeros(m.sum()))
```

Edit C — Fig-08. Replace:

```python
# ---- Fig 8: cross-target correlation (physics) ----
ct = pd.read_csv(os.path.join("vault","figures","cross_target_corr.csv"), index_col=0) if os.path.exists(os.path.join("vault","figures","cross_target_corr.csv")) else None
if ct is not None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(ct.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Cross-target correlation (shared molecules)")
    savefig(fig, "08_cross_target_corr.png")
```

with:

```python
# ---- Fig 8: cross-target correlation (computed in-notebook from Y) ----
def _canon_target_map(tt):
    mm = (dedup["target_type"] == tt).values
    return dict(zip(dedup.loc[mm, "canon"].values, dedup.loc[mm, "target"].values))

ct_maps = {t: _canon_target_map(t) for t in TARGETS}
ct = np.full((len(TARGETS), len(TARGETS)), np.nan)
for i, a in enumerate(TARGETS):
    for j, b in enumerate(TARGETS):
        if i == j:
            ct[i, j] = 1.0; continue
        shared = sorted(set(ct_maps[a]) & set(ct_maps[b]))
        if len(shared) < 20:
            continue
        aa = np.array([ct_maps[a][c] for c in shared]); bb = np.array([ct_maps[b][c] for c in shared])
        ct[i, j] = spearmanr(aa, bb).statistic
_ctdf = pd.DataFrame(ct, index=TARGETS, columns=TARGETS)
_ctdf.round(4).to_csv(os.path.join(WORK, "cross_target_corr.csv"))
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(_ctdf.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
ax.set_title("Cross-target correlation (shared molecules)")
savefig(fig, "08_cross_target_corr.png")
```

Edit D — Fig-09. Replace:

```python
# ---- Fig 9: stack improvement ----
if mc is not None and (mc["model"] == "stack").any():
    base_mean = mc[mc.model != "stack"].groupby("target")["rmse"].min()
    st = mc[mc.model == "stack"].set_index("target")["rmse"]
```

with:

```python
# ---- Fig 9: stack improvement ----
if mc is not None and (mc["model"] == "final").any():
    base_mean = mc[~mc.model.isin(["l15", "final"])].groupby("target")["rmse"].min()
    st = mc[mc.model == "final"].set_index("target")["rmse"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_fig08_in_notebook`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: fig-08 in-notebook, figures use level-2 final preds"
```

---

### Task 11: Submission uses level-2 final predictions

**Files:**
- Modify: `build_pipeline_nb.py:890-899` (the `final` test-pred builder inside the submission cell)

**Interfaces:**
- Consumes: `FINAL_TE`, `L15_TE`, `base_models_for`, `store_key`, `test_store`, `LEADERBOARD`.
- Produces: `final` array; `submission.csv` written via unchanged code below the edited block. Physics bounds (lines 901–909) unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_nb.py`:

```python
def test_submission_uses_final():
    code, _ = _build()
    assert "final[m_te] = FINAL_TE[tt][m_te]" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_pipeline_nb.py`
Expected: `FAIL test_submission_uses_final -> 'final[m_te] = FINAL_TE[tt][m_te]' not in code`

- [ ] **Step 3: Implement**

Replace in `build_pipeline_nb.py`:

```python
# build final test preds: stack when available, else best base model
final = np.zeros(len(test))
for tt in TARGETS:
    m_te = (test["target_type"] == tt).values
    if tt in STACKED_TE:
        final[m_te] = STACKED_TE[tt][m_te]
    else:
        best = min(BASE_MODELS, key=lambda b: LEADERBOARD[tt].get(b, 1e9))
        k = (best + "_" + tt, tt) if best not in ("mtnn","gnn") else (best, tt)
        final[m_te] = test_store[k][m_te]
```

with:

```python
# build final test preds: level-2 when available, else level-1.5, else best base model
final = np.zeros(len(test))
for tt in TARGETS:
    m_te = (test["target_type"] == tt).values
    if tt in FINAL_TE:
        final[m_te] = FINAL_TE[tt][m_te]
    elif tt in L15_TE:
        final[m_te] = L15_TE[tt][m_te]
    else:
        best = min(base_models_for(tt), key=lambda b: LEADERBOARD[tt].get(b, 1e9))
        k = store_key(best, tt)
        final[m_te] = test_store[k][m_te]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_pipeline_nb.py`
Expected: `PASS test_submission_uses_final`

- [ ] **Step 5: Commit**

```bash
git add build_pipeline_nb.py tests/test_pipeline_nb.py
git commit -m "v5: submission uses level-2 final predictions"
```

---

### Task 12: Regenerate notebook + local smoke run (bug-check)

**Files:**
- Verify: generated `AISEHack_Round2_Pipeline.ipynb`; smoke artifacts under `vault/pipeline_out_smoke/`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence that the full pipeline runs end-to-end with `SMOKE=True` (5 folds) and writes every artifact. Not a numeric benchmark.

- [ ] **Step 1: Full test suite passes**

Run: `python tests/test_pipeline_nb.py`
Expected: all 11 `PASS` lines, exit code 0.

- [ ] **Step 2: Regenerate the notebook**

Run: `python build_pipeline_nb.py`
Expected: `wrote AISEHack_Round2_Pipeline.ipynb with <N> cells`.

- [ ] **Step 3: Execute the notebook in smoke mode**

Run:
```powershell
$env:POLYWIN_SMOKE="1"
python -m jupyter nbconvert --to notebook --execute --inplace "AISEHack_Round2_Pipeline.ipynb"
```
(If `python -m jupyter nbconvert` is unavailable, first run `python -m pip install jupyter nbconvert`.)

Expected: notebook executes without raising; the cell-1 output prints `SMOKE = True | GLOBAL_FOLDS = 5`.

- [ ] **Step 4: Verify all artifacts exist**

Run:
```powershell
$w = "vault\pipeline_out_smoke"
Get-ChildItem $w -Filter *.parquet | Select-Object -ExpandProperty Name
Get-ChildItem $w -Filter *.csv | Select-Object -ExpandProperty Name
Get-ChildItem $w\figures | Select-Object -ExpandProperty Name
```
Expected: `oof_lgb.parquet`, `oof_cat.parquet`, `oof_xgb.parquet`, `oof_hgb.parquet`, `oof_efn.parquet`, `oof_tgnn.parquet`, `l15_ridge.parquet`, `final_meta.parquet`; `folds.csv`, `leaderboard_gbm.csv`, `final_leaderboard.csv`, `cross_target_corr.csv`, `submission.csv`; all 9 figures including `08_cross_target_corr.png`. Also confirm `folds.csv` has exactly 5 distinct fold values.

- [ ] **Step 5: Check honest-OOF summary printed**

Open the executed notebook's `final_leaderboard.csv` and confirm every target row has `lgb/cat/xgb/hgb` plus (`efn` for the 6 electronic targets / `tgnn` for tg), and `l15` + `final` columns — no `nan` in the model columns (missing only where a model truly does not serve a target).

- [ ] **Step 6: Commit**

```bash
git add build_pipeline_nb.py AISEHack_Round2_Pipeline.ipynb tests/test_pipeline_nb.py
git commit -m "v5: smoke run passes end-to-end with honest OOF artifacts"
```

---

## Self-Review (run by the plan author)

**Spec coverage:**
- §4.1 GPU bootstrap → Task 1. §4.2 fold consistency/persistence + single `GLOBAL_FOLDS` + smoke folds → Tasks 1–2. §4.3 aux physics tasks → Task 3. §4.4 EFN (encoder widths, 6+10 heads, inverse-σ weighting, per-head masks, honest fold-wise OOF, fixed width, Tg excluded) → Task 5. §4.5 Tg isolation (own NN + own stack, no cross features) → Tasks 6 & 8 (`BASE_MODELS_TG`, `CROSS_MAP["tg"]=[]`). §4.6 reliability features → Task 8. §4.7 cross-target OOF layer (map + missing-indicator + fold-safety) → Task 8. §4.8 GNN removal → Task 7. §4.9 stacking/submission (BASE_MODELS split, store_key, physics bounds) → Tasks 8 & 11. §4.10 figures (Fig-08 in-notebook, Fig-04/09 names) → Task 10. §4.11 artifact persistence (all 9 files) → Tasks 2, 4, 5, 6, 8, 9. §6 validation plan (smoke then Kaggle) → Task 12.

**Placeholder scan:** no TBD/TODO; every step carries concrete code, exact strings, and expected output.

**Type/name consistency:** `efn_fit_predict` returns `({t: array}, {t: array})` and is recorded as `("efn_"+tt, tt)`; `store_key("efn", tt)` reproduces that key. Same for `tgnn`. `base_models_for` is used identically in Tasks 8, 10, 11. `FINAL_OOF`/`FINAL_TE`/`L15_OOF`/`L15_TE` names are stable across Tasks 8–11. `save_oof_artifact` signature `(name, oof_map, te_map)` matches calls in Tasks 5, 6, 9.
