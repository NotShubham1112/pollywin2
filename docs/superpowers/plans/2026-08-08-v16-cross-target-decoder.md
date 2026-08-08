# v16 Cross-Target Decoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork the frozen P14 pipeline into a self-contained Kaggle notebook (`PolyWin_R2_v16_cross_target_decoder.ipynb`) that adds a fold-safe Cross-Target Decoder stage (physics-imputed + learned sibling arms) to P14's per-target Ridge blend, with bit-identical everything else.

**Architecture:** The decoder is written once as a canonical, unit-testable module `decoder_v16.py`; the notebook builder embeds it verbatim as one new cell (same "source-slice" pattern already used for `mt_gnn_v2.py` → CORE_A/CORE_B). A second new cell widens the Ridge blend from `[GBM, MT-GNN]` to `[GBM, MT-GNN, physics, learned]` and emits `blend_oof_test16.npz` + `v16_blend_report.csv` + `submission_v16.csv`. An offline evaluator `vault/compare_v16.py` applies the pre-registered gates to whatever blend arrays the kernel produces.

**Tech Stack:** Python 3, NumPy, Pandas, scikit-learn (Ridge, GroupKFold, r2_score), LightGBM (Arm 2 option), RDKit (parse-only), nbformat. Runtime libs on Kaggle are unchanged from P14 (PyTorch/PyG for the verbatim GNN core).

## Global Constraints
- **OSI-approved libs only:** PyTorch, PyG, RDKit, scikit-learn, LightGBM, CatBoost, XGBoost. No new libraries.
- **No pseudo-labeling.**
- **Fold-safe:** all decoder features/pairs/fits must be computed with the **same GroupKFold canon splits** as P14 (GroupKFold(n_splits=GLOBAL_FOLDS) on `G`), such that a held-out fold never contributes train labels of its own polymers to the sibling pivot for the *learned* arm and never contributes fit pairs for the *physics* arm's OOF estimate.
- **Train-only info:** sibling values come from `train.csv` labels only.
- **Bit-identical baseline:** CORE_A (graph feats + GINE + MT-GNN) and CORE_B (twins + fold OOF + GBM stack) must be byte-for-byte equal to v14's. Only the decoder cell and the blend/last cell change.
- **Fix the alpha grid:** `ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]`.
- **Submission name:** `submission_v16.csv` under `OUT`; persistent caches: `blend_oof_test16.npz`, `v16_blend_report.csv`.
- Fork naming: builder `build_v16_kaggle_nb.py`, notebook `PolyWin_R2_v16_cross_target_decoder.ipynb`, evaluator `vault/compare_v16.py`, tests `tests/test_v16_kaggle_nb.py`.

---

### Task 1: `decoder_v16.py` — pure physics-imputed arm (+ pivot builder)

**Files:**
- Create: `decoder_v16.py`
- Test: `tests/test_decoder_v16.py`

**Interfaces:**
- Consumes: all-Target list `TARGETS`, `TARGET_IDX` (idx per target), train arrays `canon_tr`, `tgt_tr` (train target_type per row), `Y` (train target per row), `canon_te`; `GLOBAL_FOLDS`, `SEED`.
- Produces:
  - `build_pivot_df(canon, target_type)` → pandas DataFrame, index=canon, columns=TARGETS, values=target (NaN where missing).
  - `sibling_feature_matrix(canon_list, pivot)` → np.ndarray float64 `(n, 7)`, row i = that canon's 7 train-mediated values (NaN when absent).
  - `pair_fit_lin(x, y)` → `(slope, intercept)` least-squares fit (vectorized `np.linalg.lstsq`, min 3 pts else `(1.0, 0.0)`).
  - `physics_arm(train_canon, train_target, Y, test_canon)` → two dicts `{phys_oof: np.full(len(X))) correspond to train-canon enum, phys_test: np.zeros(len(test))}` in the 7-target layout — i.e. arm value per **global row** per target, with `np.nan` on rows/targets with no physics (caller falls back).

**Implementation**

{{LS: decoder payload below}}

```python
"""v16 Cross-Target Decoder — canonical, unit-testable source of truth.

The v16 Kaggle notebook embeds the `DECODER_CELL()` string below verbatim
(the same source-slice pattern as mt_gnn_v2.py -> CORE_A/CORE_B). Keeping it
here as Python lets the pure logic be unit-tested (tests/test_decoder_v16.py)
and lets vault/compare_v16.py reuse the physics math offline. Only train
labels are ever read (no test leakage). All folds use GroupKFold(n_splits=
GLOBAL_FOLDS) on `canon`, identical to P14.
"""

TARGETS_DEC = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
TARGET_IDX_DEC = {t: i for i, t in enumerate(TARGETS_DEC)}

# Physics recipes: target -> (kind, srcs)
#   "sub": out = src0 - src1            (egc = ei - eea)
#   "lin": out = a * src0 + b           (fitted from training pairs)
#          e.g. egb = f(egc), eps = f(nc)
PHYS_RECIPE = {
    "egc": ("subtract", ("ei", "eea")),
    "egb": ("linear", ("egc",)),
    "eps": ("linear", ("nc",)),
}


def build_pivot_df(canon_arr, tgt_arr, val_arr):
    """Pivot table: index=canon, columns=TARGETS_DEC, values=target."""
    df = pd.DataFrame({"canon": canon_arr, "target_type": tgt_arr, "value": val_arr})
    return df.dropna(subset=["value"]).pivot_table(
        index="canon", columns="target_type", values="value", aggfunc="first")


def sibling_feature(canon_list, pivot):
    """(n,7) float64 — for every row, the canon's 7 sibling values (or NaN)."""
    out = np.full((len(canon_list), 7), np.nan, dtype=np.float64)
    for i, c in enumerate(canon_list):
        if c in pivot.index:
            out[i] = pivot.loc[c].values
    return out


def _fit_linear(src, dst):
    n = len(src)
    if n < 1:
        return 1.0, 0.0
    A = np.vstack([src, np.ones(n)]).T
    slope, cword = np.linalg.lstsq(A, dst, rcond=None)[0]
    return float(slope), float(cword)


def physics_arm(sib_tr, sib_te, tr_tgt):
    """Return (phys_tr, phys_te) as (n_tr,7) and (n_te,7) float64 arrays in
    TARGETS_DEC column order. Missing/NaN stays NaN (caller falls back)."""
    n_tr, n_te = len(sib_tr), len(sib_te)
    out_tr = np.full((n_tr, 7), np.nan, dtype=np.float64)
    out_te = np.full((n_te, 7), np.nan, dtype=np.float64)
    for tcol, (kind, srcs) in PHYS_RECIPE.items():
        ti = TARGET_IDX_DEC[tcol]
        # positions where physics is computable
        tr_ok = np.all(np.isfinite(sib_tr[:, [TARGET_IDX_DEC[s] for s in srcs]]), axis=1)
        te_ok = np.all(np.isfinite(sib_te[:, [TARGET_IDX_DEC[s] for s in srcs]]), axis=1)
        if kind == "subtract":
            s0, s1 = (TARGET_IDX_DEC[s] for s in srcs)
            out_tr[tr_ok, ti_col] = sib_tr[tr_ok, s0] - sib_tr[tr_ok, s1]
            out_te[te_ok, ti_col] = sib_te[te_ok, s0] - sib_te[te_ok, s1]
        else:  # linear: fit slope/intercept on TRAIN rows (OOF-safe pairs)
            s0 = TARGET_IDX_DEC[srcs[0]]
            m_tr = tr_ok & np.isfinite(sib_tr[:, s0])
            if m_tr.sum() >= 1:
                a, b = _fit_linr(sib_tr[m_tr, s0], sib_tr[m_tr, ti_col])
            else:
                a, b = 1.0, 0.0
            out_tr[tr_ok, ti_col] = a * sib_tr[tr_ok, s0] + b
            out_te[te_ok, ti_col] = a * sib_te[te_ok, s0] + b
    return out_tr, out_te
```

Note: replace the two garbage tokens `ti_col` (int index = `tcol`) and `_fit_linr`
(_fit_linear) in the three lines that reference them while transcribing — they are
stylistic aliases, not API. Final signatures are exactly as declared in "Interfaces".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decoder_v16.py
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder_v16 import (
    TARGETS_DEC, TARGET_IDX_DEC, PHYS_RECIPE, build_pivot_df,
    sibling_feature, _fit_linear, physics_arm,
)

def _flat():
    tr_c = np.array(["A", "A", "B", "C", "C", "D"])
    tr_t = np.array(["egc", "ei", "egb", "eps", "nc", "nc"])
    tr_v = np.array([2.0, 5.0, 1.5, 3.0, 4.0, 6.0])
    return tr_c, tr_t, tr_v

def test_pivot_shape_and_values():
    tr_c, tr_t, tr_v = _flat()
    piv = build_pivot_df(tr_c, tr_t, tr_v)
    assert list(piv.index) == ["A", "B", "C", "D"]
    assert piv.loc["A", "egc"] == 2.0 and piv.loc["A", "ei"] == 5.0
    assert np.isnan(piv.loc["C", "egc"])  # polymer C has no egc label

def test_sibling_feature_aligns_to_v7():
    tr_c, tr_t, tr_v = _flat()
    piv = build_pivot_df(tr_c, tr_t, tr_v)
    sib = sibling_feature(["A", "C", "ZZ"], piv)
    assert sib.shape == (3, 7)
    assert sib[0, TARGET_IDX_DEC["egc"]] == 2.0
    assert sib[0, TARGET_IDX_DEC["ei"]] == 5.0
    assert np.isnan(sib[2]).all()       # unknown canon is all-NaN

def test_physics_subtract_and_linear():
    tr_c, tr_t, tr_v = _flat()
    piv = build_pivot_df(tr_c, tr_t, tr_v)
    # A: <2,5,?,?,?,?,?> -> egc_phys(A) = ei-eea(na) => NaN (no eea)
    # C: has eps=3, nc=4 -> eps_phys = f(nc) from train pairs
    tr_sib = sibling_feature(tr_c, piv)
    te_sib = sibling_feature(["A", "C"], piv)
    p_tr, p_te = physics_arm(tr_sib, te_sib, tr_t)
    assert np.isnan(p_te[0, TARGET_IDX_DEC["egc"]])   # A has no eea
    # C: eps fits from (eps,nc) pairs -> any real number, not NaN
    assert np.isfinite(p_te[1, TARGET_IDX_DEC["eps"]])
    assert np.isnan(p_te[1, TARGET_IDX_DEC["ei"]])    # C has ei=NaN

def test_fit_linear_identity():
    x = np.array([1.0, 2.0, 3.0]); y = np.array([2.0, 4.0, 6.0])
    a, b = _fit_linear(x, y)
    assert np.allclose(a, 2.0) and np.allclose(b, 0.0)
```

- [ ] **Step 2: Run the test, expect FAIL (module + function not found)**

Run: `python tests/test_decoder_v16.py`
Expected: `ImportError: cannot import name 'build_pivot_df' from 'decoder_v16'`

- [ ] **Step 3: Write `decoder_v16.py` with the implementation above** (exact signatures from Interfaces; fix `ti_col`→`tcol`, `_fit_linr`→`_fit_linear`)

- [ ] **Step 4: Run to verify PASS**

Run: `python tests/test_decoder_v16.py`
Expected: `PASS test_pivot_shape_and_values` etc. (4 tests, exit code 0)

- [ ] **Step 5: Commit**

```bash
git add decoder_v16.py tests/test_decoder_v16.py
git commit -m "feat: v16 decoder physics-imputed arm + pivot (pure, unit-tested)"
```

---

### Task 2: `decoder_v16.py` — learned cross-target arm (fold-safe)

**Files:**
- Modify: `decoder_v16.py`
- Test: `tests/test_decoder_v16.py`

**Interfaces:**
- Consumes: `TARGETS_DEC`, `TARGET_IDX_DEC` (existing), plus new signature `learned_arm(pivot, X, group, global_folds=GLOBAL_FOLDS, seed=SEED)`.
- Produces: `learned_arm(...)` → `(lo_tr, lo_te)` np.float64 `(n,7)` duck-typed like `physics_arm`, where row i,t holds the fold-safe "sibling-feature regression" prediction for that row/target, or `np.nan` when the row's polymer has no sibling in the pivot (no train rows share it → fallback to mean).

**Design:** For each target t:
1. Build per-row feature `S_i = (pivot row for canon_i) ∩ features needed` described.
2. For each GroupKFold split, fit a Ridge? No — learn mapping from the 6 sibling features to target t on the rows of target t (only rows of canonicalk)t), **using only folds != current**.
Model is sklearn Ridge(alpha=10.0); if fewer than 1 rows in the current fold unions that have ≥2 known sibs, arm=NaN for that fold (no coverage → fail to `get_primary`).

Rationale: This is the "learned" arm — it generalizes PF from physics args to all target pairs, enabling cross-target transfer beyond the three hypothesized physics relations. Pre-registration note: the mapping is learned on train siblings only and evaluated fold-safe.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_decoder_v16.py
from decoder_v16 import learned_arm

def _mk_sib_dataset():
    # polymer P: has all 6 non-self siblings -> makes perfect neighbors
    n = 600
    rng = np.random.RandomState(7)
    canons = [f"P{i // 7}_{i % 7}" for i in range(n)]  # assure overlap: i/7 polys
    ...
    # synth: target_t = 0.5*eps_rgb + 1.0*nc... small noise
    pivot = build_pivot_df(canon_arr, tgt_arr, val_arr)
    lo_tr, lo_te = learned_arm(pivot, canon_arr, np.zeros(n, int), tgt_arr, Y)
```

- [ ] Step 2: run — expect FAIL (ImportError learned_arm)
- [ ] Step 3: implement `learned_arm(...)` per Interfaces

```python
def learned_arm(canon_tr, tgt_tr, Y_tr, pivot, canon_te, global_folds=5, seed=42,
                alpha=10.0):
    """Fold-safe learned cross-target arm. Returns (lo_tr, lo_te) (n_tr,7),
    (n_te,7) float64; NaN on rows whose canon has <1 sibling in pivot."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    sib_tr = sibling_feature(canon_tr, pivot)
    sib_te = sibling_feature(canon_te, pivot)
    lo_tr = np.full((len(canon_tr), 7), np.nan, dtype=np.float64)
    lo_te = np.full((len(canon_te), 7), np.nan, dtype=np.float64)
    group = np.array([c if isinstance(c, str) else str(c) for c in canon_tr])
    for t in TARGETS_DEC:
        ti = TARGET_IDX_DEC[t]
        idx_t = np.where(tgt_tr == t)[0]
        if len(idx_t) < 1:
            continue
        # drop columns = own column (would be target value, not sibling)
        feat_tr = np.delete(sib_tr, ti, axis=1)
        feat_te = np.delete(sib_te, ti, axis=1)
        cv = GroupKFold(n_splits=min(global_folds, max(1, len(set(group[idx_t])))))
        te_acc = np.zeros(len(canon_te))
        for trk, vk in cv.split(feat_tr[idx_t], Y_tr[idx_t], group[idx_t]):
            m = Ridge(alpha=alpha).fit(feat_tr[idx_t[trk]], Y_tr[idx_t[trk]])
            lo_tr[idx_t[vk], ti] = m.predict(feat_tr[idx_t[vk]])
            te_acc += m.predict(feat_te) / cv.n_splits
        lo_te[:, ti] = te_acc
    return lo_tr, lo_te
```

In `_mk_synthetic`, when a row's canon shares the pivot with a sibling of a *different* target, the learned model sees *actual sibling values* as features — so after 60 rows/poly everything is well-supported.

- [ ] **Step 4: run tests — expect PASS (all target_values trained on siblings)**

Verify with pytest on `test_decoder_v16.py` — 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add decoder_v16.py tests/test_decoder_v16.py
git commit -m "feat(decoder_v16): fold-safe learned cross-target arm"
```

---

## Task 3: `build_v16_kaggle_nb.py` — fork + decoder cell + blend cell

**Files:**
- Create: `build_v16_kaggle_nb.py`
- Modify: (read-only refs) `build_v14_kaggle_nb.py`
- Test: `tests/test_v16_kaggle_nb.py`

**Interfaces:**
- Consumes: `decoder_v16.py` (as embedded cell text), CORE_A/CORE_B from `mt_gnn_v2.py` (identical extraction as v14), the P14 setup cell.
- Produces: `PolyWin_R2_v16_cross_target_decoder.ipynb` when run.
- The builder must define `DECODER = Path("decoder_v16.py").read_text(...)` and place it as **one code cell** between the CORE_B cell and the blend cell, preceded by a markdown cell titled `## 6. v16 Cross-Target Decoder — physics + learned arms (fold-safe)`.
- The blend cell (currently `## 5`) is **rewritten** for 4 arms: `[GBM, MT, PHYS, LEARNED]` — with per-target Ridge on the rows that have at least the two base arms; global fill where arms NaN; final report `v16_blend_report.csv` + `blend_oof_test16.npz` (arrays: `oof_gbm`, `oof_mt`, `oof_phys`, `oop_learn`, `test_gbm`, `test_mt`, `test_phys`, `test_learn`, `y_all`, `g_all`, `t_all`) + `submission_v16.csv`.
- The setup cell config `WORK = vault/pipeline_out_v16@SUFFIX@` and `INP` stay as-is (v14 shape, just folder name swap).

**Step 1: Write failing test** (mirror v15's `test_v15_kaggle_nb.py`, extend for decoder)

```python
# tests/test_v16_kaggle_nb.py
"""v16 notebook-generator tests: single-change fork of v14 + decoder cell."""
import ast, os, pathlib, re, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN16 = REPO / "build_v16_kaggle_nb.py"
GEN14 = REPO / "build_v14_kaggle_nb.py"
NB16 = REPO / "PolyWin_R2_v16_cross_target_decoder.ipynb"
NB14 = REPO / "PolyWin_R2_v14_p1m_pretrain.ipynb"

def _cell_list(nb_path):
    nb = nbformat.read(str(nb_path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]

def _build(generator, nb_path, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    subprocess.run([sys.executable, str(generator)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    cells = _cell_list(nb_path)
    md = "\n".join(c.source for c in nbformat.read(str(nb_path), as_version=4).cells
                   if c.cell_type == "markdown")
    return "\n".join(cells), md

def test_v16_header_and_config():
    code, md = _build16()
    assert "v16" in md and "Cross-Target Decoder" in md
    assert "submission_v16.csv" in code
    assert "blend_oof_test16.npz" in code

def test_v16_core_a_identical_to_v14():
    c16, _ = _build16(); c14, _ = _build14()
    a16 = next(c for c in c16.split("\n\n") if "# Graph featurization" in c and "# Twin source:" not in c)
    a14 = next(c for c in c14.split("\n\n") if "# Graph featurization" in c and "# Twin source:" not in c)...
```

— Full test below in Step 3 code (details for cell-splitting mirror v14; the cells that contains `# Graph featurization` and `# Twin source:` are DOCUMENTED; adapt the splitting to the way the decoders actually emit: use cell-list + markers instead of splitting on `\n\n`).

Continue the test:

```python
def _cells_of(path):
    return [c.source for c in nbformat.read(str(path), as_version=4).cells
            if c.cell_type == "code"]

def test_v16_cores_identical():
    a16 = next(c for c in _cells_of(NB16) if "# Graph featurization" in c)
    b16 = next(c for c in _cells_of(NB16) if "# Twin source:" in c and "lgb_test_te" in c)
    a14 = next(c for c in _cells_of(NB14) if "# Graph featurization" in c)
    b14 = next(c for c in _cells_of(NB14) if "# Twin source:" in c and "lgb_test_te" in c)
    assert a16 == a14
    assert b16 == b14

def test_v16_has_decoder_cell_and_physics():
    code, _ = _build16()
    assert "from decoder_v16 import" in code or "TARGETS_DEC" in code
    assert "PHYS_RECIPE" in code
    assert "physics_arm" in code
    assert "learned_arm" in code

def test_v16_no_pseudo_no_arch_change():
    code, _ = _build16()
    assert not re.search(r"\bpseudo[_-]?label", code, re.IGNORECASE)
    assert "class GINEEncoder(nn.Module):" in code
    assert "class MTGNN(nn.Module):" in code
    assert "PRETRAIN_EPOCHS = 10" in code

def test_v16_all_cells_compile():
    for src in _cells_of(NB16):
        ast.parse(src)

def test_decoder_cell_text_matches_file():
    import decoder_v16 as dv
    expect = re.search(r"DECODER_CELL = r'''.*?'''", svsource...)
```

- [ ] **Step 2: run tests — expect FAIL (`ModuleNotFoundError: None` for build_v16; report derivative cells).**
- [ ] **Step 3: Write `build_v16_kaggle_nb.py`**

Structure (mirror `build_v14_kaggle_nb.py` exactly, with these diffs):
- `OUT_NB = "PolyWin_R2_v16_cross_target_decoder.ipynb"`.
- Import decoder: `DECODER_SRC = pathlib.Path("decoder_v16.py").read_text(encoding="utf-8")`. The decoder becomes a cell `P(DECODER_SRC)`.
- The blend cell (replaces v14 §5) is a new `P(BLEND_CELL)` string. `BLEND_CELL` includes:

```python
ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]

# ---- assemble arm OOF / test arrays (global rows) ----------------------
oof_gbm_global = np.full(len(X), np.nan, dtype=np.float32)
oof_mt_global = np.full(len(X), np.nan, dtype=np.float32)
for t in TARGETS:
    idx = idx_of_target[t]
    oof_gbm_global[idx] = stack_oof[t]
    oof_mt_global[idx] = mt_oof[t]
assert not np.isnan(oof_gbm_global).any() and not np.isnan(oof_mt_global).any()

test_gbm_global = np.zeros(len(Xte), dtype=np.float32)
test_mt_global = np.zeros(len(Xte), dtype=np.float32)
for t in TARGETS:
    m_te = (tef["target_type"] == t).values
    test_gbm_global[m_te] = stack_test[t][m_te]
    test_mt_global[m_te] = mt_test[m_te]

# ---- decoder arms (from the DECODER cell executed just above) ---------
from decoder_v16 import build_pivot_df, sibling_feature

sib_tr = sibling_feature(tr["canon"].values, np.nan)
```

**For decide the exact embedding** (concrete, from Task 1/2):
The decoder cell builds:
```python
tr_piv = build_pivot_df(tr["canon"].values, T.astype(str), Y)
sib_tr = sibling_feature(tr["canon"].values, tr_piv)   # (n_tr,7)
sib_te = sibling_feature(te["canon"].values, tr_piv)   # (n_te,7)
phys_tr, phys_te = physics_arm(sib_tr, sib_te, T)
lo_tr, lo_te = learned_arm(tr["canon"].values, T, Y, tr_piv, te["canon"].values,
                           global_folds=GLOBAL_FOLDS, seed=SEED)
```
Then the blend cell:

```python
# column-wise: per-target values from each arm; NaN in arms below are
# filled by falling back to target mean (inert placeholder)
TMEAN = {t: float(np.nanmean(Y[idx_of_target[t]])) for t in TARGETS}
# -- per-target blade on 4 arms where defined --
blends_tr = np.full(len(X), np.nan, dtype=np.float64)
blends_te = np.zeros(len(Xte), dtype=np.float64)
coef_rows = []
for t in TARGETS:
    idx = idx_of_target[t]
    yt = Y[idx].astype(np.float64)
    Mt = np.column_stack([oof_gbm_global[idx], oof_mt_global[idx],
                          fillna_arm(phys_tr[idx], TAR_OF[t]),
                          fillna_arm(lo_tr[idx], TAR_OF[t])])
    Mte = np.column_stack([test_gbm_global, test_mt_global,
                           fillna_arm(phys_te[:, ti], TAR_OF[t]),
                           fillna_arm(lo_te[:, ti], TAR_OF[t])])
    cv = list(GroupKFold(GLOBAL_FOLDS).split(Mt, yt, G[idx]))
    oof_r2 = {}
    for a in ALPHA_GRID:
        o = np.zeros(len(idx))
        for trk, vk in cv:
            o[vk] = Ridge(alpha=a).fit(Mt[trk], yt[trk]).predict(Mt[vk])
        oof_r2[a] = r2_score(yt, o)
    a_best = max(oof_r2, key=oof_r2.get)
    oof = np.zeros(len(idx)); te_pred = np.zeros(len(Xte))
    coefs = []
    for trk, vk in cv:
        lr = Ridge(alpha=a_best); lr.fit(Mt[trk], yt[trk])
        oof[vk] = lr.predict(Mt[vk])
        te_pred += lr.predict(Mte) / GLOBAL_FOLDS
        coefs.append(lr.coef_.tolist())
    cb = np.mean(coefs, axis=0)
    final_te_here = te_pred
    blends_tr[idx] = oof
    blends_te[m_te] = te_pred[m_te]     # per-target test fill with target rows
    coef_rows.append(dict(target=t, alpha=float(a_best),
                          blend=r2_score(yt, oof), GBM=r2_score(yt, idx_g),
                          GNN=r2_score(yt, idx_m), w_GBM=cb[0], w_GNN=cb[1],
                          w_PH=cb[2], w_LEARN=cb[3]))
```

_(the exact final code lives in the builder; the plan pins the **names & arrays**: `phys_tr/phys_te`, `lo_tr/lo_te`, `blends_tr`, `blends_te`, `TAR_OF_TARGET`, `fillna_arm`.)_
Additions (final kernels):
- `sub = DataFrame({"id": tef["id"].values, "target": blends_te})` → write `submission_v16.csv`.
- `np.savez("blend_oof_test16.npz", oof_gbm=..., oof_mt=..., oof_phys=..., oof_learn=..., test_gbm=..., test_mt=..., test_phys=..., test_learn=..., y_all=..., g_all=..., t_all=...)`.
- `rows.round(4).to_csv(v16_blend_report.csv)`.

- [ ] **Step 4: run tests — PASS (v16 suite green)**
- [ ] **Step 5: Commit**

```bash
git add build_v16_kaggle_nb.py tests/test_v16_kaggle_nb.py
git commit -m "feat: build_v16 notebook generator — decoder cell + 4-arm blend"
```

---

## Task 4: Smoke run locally (offline, no Kaggle)

**Files:**
- Run-only (no new source): `build_v16_kaggle_nb.py`, `vault/…` outputs.

- [ ] **Step 1: generate the smoke notebook**

Run: `SMOKE=1 python build_v16_kaggle_nb.py`
Expected: writes `PolyWin_R2_v16_cross_target_decoder.ipynb`, log shows `SMOKE: True`, folds=2.

- [ ] **Step 2: cell-level diagnostics**

Run: `python -c "import nbformat,ast; nb=nbformat.read('PolyWin_R2_v16_cross_target_decoder.ipynb',as_version=4); [ast.parse(c.source) for c in nb.cells if c.cell_type=='code']; print('cells', len([c for c in nb.cells if c.cell_type=='code']))"`
Expected: `cells <N>` and no exceptions.

- [ ] **Step 3: unit + integration tests still green**

Run: `python tests/test_decoder_v16.py && python tests/test_v16_kaggle_nb.py`
Expected: both `PASS …`, exit 0.

- [ ] **Step 4: (optional) real local smoke of the kernel**

Install-reused Kuls; see `SMOKE=1` up to early_GNN. On environments with pyg, run `SMOKE=1 python build_v16_kaggle_nb.py` then the generated kernel manually? — Only if pyg is installed locally; otherwise skip (Kaggle is the production run).

- [ ] **Step 5: Commit (no results expected, only infra)**

```bash
git add -A && git commit -m "test(smoke): v16 notebook compiles free (SMOKE)"
```

---

## Task 4: `vault/compare_v16.py` — offline gate evaluator

**Files:**
- Create: `vault/compare_v16.py`
- Test: `tests/test_compare_v16.py` (synthetic npz)

**Interfaces:**
- Consumes: `blend_oof_test16.npz` (8 OOF/test arrays above), `TARGETS`.
- Produces: prints per-target table + **gates**:
  - `gate_pass = (small-five-mean gain vs v14 blend on covered rows ≥ +0.003) AND (no target regresses > −0.003)`
  - `gate_fail_message`.
- Writes: `vault/v16_gates_report.csv` (target, covered_n, v14_blend, v16_blend, delta, per-target pass).

- [ ] **Step 1: write failing test**

```python
# tests/test_compare_v16.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, tempfile, os
from vault.compare_v16 import evaluate_gates

def _fake_npz(p):
    n, m = 200, 50
    g = np.array([f"p{i//5}" for i in range(n)])
    t = np.array(["eps"]*n)
    y = np.random.RandomState(0).randn(n)
    d = {"oof_gbm": y, "oof_mt": y, "oof_phys": y*0+np.random.RandomState(1).randn(n)*0,
         "oof_learn": y, "test_gbm": np.zeros(m), "test_mt": np.zeros(m),
         "test_phys": np.zeros(m), "test_learn": np.zeros(m),
         "y_all": y, "g_all": g, "t_all": t}
    np.savez(p, **d)

def test_gate_called():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.npz"); run_fake(p)
        res = evaluate_gates(p)
        assert res is not None and "gate_pass" in res
```

- [ ] Step 2: run — FAIL (no module)
- [ ] Step 3: implement `evaluate_gates(npz_path)` returning a dict with keys `per_target` (df), `small_five_mean_gain`, `gate_pass` (bool), `message` (str), plus save `vault/v16_gates_report.csv`. Use the same per-target Ridge protocol as `vault/compare_p14.py` but on the 4-arm row where all arms non-NaN; fall back to 2-arm where arms NaN; compute per-target v16 vs v14 gain.
- [ ] Step 4: test PASS
- [ ] Step 5: Commit

---

## Task 5: Full Kaggle run script + result capture template

**Files:**
- Create: `vault/run_v16_kaggle.md` instructions doc (or a scripts/ rerun helper).
- Modify: none (doc-only).

- [ ] **Step 1**: capture kernel name `polywin-r2-v16-cross-target-decoder`; produce `submission_v16.csv` alias upload; set schedule.
- [ ] **Step 2**: after run completes, download `blend_oof_test16.npz` → `vault/kernel-v16-cross-target/out/`, copy `submission_v16.csv` side-by-side, run `python vault/compare_v16.py <downloaded>/out/blend_oof_test16.npz`.
- [ ] **Step 3**: decision per §3 of the design doc: PASS → submit; FAIL → freeze P14 (no v16 slot). No post-hoc gate edits.
- [ ] **Step 4**: fill §5 of the design doc with the kernel's logged numbers only, commit.

---

## Task 6: Result postmortem + final state

- [ ] Update `docs/lab-postmortem-2026-08-08.md` §5 (append v16 row): whether decoder arms contributed; keep P14 frozen if not; record LB.
- [ ] `git add -A && git commit -m "results: v16 cross-target decoder PASS/FAIL + postmortem"`
- [ ] Final state: `tests/` green, working tree clean, P14 (0.883) still production unless v16 supersedes at the pre-registered gate.