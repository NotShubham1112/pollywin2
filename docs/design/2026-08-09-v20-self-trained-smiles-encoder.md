# v20 Self-Trained SMILES Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Each task is TDD: write the failing test, run it, implement, run it, commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leak-free, self-trained polymer-SMILES encoder arm (`oof_trf`/`test_trf`) to the P14 per-target Ridge blend, gate it locally, and ship it as a single self-contained Kaggle notebook — only if the pre-registered OOF gates pass.

**Architecture:** A compact RoBERTa-style masked-region encoder (~5–15M params) pretrained from scratch on PI1M SMILES inside the notebook (no pretrained weights; OSI-only code, no `transformers`/`tokenizers`). Since the encoder is label-free, encode train+test SMILES once, then use fold-safe per-target Ridge heads on the frozen pooled features to produce `oof_trf`/`test_trf`. Final per-target Ridge over `[oof_gbm, oof_mt, oof_trf]` with the same alpha grid and GroupKFold-on-`smiles` protocol as P14 (`vault/final_synthesis.py` `fold_safe_blend`). Gates decide the submission.

**Tech Stack:** Python ≥3.10, PyTorch (env `torch 2.11+cu128`), scikit-learn, numpy, pandas. No RDKit needed for the encoder path (raw SMILES strings only). No `transformers`/`tokenizers` runtime dependency — encoder tokenizer is pure-Python.

## Global Constraints

- USE ONLY official data (`official_dataset/train.csv`, `test.csv`, `PI1M.csv`). No uploaded/downloaded weights, no external datasets, no cached artifacts inside the shipped notebook.
- All model weights/artifacts are produced during the notebook run (vault/polywinr2 constraint).
- OSI-approved libs only (PyTorch, scikit-learn). No `pip install transformers` / `tokenizers` in the shipped kernel.
- Leak-safe by construction: GroupKFold on `smiles`; NO sibling/cross-target features anywhere (v16/v19 leak class is forbidden).
- Deterministic seeding: `random`, `np.random`, `torch.manual_seed` seeded from `SEED = 42` (`V20_SEED` env override).
- Run from repo root `D:\Parth\ploywin r2`. PowerShell has no heredocs → write scripts via file tools, run `python <file>`.
- SMOKE mode via env `SMOKE=1` (fast path, small encoder, few rows) must pass end-to-end before the full gate run.
- This is a Windows PowerShell host: single-line commands with `;` chaining, no `&&`.

## Data Verified

- `official_dataset/train.csv` (7409 rows; cols `smiles, target, target_type`); `test.csv` (4940 rows; cols `id, smiles, target_type`); `PI1M.csv` (col `SMILES`).
- `vault/pipeline_out_pretrain/superblend_oof.npz` keys: `oof_gbm` (7409), `oof_mt` (7409), `test_gbm` (4940), `test_mt` (4940), `target_type_train` (7409), `target_type_test` (4940), `y_train` (7409). P14 per-target OOF mean R² = 0.8641 (equal weight); LB 0.883. TARGETS = `[eea, egb, egc, ei, eps, nc, tg]`.
- `vault/final_synthesis.py` `fold_safe_blend(M, y, g, alphas, n_splits=5)` is the VERBATIM P14 blend to extend (T3 copies it, only the matrix gains a 3rd column). Groups `g` are `train["smiles"]` canonical.

---

### Task 1: Pure-Python SMILES tokenizer (no deps)

**Files:**
- Create: `v20_codec.py`
- Create: `tests/test_v20_codec.py`

**Interfaces:**
- `build_tokenizer(smiles, max_vocab=1600, min_count=2) -> dict(tok2id: dict, id2tok: dict)`
  - Reserved ids `{"[PAD]":0, "[CLS]":1, "[MASK]":2, "[UNK]":3}`; `[CLS]` prepended on encode.
- `tokenize_batch(tok, smiles, max_len=128) -> np.ndarray (n, max_len) int32`
  - `[CLS]` first; unknown → `[UNK]`; right-pad with `[PAD]`; cut to `max_len`.
  - NOTE: parameter order is `(tok, smiles)` — the plan's Task-1 prose originally said `(smiles, tok)` but its own test calls `tokenize_batch(toks, [...])`; the implemented signature follows the test and is authoritative for all later tasks.
- Token regex (pure ASCII, no external lib):
  `_TOK = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|\*|[A-Z][a-z]?|[0-9]{2}|[0-9]|[()\[\]=#\\/@+%.])")`

**Steps:**

- [ ] **Step 1:** Write failing tests in `tests/test_v20_codec.py`

```python
import numpy as np, pytest
from v20_codec import build_tokenizer, tokenize_batch

def test_vocab_specials_and_idx():
    toks = build_tokenizer(["C"*5, "c1ccccc1", "[Fe]"])
    for s in ("[PAD]", "[CLS]", "[MASK]", "[UNK]"):
        assert s in toks["tok2id"]
    assert toks["tok2id"]["[PAD]"] == 0

def test_tokenize_properties():
    toks = build_tokenizer(["CCCC"*10, "N[Fe]Cl"*3])
    x = tokenize_batch(toks, ["CCCC"*10, "", "N[Fe]Cl"*3], max_len=16)
    assert isinstance(x, np.ndarray) and x.shape == (3, 16)
    assert x[0,0] == toks["tok2id"]["[CLS]"]
    assert (x[0,:] == 0).sum() >= 3       # padding present
    assert x.dtype == np.int32
```

- [ ] **Step 2:** Run `python -m pytest tests/test_v20_codec.py -v` → FAIL (`ModuleNotFoundError`)
- [ ] **Step 3:** Implement `v20_codec.py`
- [ ] **Step 4:** Run tests → PASS
- [ ] **Step 5:** Commit `feat(v20): pure-Python SMILES tokenizer`

---

## Task 2: `MaskEncoder` (masked-region transformer) + `pool_embeddings`

**Files:**
- Create: `v20_encoder.py`
- Test: `tests/test_v20_encoder.py`

**Interfaces:**
- `class MaskEncoder(nn.Module)` init `(vocab, d=128, layers=2, heads=4, ff=512, max_len=128, dropout=0.1)` — same layout as `mt_gnn_v2.GINEEncoder` (self-attention only, no RDKit).
- `forward(ids, mask=None) -> (logits, features)` where `features` (n,max_len,d) is the last-layer output used for pooling.
- `pool_embeddings(model, ids, device="cpu", max_len=None) -> np.ndarray (n, d)` — `ids != [PAD]` mask, mean-pool over non-pad tokens.
- `pretrain_encoder(model, ids, epochs=2, bs=64, lr=3e-4, seed=SEED, mask_p=0.15) -> list[float]` — masked-token prediction (predict masked positions), AdamW.

**Steps:**

- [ ] **Step 1:** failing tests

```python
def test_forward_shape_and_pool():
    import numpy as np, torch
    from v20_encoder import MaskEncoder, pool_embeddings
    ids = torch.randint(4, 20, (6, 32))
    m = MaskEncoder(vocab=100)
    _, feats = m(ids)
    assert feats.shape == (6, 32, 128)
    pool = pool_embeddings(m, ids, max_len=32)
    assert pool.shape == (6, 128)
    assert np.isfinite(pool).all()
```

- [ ] **Step 2:** run → FAIL
- [ ] **Step 3:** implement
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat(v20): masked-region encoder + pooled embeddings`

---

## Task 3: `compute_trf_arm` — fold-safe per-target Ridge on frozen pool

**Files:**
- Create: `v20_arm_cv.py`
- Test: `tests/test_v20_arm_cv.py`

**Interface:**
- `compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g, n_splits=5, seed=42) -> (oof_trf: (n_tr,), test_trf: (n_te,))`
  - `pool_tr`/`pool_te` are FROZEN, label-free SMILES-encoder embeddings (n_tr=7409/n_te=4940 unfolded rows). `tt_tr`/`tt_te` are per-row target types (test must be passed explicitly so test rows align at original indices). `g` = canonical smiles group id per train row.
  - `GroupKFold(n_splits)` ONCE over all train rows, grouped on `g` (= smiles): same-polymer rows always land in one fold → no leakage.
  - Per fold: per-target `Ridge(alpha=1.0, fit_intercept=True)` heads trained on that fold's train rows for each target present; that fold's held-out rows of each target get their target head's prediction (written back at exact original indices). Test rows: mean of the 5 fold heads (per target). Degenerate fallbacks (target < n_splits rows, single-row group, test-only target) documented in the docstring; never NaN, never crash.
  - NOTE: authoritative signature has `tt_te` as 6th positional arg (the brief's prose omitted it; test alignment at original 4940 indices requires it).

**Steps:**
- [ ] **1:** failing tests (synthetic: make y linear in a known column → corr>0.98; leak check: group never shared between train/valid per row)
- [ ] **2:** run → FAIL
- [ ] **3:** implement `trf_arm_cv.py`
- [ ] **4:** tests → PASS
- [ ] **5:** commit `feat(v20): fold-safe per-target RA arm on frozen pool`

---

## Task 4: `blend_3d` — copy `fold_safe_blend` from `final_synthesis.py:84-105`, add third column

**Files:**
- Create: `v20_blend.py`
- Test: `tests/test_v20_blend.py`

**Interfaces:**
- `blend_3d(M_tr, y, g, alphas=[0.1,0.5,1.0,2.5,5.0,10.0,25.0], n_splits=5) -> (oof: (n,), coefs_mean: (3,))` — per-target inner alpha selection then refit at best alpha (verbatim `fold_safe_blend`, matrix `M` has 3 columns).

**Steps:**
- [ ] **1:** failing test: with `M` = 2 informative columns + a 3rd near-zero column, `blend_3d` OOF corr ≥ 0.999; 3-col alpha in grid; coefs length 3.
- [ ] **2:** run → FAIL
- [ ] **3:** implement calling mono-ridge logic verbatim from P14 with `M[:, [0,1,2]]`.
- [ ] **4:** run → PASS
- [ ] **5:** commit `feat(blend): per-target 3-arm Ridge (still fold-safe on smiles)`

---

## Task 5: `run_v20_gate.py` — local gate runner (SMOKE + FULL)

**Files:**
- Create: `run_v20_gate.py`, `v20_gate_report.py`
- Test: `tests/test_v20_gate.py`

**Interfaces:**
- `compute_gate_report(mean_delta, worst_delta, alphas, thr_mean=0.003, thr_worst=0.003, alpha_cap=0.30) -> dict(pass, mean_delta, worst_delta, alphas_ok)`
- `write_submission(df, path)` — header `id,target`, exactly 4940 rows, id order = `test.csv`.

Pipeline:
1. Load `train.csv`, `test.csv`, `superblend_oof.npz`, `PI1M.csv`.
2. Build tokenizer on PI1M (rows budget `V20_PI_COUNT`, default 20000; SMOKE 300).
3. Compute train/test masks → pretrain `MaskEncoder` (SMOKE: d=32, layers=2, 1 epoch).
4. `pool_embeddings` on all train & test rows (each row encoded once; pool for train = `(7409,d)`, test `(4940,d)`).
5. `compute_trf_arm(...)` → `oof_trf`, `test_trf`.
6. `blend_3d` on `[oof_gbm, oof_mt, oof_trf]` against `y` (all 7409) → `oof_b20`, per-target alpha.
7. Gates:
   - equal-weight mean R² (7 targets): `mean_b20 - 0.8641 >= +0.003`
   - every target `Δ >= -0.003`
   - all per-target blend alphas ≤ 0.30
8. If pass: `test_pred = blend_3d(...)[1]` on `[test_gbm, test_mt, test_trf]`; write `submission_v20.csv` in exact P14 writer format (n=4940, order = test order). If fail: log `GATE=FAIL → P14 stays final`.

**Steps:**
- [ ] Unit tests for `compute_gate_report` and `write_submission` (header `id,target`, 4940 rows, correct order).
- [ ] SMOKE full end-to-end run ≤ a few minutes on CPU → prints gate block.
- [ ] commit `feat(v20): gate runner (submit only if gates pass)`

---

## Task 6: notebook builder — `build_v20_kaggle_nb.py` (mirror `build_v14_kaggle_nb.py`)

**Files:**
- Create: `build_v20_kaggle_nb.py` (mirror structure; code slices from the runner/encoder files series)
- Test: `tests/test_v20_nb.py` (assert notebook roundtrip: contains-Gate block, has no absolute local paths, at the end globals `oof_v20`, `test_pred`, `GATE`)

**Steps:**
- [ ] Notebook builds; `SMOKE` path runs the `.ipynb` via `nbconvert --execute` headless to verify reproducibility.
- [ ] commit `feat(v20): self-contained notebook builder`

---

## Task 7: full gate run (local) + decision

- [ ] FULL run: `python run_v20_gate.py` (20k PI1M, d=256, 4 layers, 2 epochs) — budget ~2 h on the local GPU; CPU fallback ok.
- [ ] Record report: per-target R², alphas, deltas vs P14, gate verdict.
- [ ] Re-run `V20_SEED=7` → confirm stability.
- [ ] Decision per pre-registration: PASS (both seeds) → Task 6 notebook build + SMOKE + bit-check against local; then (one-shot) Kaggle push, run, download submission, submit. FAIL → P14 final; update `vault/final_status.md` with v20 report (no submission slot spent).

---

## Self-Review Checklist

- [ ] No external weights/artifacts in ANY shipped notebook; `enable_internet=false` in `kernel-metadata.json`.
- [ ] Leak-safe: encoder trained only on PI1M strings (no labels); per-target heads GroupKFold on `smiles`; blend GroupKFold on `train["smiles"]` (same protocol as P14).
- [ ] Same authoritative P14 blend (verbatim `fold_safe_blend`) — third column only.
- [ ] Gates: +0.003 mean / −0.003 worst / alphas ≤ 0.30 / cross-seed stable → enforced before any submission.
- [ ] Budget: after full local gate (≤ ~8 h), one Kaggle GPU run only if all gates hold.
- [ ] Commit after each task; no generated `.npz`/`.csv` artifacts committed.