# PolyWin R2 — v15 EPS/NC Focus (Design)

Date: 2026-08-08
Status: Approved (pre-registered in session; single-point change)
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v15 exists

P14 (full-PI1M pretraining, LB **0.883**) was judged **PASS** on the OOF gate:

- Blend OOF **0.8769** vs v13 **0.8689** → +0.0080 (criterion B ≥ +0.005).
- Criterion A (correlation drop) was **falsified** — GBM↔MT-GNN corr actually **rose**
  0.9513 → 0.9552. The gain is **NOT** from added diversity; it is from a **stronger
  shared encoder**.
- Gain is concentrated entirely in the two weakest targets:

  | target | v13 blend OOF | v14 blend OOF | Δ |
  |--------|---------------|---------------|-----|
  | eps    | 0.7749        | 0.8009        | **+0.0260** |
  | nc     | 0.8417        | 0.8657        | **+0.0240** |
  | egc    |               |               | +0.0036 |
  | tg     |               |               | +0.0018 |
  | egb    |               |               | +0.0031 |
  | ei     |               |               | −0.0009 |
  | eea    |               |               | −0.0018 |

Because the metric is the **unweighted mean R² over all 7 targets**, eps/nc (the two
rows with the largest absolute gaps to 0.90) are the highest-leverage lever. P14 already
pulled +0.026/+0.024 out of them purely by pretrain strength. **v15 asks: can we pull
more of that by giving eps/nc a larger share of the fine-tune gradient?**

Pseudo-labeling is **explicitly out of scope** (v8 history: OOF +0.017 → LB −0.019).
v15 is a single, fully-pre-registered, low-risk experiment. If it fails, **P14 (0.883)
is the final submission** — no further tuning.

## 2. The single change

v14 weights every fine-tune row by the target's inverse frequency:

```python
g.w = torch.tensor([1.0 / freq[row.target_type]], dtype=torch.float)
```

(freq is the normalized target_type count: eps/nc are 229/7409 ≈ 0.0309 → base w ≈ 32.4.)

**v15 multiplies that weight by an extra focus factor for eps and nc only:**

```python
g.w = torch.tensor([1.0 / freq[row.target_type] * TGT_FOCUS.get(row.target_type, 1.0)], dtype=torch.float)
```

with `TGT_FOCUS = {"eps": 2.0, "nc": 2.0}`.

- Applied **only inside `build_graphs`** (the fine-tune graph construction), so it
  affects **only the training loss** — test graphs and the val MSE are untouched.
- Everything else is **BIT-IDENTICAL to v14**: full-PI1M pretrain (995k, 10 epochs),
  GNN seeds 42/999/2025, 5 folds, max-epochs 120, patience 20, batch 256, GBM trio
  stack, per-target Ridge(alpha∈grid) blend, descriptors/fingerprints, submission path.
- No pseudo-labeling. No architecture change. No new libraries (OSI-approved only).

## 3. Baseline & success criteria (pre-registered)

- Reference: **v14** (its verbatim parent), same offline compare protocol as
  `vault/compare_p14.py` (GroupKFold(5) on canonical smiles, per-target R², ridge
  blend over the same alpha grid).
- **Pass:** (EPS per-target blend OOF gain ≥ **+0.01** OR NC ≥ **+0.01**) **AND**
  overall (7-target mean) blend OOF ≥ **+0.003** vs v14.
- **Fail:** overall gain < +0.003, OR eps/nc gain but other targets regress enough to
  offset the mean gain.
- On **Fail → STOP**: keep **P14 (LB 0.883)** as the final submission. No more
  fine-tune experiments. A very constrained pseudo-label probe may be discussed as the
  *only* remaining lever, and only after the fail result is in hand.
- LB is the arbiter; if v15 passes OOF, submit `submission_v15.csv` and confirm on LB
  before replacing P14 as the standing best.

## 4. Deliverables

- `build_v15_kaggle_nb.py` (fork of `build_v14_kaggle_nb.py` + the one-line `g.w` boost
  via a CORE_A substitution; `TGT_FOCUS` set in the notebook setup cell) →
  `PolyWin_R2_v15_epsnc_focus.ipynb`.
- `tests/test_v15_kaggle_nb.py` (TDD; asserts the boost line is present AND that CORE_A
  differs from v14 by exactly that one line; everything else bit-identical; cells compile).
- Smoke run locally, then push to Kaggle (`polywin-r2-v15-epsnc-focus`), full run,
  download, evaluate vs v14, decide per §3.
