# v16 Cross-Target Decoder — Kaggle run script

Kernel: `shubhamkambli11/polywin-r2-v16-cross-target-decoder`
Competition: `ppp-round-2` | GPU + Internet enabled | private notebook

Same push pattern as v14/v15: a push dir with `kernel-metadata.json` + the built
.ipynb, then `kaggle kernels push`.

---

## 0. Prereqs (already done)

- [x] Smoke run green: `vault/pipeline_out_v16_smoke/blend_oof_test16.npz`
  - `oof_gbm/oof_mt` OOF fully finite (7409), `oof_phys` 432, `oof_learn` 1311.
  - `submission_v16.csv` 4940 rows, 0 NaN.
  - Gate on the smoke blend: **PASS** — small-five weighted gain **+0.0403**,
    worst target delta `-0.0000` (> -0.003). See `vault/v16_gates_report.csv`.
- Unit + integration suites green:
  `python tests/test_decoder_v16.py && python tests/test_v16_kaggle_nb.py && python tests/test_compare_v16.py`

## 1. Build the production (non-smoke) notebook

The v16 notebook currently on disk at repo root was generated with `SMOKE=1`.
Regenerate in production mode (folds=5, 3 GNN seeds, full pretrain):

```
$env:SMOKE="0"; python build_v16_kaggle_nb.py
```

Verify the log prints `SMOKE: False` and `folds=5` (only `@FOLDS@` etc change).

## 2. Assemble push dir

```
mkdir vault/kernel-v16-cross-target
copy PolyWin_R2_v16_cross_target_decoder.ipynb vault/kernel-v16-cross-target/
```

Write `vault/kernel-v16-cross-target/kernel-metadata.json` (mirror v15):

```json
{"competition_sources":["ppp-round-2"],
 "id":"shubhamkambli11/polywin-r16-cross-target-decoder",
 "enable_gpu":true,"kernel_type":"notebook","dataset_sources":[],
 "is_private":true,"model_sources":[],
 "enable_internet":true,"language":"python",
 "title":"polywin r2 v16 cross target decoder",
 "code_file":"PolyWin_R2_v16_cross_target_decoder.ipynb"}
```

> The metadata `id`/`title` must be consistent with the kernel slug
> `polywin-r2-v16-cross-target-decoder`; the `code_file` must equal the ipynb
> filename in the push dir.

## 3. Push

```
kaggle kernels push -p vault/kernel-v16-cross-target
```

Metrics only visible after a complete run. This run is **1 public-LB-portfolio
judgement**: submission_v16.csv becomes the v16 submission (copies the validated
P14 inputs side-by-side).

## 4. Download outputs on completion

Wait for kernel status `complete` (sync with `kaggle kernels status
shubhamkambli11/polywin-r16-cross-target-decoder`).

```
mkdir -p vault/kernel-v16-cross-target/out
kaggle kernels output shubhamkambli11/polywin-r16-cross-target-decoder -p vault/kernel-v16-cross-target/out
```

Expected files in `vault/kernel-v16-cross-target/out/`:
- `blend_oof_test16.npz`
- `v16_blend_report.csv`
- `submission_v16.csv`
- `pretrained_encoder.pt`, `*.log`

## 5. Evaluate the production gate (pre-registered, no edits)

```
python vault/compare_v16.py vault/kernel-v16-cross-target/out/blend_oof_test16.npz
```

Gate (design doc §3):
- `gate_pass` = small-five weighted-mean gain on arms-covered rows ≥ **+0.003**
  AND no target regresses **> -0.003**.
- Also fail if physics/learned arms contribute ≤ 0 at runtime.

Decision:
- **PASS** → submit `submission_v16.csv`; record public LB.
  - confirm field: public LB ≥ **0.886** (honest bar). **0.898** = stretch/top-10 (not gated).
- **FAIL → freeze P14 (LB 0.883).** Do NOT spend a v16 submission slot.
- No post-hoc tightening/loosening of the gate numbers.

## 6. Record results

Fill §5 of `docs/superpowers/specs/2026-08-08-v16-cross-target-decoder-design.md`
with only the run's logged numbers (gate printout + public LB + public rank), then
append to `docs/lab-postmortem-2026-08-08.md`. Commit with the PASS/FAIL outcome.

Files to keep green and committed:
- Decoder/under tests: `tests/test_decoder_v16.py`, `tests/test_v16_kaggle_nb.py`, `tests/test_compare_v16.py`.
- The evaluator + report: `vault/compare_v16.py`, `vault/v16_gates_report.csv`.
- The canonical decoder + builder: `decoder_v16.py`, `build_v16_kaggle_nb.py`.