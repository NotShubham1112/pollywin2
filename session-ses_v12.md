# Session — v12 Chemistry Bucket MoE (implementation + Kaggle run)

Date: 2026-08-07
Repo: `D:\Parth\ploywin r2` (branch `main`)

## What was done

Executed `docs/superpowers/plans/2026-08-06-v12-bucket-moe.md` (9 tasks, TDD) via `executing-plans`.

Commits (all on `main`):
- `efe1847` Task 1: scaffold generator + bootstrap/data cells + tests
- `e7ce0f8` Task 2: feature factory + cleaning cells (v8 Layer 2 reuse)
- `46636cc` Task 3: validation harness + GBM experts (smoke cache branch)
- `220979a` Task 4: stacking L1.5 + L2 meta (Layer 9 reuse)
- `5fcfdd4` Task 5: pretrained GNN stage (v10 reuse, smoke cache branch)
- `d53c88c` Task 6: v11 reference blend (per-target fold-safe weights)
- `6861c7c` Task 7: chemistry bucket MoE stage (KMeans routing + fold-safe per-cluster weights + diag/compare CSVs)
- `d0327e6` Task 8: submission selection (bucket-MoE vs v11 fallback) + figure 24 + complete marker
- `0e04028` Task 9 bugfix: `NameError` in `run_bucket_moe` (`km` was unpacked as `_` but returned) + regression test; smoke run validated
- `2803536` docs commit
- `73be6c6` LB progression table updated with v8–v12 values

## Plan deviations (all with regression tests)
1. `test_v12_feature_factory` marker fixed to match real v8 `POLY_NAMES` order (`ring_count` between `ring_density` and `rigidity`).
2. `test_v12_no_trained_gate` uses `\bgate\b` word-boundary regex — plan's `"GATE" in code.upper()` false-positived on `GetIsConjugated`.
3. Smoke caught a real `NameError` (`km` unpacked as `_`); fixed to `km, labs, labs_te, idx = cluster_assignment(...)`.

## Artifacts
- `build_v12_kaggle_nb.py` (22-cell generator), `tests/test_v12_kaggle_nb.py` (11 tests, all pass), `PolyWin_R2_v12_bucket_moe.ipynb`.
- Smoke (cached experts): `vault/pipeline_out_v12_smoke/` — bucket-MoE mean OOF 0.8605 < v11 0.8628 → fallback engaged correctly.
- Full Kaggle run: kernel `shubhamkambli11/polywin-r2-v12-chemistry-bucket-moe`, outputs in `vault/kernel-v12-output/`.

## Kaggle full-run results (10-fold, honest OOF)

`v12_bucket_compare.csv`:

| target | stack_oof | gnn_oof | v11_blend_oof | v12_bucket_oof | K | mean_w |
|--------|-----------|---------|---------------|----------------|---|--------|
| tg     | 0.9020    | 0.8230  | 0.9020        | 0.9021         | 2 | 0.975  |
| egc    | 0.9085    | 0.8309  | 0.9081        | 0.9090         | 4 | 0.907  |
| egb    | 0.8937    | 0.9079  | 0.9163        | 0.9156         | 2 | 0.403  |
| eps    | 0.7364    | 0.7705  | 0.7778        | 0.7767         | 2 | 0.323  |
| nc     | 0.8304    | 0.8065  | 0.8468        | 0.8460         | 2 | 0.562  |
| ei     | 0.7763    | 0.7757  | 0.7939        | 0.7923         | 2 | 0.508  |
| eea    | 0.8535    | 0.8776  | 0.8929        | 0.8966         | 4 | 0.354  |

Mean OOF: stack 0.8430 | gnn 0.8274 | v11 blend 0.8625 | bucket-MoE 0.8626.

Success criteria verdict: **Moderate** — bucket beats v11 on 3/7 targets (tg +0.0001, egc +0.0009, eea +0.0037); 4 targets regress (egb −0.0007, eps −0.0011, nc −0.0008, ei −0.0016), worst −0.0016 (within −0.003). Mean OOF ≥ v11 → notebook emitted the **bucket-MoE submission**.

Submission `vault/kernel-v12-output/submission.csv` (4940 rows, physics bounds verified: egc/egb/ei ≥ 0, eps ≥ 1, nc ∈ [1,3]) submitted.

## Leaderboard

- **v12 bucket-MoE: 0.849 public** (ref 55305403)
- v11 blend: 0.852 (ref 55286407) ← best standing submission
- v10 pretrained GNN: 0.830
- v8 pseudo-label stack: 0.828
- v6 stack: 0.847
- v4 baseline: 0.828

## Verdict / next steps

- v12's marginal OOF gain (0.8626 vs 0.8625) did **not** transfer to the LB (0.849 vs 0.852). **Do not adopt bucket-MoE as the standing submission**; v11 blend (0.852) remains the best.
- Bucket structure exists in the diag (e.g., eea cluster 1 w_stack 0.017 → nearly pure GNN; egb cluster 0 w 0.31) but the per-cluster gains are too small to beat the scalar blend on the leaderboard.
- Phase C (soft router, v13) gate: v12 shows weak structure but no LB gain — defer until a stronger signal (graph-embedding routing / trainable gate) is justified. Keep v11 blend as the submission to beat.
- LB table in `build_pipeline_nb.py`/`AISEHack_Round2_Pipeline.ipynb` (`20_lb_progression.png`) updated with v8–v12.
