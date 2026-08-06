# PolyWin R2 — v12 Chemistry Bucket MoE (Design)

Date: 2026-08-06
Status: Approved in brainstorm (awaiting spec review)
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v12 exists

The v10/v11 experiments established the blend ceiling:

| Model | Mean OOF R² |
| ----- | ----------- |
| v6 Stack (rebuilt) | 0.844 |
| v10 Pretrained GNN | 0.838 |
| v11 target-wise blend | **0.861** |

`vault/pipeline_out/gnn_arm/v10_blend_compare.csv` shows the fold-safe target-wise
blend beats the stack alone on **all 7 targets** (tg 0.9045, egc 0.9083, egb 0.9165,
eps 0.7484, nc 0.8519, ei 0.8021, eea 0.8972). So Phase A is **proven**: the pretrained
GNN carries orthogonal signal, and a linear combination is worth up to ~+0.017 OOF.

v12 is the first **chemistry-aware** experiment (Phase B in the approved roadmap).
The hypothesis:

> Some chemistry families prefer the GBM stack, others prefer the pretrained GNN.
> A per-target scalar `w` (v11) averages over this structure; a **per-bucket** `w`
> should exploit it — with almost zero overfitting risk because the routing is a
> fixed clustering + a coarse weight grid, not a trained gate.

Deliverable (user-approved): **one single end-to-end Kaggle notebook** that rebuilds
the GBM stack, pretrains the GNN on PI1M, applies the chemistry bucket MoE, and emits
a submission — no dependency on local caches.

## 2. Baseline & success criteria

- Reference baseline: **v11 target-wise blend OOF (0.861 mean)**, plus the standing
  v6 submission (0.847 LB) and the v11 blend submission as LB references.
- **Locked success criteria (pre-registered):**

  | Outcome | Condition |
  | ------- | --------- |
  | Strong success | Bucket MoE beats v11 blend on ≥4 targets in honest OOF, and no target regresses by more than −0.003 |
  | Moderate success | Bucket MoE beats v11 blend on 2–3 targets in OOF |
  | Failure | Bucket MoE beats v11 blend on ≤1 target → ship v11 blend; move to Phase C (soft router) only if v12 shows structure |

- Primary diagnostic: per-target table `v12_bucket_compare.csv` —
  `stack_oof`, `gnn_oof`, `v11_blend_oof`, `v12_bucket_oof`, `w_used`.
- Secondary diagnostics: per-(target, cluster) cell count + chosen `w`, to show
  *which* chemistry families prefer GNN vs stack.

## 3. Architecture (v12)

### 3.1 One notebook, four stages

`PolyWin_R2_v12_bucket_moe.ipynb` (built by `build_v12_kaggle_nb.py`) is fully
self-contained:

1. **Data + features** (reuse v8 pipeline Layer 1–2): load train/test from
   `/kaggle/input`, canonicalize, dedup, GroupKFold (10 folds), RDKit descriptor +
   fingerprint + polymer-physics + fragment features → `Xtr`/`Xte`.
2. **GBM stack** (reuse Layer 4 + Layer 9): 4 GBMs per target (LGB/Cat/XGB/HistGB)
   with fold-safe OOF, L1.5 Ridge, L2 meta (reliability + cross-target) → stack
   OOF/test. This is the "tabular expert".
3. **Pretrained GNN** (reuse v10 pretrain notebook): GINE encoder pretrained on PI1M
   (masked atom/feature reconstruction), fine-tuned per target fold-safely → GNN
   OOF/test. This is the "graph expert".
4. **Chemistry bucket MoE** (new): cluster molecules per target into K buckets on
   curated chemistry features, run v11's fold-safe weight grid *inside each cluster*,
   blend, submit.

### 3.2 Routing features (curated chemistry + expert disagreement)

The bucket features are the user-approved "expert disagreement + chemistry" set,
subset from the feature factory so the router does not reproduce the stack:

- Chemistry (from `polymer_physics` + descriptors): `MolWt`, `ExactMolWt`,
  `HeavyAtomMolWt`, `ring_density`, `arom_ratio`, `hetero_density`,
  `halogen_density`, `sulfur_density`, `flexibility`, `rigidity`, `logp`,
  `hbd_density`, `hba_density`.
- Expert disagreement: `|stack_pred − gnn_pred|`, `stack_pred`, `gnn_pred`
  (only for diagnosis/interpretation, not part of the clustering input if it risks
  instability — see 3.3).

### 3.3 Bucketing scheme (per target, fold-safe)

Per target `tt`:

1. Take the target's training rows and the **chemistry feature subset** above,
   standardize (train-only fit), and run **KMeans** with `K ∈ {2, 3, 4}`.
   Choose K by the fold-safe OOF of the *bucket blend itself* (not by silhouette)
   so the selection is honest.
2. **Fold-safe weight search inside each cluster** (mirrors v11):
   for each cluster, grid-search `w ∈ [0,1]` on the training folds' OOF pairs
   `(stack_oof, gnn_oof)` and apply to the held-out fold. Rows in a cluster are
   routed with that cluster's `w`.
3. Test rows: assign to the nearest cluster centroid (KMeans `predict`), apply the
   same per-cluster `w` (weight averaged over folds, as in v11).

K is shared across folds for a given target (fit once on all of that target's rows —
clustering is unsupervised so this is not label leakage; the *weights* are still
fold-safe).

### 3.4 Blend & submission path

- `final_tt(row) = w_tt,cluster(row) * stack_tt(row) + (1 − w_tt,cluster(row)) * gnn_tt(row)`.
- On success, write the blended `submission.csv` (standard physics bounds: clip
  egc/egb/ei ≥ 0, eps ≥ 1, nc ∈ [1,3]).
- If Bucket MoE fails, the notebook still emits the **v11 blend submission**
  (single target-wise `w`) so a valid submission always exists.

### 3.5 Compute strategy & runtime

- Dominant costs: feature factory (~8 min), GBM stack (~30–40 min), PI1M pretrain +
  GNN fine-tune (~2–2.5 h, same budget as v10), bucketing + weight grid (seconds).
  Projected total ≈ **3–3.5 h** (well under the 9–12 h kernel limit).
- Hard caps: KMeans K ≤ 4; weight grid = 21 points; no unbounded loops. Smoke mode
  (`POLYWIN_SMOKE=1`) reduces folds/epochs and runs the bucket stage on a
  small sample to keep local smoke fast.

### 3.6 Artifact persistence

- `v12_bucket_diag.csv` — per (target, cluster): `n`, `stack_oof`, `gnn_oof`,
  `blend_oof`, `w_stack`.
- `v12_bucket_compare.csv` — per target: `stack_oof`, `gnn_oof`, `v11_blend_oof`,
  `v12_bucket_oof`, chosen `K`, mean `w`.
- `gnn_oof.csv` / `gnn_test.csv` (regenerated by the GNN stage), `submission.csv`.

### 3.7 Figures

v12 adds one centerpiece figure:

| Fig | File | Content | Purpose |
| --: | ---- | ------- | ------- |
| 24 | `24_bucket_moe.png` | grouped bars per target: stack / gnn / v11-blend / bucket-MoE OOF R² | **centerpiece**: does chemistry-aware routing beat the scalar blend? |

## 4. Roadmap

| Version | Workstream |
| ------- | ---------- |
| v11 | Target-wise blend (DONE, proven: 0.861 OOF) |
| v12 | Chemistry Bucket MoE (this design) |
| v13 | Soft router MoE (Approach 1: Ridge-logistic/tiny-MLP gate on disagreement + chemistry) — only if v12 shows structure |
| v14 | Embedding router (Approach 3) — only if v13 insufficient; requires re-exporting latent embeddings |

## 5. Deferred to v13+ (explicitly out of scope for v12)

- Trainable soft gate (per-molecule `p(stack)` from a learned model).
- Graph-embedding routing (needs new `latent_embeddings.npy` cache).
- Rule-based hand-tuned bucket definitions (KMeans is the default; rule-based is a
  fallback only if KMeans clusters look uninterpretable).

## 6. Validation plan

1. Local smoke run (`POLYWIN_SMOKE=1`) — bucket stage regression on cached
   `moe_gbm_chk.parquet` + v10 GNN cache to validate the bucketing/blend logic
   quickly before the full Kaggle run.
2. Tests: extend `tests/` — bucket stage (chemistry subset columns exist, KMeans K∈
   {2,3,4} selection, fold-safe per-cluster weight, test-cluster assignment, emission
   of `v12_bucket_compare.csv` and submission). Existing tests still pass.
3. One Kaggle run of the end-to-end notebook. Report `v12_bucket_compare.csv` +
   fig 24, then submit if Strong/Moderate and compare LB vs v6 (0.847) and v11.
