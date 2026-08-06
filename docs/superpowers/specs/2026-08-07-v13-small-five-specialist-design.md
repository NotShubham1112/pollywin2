# PolyWin R2 — v13 Small-Five Specialist (Design)

Date: 2026-08-07
Status: Approved in brainstorm (awaiting spec review)
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)
Deadline: ~6 days from this design (final submission imminent; private LB decides rankings)

## 1. Why v13 exists

v11/v12 established that incremental blending has hit a wall. v12 (chemistry bucket MoE)
earned **LB 0.849**, *worse* than the v11 blend's **0.852** — the marginal OOF gain
(+0.0001) did not transfer to the leaderboard. The current best standing submission is
the **v11 blend (0.852)**.

The leaderboard gap analysis is decisive:

| target | v11 OOF R² | train rows | gap to 0.90 |
| ------ | ---------- | ---------- | ----------- |
| tg     | 0.902      | 4143       | ✓           |
| egc    | 0.909      | 2028       | ✓           |
| egb    | 0.916      | 337        | ✓           |
| **eps**| **0.777**  | 229        | **+0.123**  |
| **nc** | **0.846**  | 229        | **+0.054**  |
| **ei** | **0.792**  | 222        | **+0.108**  |
| eea    | 0.897      | 221        | +0.003      |

The metric is the **unweighted mean R² across all 7 targets** (each target contributes
equally regardless of sample size). The big three (tg/egc/egb) are already ≥0.90. **The
entire gap to a top-10 finish (≥0.90 mean R²) is in the under-sampled targets — eps, ei,
nc are the real problem (gaps +0.12/+0.11/+0.05); eea (+0.003) and egb (−0.016) are
already strong but can be pushed toward 0.92+.** Each of these five has only ~220–340
train rows; the architecture below treats all five ("small five") as one specialist so
the 415 multi-labeled polymers and the big targets' volume help all of them together.

Two exploitable facts discovered during the brainstorm:

1. **Cross-target leakage (legal, train-labels-only):** 92% of small-five test rows have
   their exact polymer present in train under *other* target types (eps 148/153, ei
   142/148, nc 148/153, eea 141/147, egb 180/224). For those polymers we know 1–5 of
   their other 7 property values from train.
2. **Physics relations hold strongly in our own data:**
   - `corr(egc, ei − eea) = 0.995` (fundamental gap = IP − EA), mean `|egc−(ei−eea)| = 0.093`
   - `corr(egb, egc) = 0.926`, `mean(egb−egc) = −0.10` (bulk gap below chain gap)
   - `corr(nc², eps) = 0.925` (Maxwell relation for refractive index / dielectric)

Direct physics imputation alone gives R² 0.85–0.96 on ei/eea/egb but only 0.34/0.17 on
eps/nc (narrow variance → imputation noise dominates); cross-target-as-features alone
gives R² 0.48–0.78. So **neither alone wins — a multi-task specialist that (a) lets the
big targets teach a shared trunk and (b) consumes known-other-target values as features
is the right design.**

Deliverable (user-approved): one single end-to-end Kaggle notebook that builds a
**multi-task specialist** (PI1M-pretrained GINE trunk + 7 heads + physics-consistency
auxiliary losses) consuming **fold-safe leakage features** and **physics imputations**,
then **blends per-target** with the existing GBM stack / GNN / v11 blend.

## 2. Baseline & success criteria

- Reference baseline: **v11 target-wise blend OOF (0.8625 mean)** and the standing v11
  blend submission (**0.852 LB**), plus the v12 bucket-MoE result (**0.849 LB**).
- **Locked success criteria (pre-registered):** "small-five mean" = unweighted mean R² of
  the five specialist targets {eps, nc, ei, eea, egb}, compared fold-for-fold against the
  v11 blend's per-target OOF.

  | Outcome | Condition |
  |---------|-----------|
  | **Strong** | small-five mean OOF R² improves ≥ +0.03 over v11 blend (i.e., mean 0.846 → ≥0.876) AND no big-three target regresses > −0.003 |
  | **Moderate** | small-five mean OOF R² improves ≥ +0.01 AND no target regresses > −0.003 |
  | **Failure** | any target regresses > −0.003, or small-five mean fails to beat v11 → blend falls back to v11 candidates only |

- Decision rule: the fold-safe per-target blend **cannot regress below the best single
  candidate** on any target (weight search is free to pick weight 1 on the incumbent),
  so the standing submission is guaranteed ≥ v11 blend OOF. The notebook's runtime
  choice is `USE_SPECIALIST = mean_specialist_blend_oof >= mean_v11_blend_oof`.
- LB target: ≥ 0.90 mean R² (top-10 qualification line is ~0.897 as of 2026-08-07).

## 3. Architecture

```
SMILES → canonicalize/dedup → GroupKFold (key: canonical SMILES)
   │
   ▼
v10 PI1M-pretrained GINE trunk (last 2 MP layers unfrozen) ──► graph embedding
   │                                                                │
   ├── Component D: tg, egc heads (embedding only)                  │
   ├── Component C: eps, nc, ei, eea, egb heads                     │
   │                  (embedding ⊕ known_other_targets ⊕ imputed)   │
   └── Component E: physics residuals (in loss, not separate heads) │
                                                                    ▼
   Component F: fold-safe per-target blend
        candidates = {specialist, specialist_no_leakage, leakage_only,
                      stack, gnn, v11_blend, physics-imputed}
        weights tuned on other folds → apply to held-out fold
                                                                    ▼
   submission.csv (physics bounds enforced)
```

**Component A — trunk:** v10 GINE (PI1M SSL). Freeze all but the last 2 message-passing
layers + embedding projection. Multi-task fine-tune so tg/egc's 6000+ rows reshape the
embedding for the small five. If the pretrained checkpoint is unavailable (smoke), the
specialist is **skipped** and the blend uses `{stack, v11, imputed}`.

**Component B — fold-safe leakage features:** pivot `smiles → {target_type: value}`.
Train-row features are computed from a pivot built on **other folds only** (GroupKFold
keeps all rows of a polymer in one fold, so this is clean). Test-row features use the
full-train pivot (92% small-five coverage). Missing values → target-wise train mean +
boolean `known_{target}` mask. No train-label leakage into OOF.

**Coverage asymmetry (measured):** train has only **6%** multi-labeled polymers
(415/6565); test has **92%** small-five coverage. So learned leakage features are rarely
seen in training — the model must learn to exploit them from just 415 rows, and the
**physics-imputed candidate (no training needed)** carries most of the leak-exploitation
load at test time. The learned leakage features are a secondary, additive signal.

**Component C — small-five heads:** input `[embedding, known_other_targets (filled),
physics_imputed]`, MLP heads (one per small target: eps, nc, ei, eea, egb). The model
learns to use leakage features where present and ignore them where absent (they become
train means). **Also produced: `specialist_no_leakage`** — the same heads fed the
embedding only, so the blend can downweight leakage-fed predictions on the ~8% of
non-leaked small-test rows.

**Component D — big-two heads:** input `embedding` only (tg/egc are 0.3–5% leaked, no
leakage features).

**Component E — physics residuals (direct losses, no extra heads):**
`physics_loss = mse(pred_egc, pred_ei − pred_eea) + mse(pred_eps, pred_nc²) +
mse(pred_egb, pred_egc − Δ)` where Δ is a single learned scalar. Applied only on rows
where all involved labels are known — sample counts are tiny: 10 / 134 / 82 polymers —
so the total weight is **0.05–0.1** (guide, not dominate). No separate auxiliary
networks, no inference-time penalty.

**Leakage-only baseline (required ablation, run before the specialist):** a CatBoost /
LightGBM per small target using only `{known other targets, physics imputations}` — no
trunk, no neural net. **Measured expectation (fold-safe):** R² eps 0.51, nc 0.62, ei
0.34, eea 0.45, egb 0.58 — *below* the v11 blend per-target, so it is an additive blend
candidate, not a replacement. It also diagnoses how much of the gain is information
transfer vs. representation learning. Runs in minutes on the 415-row multi-labeled set.

**Component F — blend:** per-target fold-safe weight search over the **seven candidates**
{specialist, specialist_no_leakage, leakage_only, stack, gnn, v11_blend, physics-imputed},
exactly the v11/v12 protocol (tune on other folds, apply to held-out). Imputed candidate
carries weight where coverage is strong (eea, ei, egc); for tg/egc specialist+stack
dominate. Guaranteed floor: weight search can pick the incumbent candidate.

## 4. Training details

- Multi-task loss: weighted per-target MSE `[tg:1, egc:1, egb:1.5, eps:3, nc:2.5, ei:3,
  eea:2]` — small five weighted up ~2–3× so they get enough gradient despite 220 vs
  4143 rows. **Physics residuals added at total weight 0.05–0.1** (Component E), so they
  guide the small-five heads without dominating the tiny labeled subsets.
- Optimizer: AdamW, `lr=3e-4`, batch 128. 10-fold GroupKFold, same seed/protocol as
  v11/v12 → honest OOF.
- Smoke flags inherited from v12: `SMOKE=1 → 3 epochs, 1 pretrain epoch, 2000-row cap` so
  the smoke run finishes in < ~15 min on the cached path and validates the full notebook
  end-to-end without a Kaggle GPU run.
- Test-time: full-train pivot for leakage features; `specialist_oof` + `specialist_test`
  feed the blend. The leakage-only baseline runs before the specialist to set the
  information-transfer expectation.

## 5. Data flow

1. Canonicalize SMILES (strip `*`/`[*]`), dedup by `smiles+target_type`, GroupKFold.
2. Build per-fold leakage pivot + physics imputation tables.
3. Run the **leakage-only baseline** (CatBoost/LGBM per small target) → ablation CSVs,
   sets the information-transfer expectation.
4. Train multi-task specialist (Components A–E) → specialist_oof/test + specialist_no_leakage.
5. Fold-safe per-target blend (Component F) over the seven candidates.
6. `USE_SPECIALIST` decision; submission with physics bounds (egc/egb/ei ≥ 0, eps ≥ 1,
   nc ∈ [1,3]).

## 6. Error handling

- GINE checkpoint missing → specialist skipped, blend falls back to `{stack, v11,
  imputed, leakage_only}`. Never a crash, never a NaN submission.
- A fold with insufficient multi-labeled polymers for physics residuals → those
  residuals are masked for that fold (not skipped globally).
- `SMOKE=1` → all heavy loops capped; full pipeline must complete end-to-end.

## 7. Testing (extend the v12 suite)

- Source markers: multi-task trunk + 7 heads present; leakage features built from
  other-fold pivot only; physics residual losses in training loss (egc=ei−eea, eps=nc²,
  egb=egc−Δ) at total weight ≤ 0.1; blend candidate list includes specialist,
  specialist_no_leakage, leakage_only + imputed; no trained gate (`\bgate\b`
  word-boundary regex); submission physics bounds; per-cell `ast.parse` compile.
- New tests: leakage-pivot fold-safety (no same-fold label in features), physics-impute
  R² sanity on multi-labeled polymers, blend-candidate list (≥7 candidates),
  leakage-only baseline runs before specialist, USE_SPECIALIST fallback.
- Smoke run must pass end-to-end on cached experts (validates notebook before Kaggle).

## 8. Deliverables

- `build_v13_kaggle_nb.py` (single end-to-end Kaggle notebook generator, 7–component
  pipeline) + `PolyWin_R2_v13_specialist.ipynb`.
- `tests/test_v13_kaggle_nb.py` (extend v12 test patterns).
- Smoke run output in `vault/pipeline_out_v13_smoke/` (not committed).
- Push to Kaggle (`vault/kernel-v13-push/`), run, download outputs to
  `vault/kernel-v13-output/`, score vs v11, decide submission per criteria.
- LB progression table + session doc updated.
