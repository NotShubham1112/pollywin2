# PolyWin R2 — v22 In-Notebook SMILES Transformer Arm (Design)

Date: 2026-08-11
Status: **DESIGN (pre-registered) — not yet implemented.**
Competition: AISEHack 2.0 Polymer Property Prediction Round 2 (`ppp-round-2` on Kaggle)

## 1. Why v22 exists

P14 (full-PI1M pretrain, public LB **0.883**, honest OOF **0.8641**) is the frozen final
submission. The post-mortem (`vault/final_status.md`) closed every later attempt v15–v21:

- v15 loss-reweighting: **FAIL** (−0.0051) → STOP.
- v16/v17/v18/v19 sibling/cross-target family: **FAIL on LB** (0.874/0.862/0.864/0.874) —
  root cause: pivot built from full-train true labels reused inside val folds → leakage.
- v20 self-trained SMILES transformer (d=256, 4 layers, char vocab 1.6k, 2 epochs on 20k PI1M):
  **local gate FAIL** — 3rd arm near-collinear with GBM+GNN (mean Δ −0.0004, nc −0.0033).
  Its two Kaggle runs errored before training (pip rdkit / input-path assert), so its true
  LB is **unknown**; only the local gate report exists.
- v21 leak-safe sibling arm: local gate FAIL (eps/nc/ei mean −0.0012), Kaggle LB **0.867** < P14.
  Cross-target sibling signal is closed.

v22 re-tries the **self-supervised SMILES representation arm**, but at materially larger
scale than v20 so the "saturation" failure mode is directly addressed:

- **Tokenizer:** in-notebook BPE on a stratified ~150k PI1M subset → target ~4k tokens
  (v20: 1.6k character-level vocab). BPE captures recurring monomer/motif tokens instead
  of character fragments.
- **Model:** `nn.TransformerEncoder`, **d=384, 6 layers, 8 heads** (~13M params), MLM
  pretraining on the **full 995k PI1M** corpus (v20: 20k rows, 2 epochs). Runtime
  ~30–45 min on P100.
- **Arm:** fold-safe per-target Ridge heads on frozen mean-pooled embeddings (same
  `compute_trf_arm` structure as v20, already leak-safe and tested).
- **Blend:** P14 2-arm reference recomputed in-cell; **3-arm** `Ridge([gbm, mt, bert])`,
  same fold-safe per-target alpha sweep as P14/v21.
- **Gates:** identical pre-registered gates 0–3 as v21; **FAIL → P14 (0.883) stays final**,
  no submission written.

The whole thing runs inside one notebook from train/test/PI1M only — no external data, no
uploaded artifacts, no rules violation.

## 2. Rules scope (pre-audited)

- No external data, no uploaded artifacts: everything (incl. PI1M pretraining) is produced
  inside one notebook run. **No frozen external transformer embeddings** (ChemBERTa-2 /
  MoLFormer violate this). v22 trains its own encoder in-notebook from PI1M.
- No true train labels in any feature path. The encoder is trained **unsupervised** (MLM)
  on PI1M; embeddings carry no label information by construction.
- Notebook-only, pinned-version, host-shared submission.
- **Out of scope / rejected:** frozen external transformer embeddings, GHM/loss-reweighting
  (falsified by v15), non-linear Level-2 meta (falsified by v7/v12), sibling/cross-target
  arms (closed by v21), seed-scaling heterogeneous GNNs (contradicts blend ceiling).

## 3. The single change

A **self-supervised SMILES transformer arm** added as a third column in P14's per-target
Ridge blend. GNN, GBM trio stack, folds (GroupKFold on canonical SMILES), GNN seeds
42/999/2025, descriptors, pretrained GINE encoder, P14 submission path are all
**bit-identical to P14**. Only the blend widens `Ridge(X=[GBM, MT, BERT])` (with P14's
2-arm reference recomputed in-cell for like-for-like gate 2). P14 = 2 arms; v22 = 3 arms.

**Tokenizer (`v22_tokenizer.py`).**
- Learn BPE merges on a **stratified ~150k-subset of PI1M** canonical SMILES
  (`learn_bpe`): char-level init vocab (~1k) → target ~4k merges, fixed seed → deterministic.
- `encode(smiles) -> token ids`, `decode(ids) -> smiles` round-trip for survivable rows.
- `*` (polymer attachment), `(`, `)` (branching) are **protected tokens** — never split and
  never masked during MLM, so structural topology survives.
- Pure functions, no RDKit dependency in the tokenizer path (works on raw SMILES strings).

**Encoder (`v22_encoder.py`).**
- `BertEncoder`: d=384, 6 layers, 8 heads, learned positional embeddings, MLM head,
  ~13M params. Masking ~15% of maskable tokens (excluding protected/special tokens).
- `pretrain_mlm`: AdamW, cosine LR, best-val checkpoint on a held-out PI1M slice
  (~5k rows). Full-corpus pass over 995k SMILES, batched for P100.
- `pool_embeddings`: mean-pool of final-layer hidden states over non-`[PAD]` tokens.
- CPU fallback: if no GPU, train a reduced config (like P14's `if DEVICE == "cpu"` path).

**Arm (`v22_arm_cv.py`).**
- `compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g, n_splits=5, seed=42)` — per-target
  fold-safe Ridge heads on frozen embeddings; **verbatim reuse of v20's `compute_trf_arm`
  structure** (GroupKFold on canonical SMILES, per-target Ridge, OOF + fold-bagged test).
- Outputs `bert_oof` (7409×7 aligned to train rows) and `bert_test` (4940×7).

**Blend (`v22_blend.py`).**
- Generalize v21's `blend_3arm_oof` to n arms (`blend_narm_oof`): per-target GroupKFold
  alpha scan over `ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]`, refit at best alpha.
- P14 2-arm reference = `blend_narm_oof` on `[gbm, mt]` (identical to `_p14_2arm_oof`).
- 3-arm = `[gbm, mt, bert]`. Per-target `w_BERT` reported (Ridge shrinks collinear arm → 0).

**Gate report (`v22_gate_report.py`).**
- Gates 0–3 (identical thresholds to v21, see §5), `GATE: PASS/FAIL`, submission written
  **only on PASS**. Report written to `vault/pipeline_out_v22/v22_gate_report.csv`.

## 4. Leak-safety (by construction + audited)

- The encoder is trained **unsupervised** on PI1M only (MLM) — no train labels, no folds.
  Embeddings are label-free by construction.
- `compute_bert_arm` uses a single GroupKFold on canonical SMILES; the Ridge head for a
  val-fold row never sees that fold's labels. Same fold-safety proof as v20/v21.
- **Audit (gate 1):** v19-style leak-eligibility check — for every val fold, exact-match
  count between any arm feature and the polymer's true labels = **0**. For the BERT arm this
  is trivially 0 (embeddings are continuous, label-free), but the audit still runs over the
  full blend input columns.
- Row-alignment asserts: `target_type_train == train.csv` order, `corr(y_train, target) >
  0.999` (mirror of `run_v20_gate.py`).

## 5. Baseline & success criteria (pre-registered)

Reference: **P14** (`vault/pipeline_out_pretrain/superblend_oof.npz`), same offline
compare protocol as v21 (`_p14_2arm_oof`, GroupKFold(5) on canonical smiles).

- **Gate 1 — leak audit:** exact-match count = **0** across all val folds. Non-negotiable.
- **Gate 0 (diagnostic, informational) — BERT-only signal probe:** per-target
  `bert_only_r2[t]` = OOF R² of the BERT arm Ridge alone (own GroupKFold). If ≈ 0 on
  eps/nc/ei, the arm is dead before blending — strong stop signal.
- **Gate 2 — OOF gain (tiered):**
  - **Soft success** (proceed to a Kaggle run, eyes open): blend OOF mean over
    **{eps, nc, ei}** ≥ P14 + **+0.0015** **and** overall blend OOF mean ≥ P14 + **+0.0015**.
  - **Strong success** (confident submit): ≥ **+0.003** on both (reported, not gating).
- **Gate 3 — worst-target guard:** no per-target OOF regression > **−0.003** vs P14.
- **PASS condition (identical to v21):** Gate 1 = 0 leaks **and** Gate 2-soft **and** Gate 3.
- **Pre-registered expectation:** prior is v20's near-zero mean Δ at much smaller scale;
  v22's honest OOF mean is most likely **[0.864, 0.867]** (i.e. Δ −0.000 to +0.003); a
  strong-tier (+0.003) result would be an upside surprise.
- **Confirmatory (not gated):** public LB ≥ **0.886** (top-20 zone) on the Kaggle run; the
  notebook's own gate report must match the local gate report before the score is trusted.
- **Fail → STOP:** gate 1 fails, or gate 3 fails, or gate 2 fails at **both** tiers → keep
  **P14 (0.883)**, no v22 slot spent. Record the numbers; do not re-tune gates post-hoc.
- **No pseudo-labeling. No true-label features. No test-row train-label lookup. No
  architecture change to P14 level-0. No new libraries beyond what v20/v21 already use
  (torch, torch_geometric, rdkit, sklearn, lightgbm, xgboost, catboost).**

## 6. Deliverables

All v22 artifacts land in **`d:\Parth\ploywin r2\New folder\`** at implementation time
(user decision). The design spec stays in `docs/superpowers/specs/`.

- `v22_tokenizer.py`, `v22_encoder.py`, `v22_arm_cv.py`, `v22_blend.py`,
  `v22_gate_report.py` — modules (BPE, MLM encoder, fold-safe arm, n-arm blend, gates).
- `run_v22_gate.py` — **local gate harness** (CPU, minutes): tokenizer + encoder on a small
  PI1M slice, arm OOF + `bert_only_r2` diagnostic, blend vs cached P14 OOF, gates 0–3.
- `build_v22_kaggle_nb.py` (fork of `build_v20_kaggle_nb.py` + v22 module cells) →
  `PolyWin_R2_v22_bert_arm.ipynb` (full) and `PolyWin_R2_v22_bert_arm_smoke.ipynb`.
- `tests/test_v22_*.py` (TDD): tokenizer round-trip + protected tokens, MLM loss decrease,
  arm shapes/alignment, gate boundaries (exact thresholds pass, one step below fails),
  notebook build + CORE bit-identity to v14 + forbidden-refs + smoke subset.
- Smoke run locally, then full Kaggle P100 run, download, evaluate vs P14, decide per §5.

## 7. Error handling

- **Canonicalization failure / malformed SMILES** → `[PAD]`-only token; embedding = learned
  `[CLS]`-pool fallback (no crash on a bad test row).
- **BPE on too-large corpus** → learn merges on stratified ~150k subset (vocab stabilizes
  before 1M); full corpus only used for MLM.
- **No GPU** → reduced encoder config via env knobs (`V22_D`, `V22_LAYERS`, `V22_PI_COUNT`,
  `V22_EPOCHS`), same pattern as v20.
- **Collinear BERT arm** → `w_BERT ≈ 0`, blend ≈ P14 (safe by construction); `w_BERT`
  printed per target.
- **Row alignment** → asserts (npz == train/test order, `corr > 0.999`), mirror of v20/v21.

## 8. Result (to be filled after run)

- [ ] Local gate harness: `bert_only_r2` diagnostic (gate 0) + gates 1–3 pass/fail (numbers per tier).
- [ ] Notebook smoke run: cells compile, smoke subset matches harness.
- [ ] Kaggle full run: kernel URL, gate report, public LB, verdict.
