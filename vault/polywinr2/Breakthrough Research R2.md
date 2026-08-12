# PolyWin Round 2 — Breakthrough Architecture Research

Principal-level gap analysis, literature mining, architecture discovery, score analysis, and a single top recommendation to move the leaderboard from ~0.883 (P14) toward 0.90+ in ~1 week inside a single notebook.

> All numbers below were re-measured from `official_dataset/` in this repo — not assumed.

---

## 0. Verified ground truth

### 0.1 Task structure
Long-format CSV: `smiles, target, target_type`. 7 targets. Metric = **unweighted mean R² across all 7 targets**, so a 220-row target scores with the same weight as the 4,143-row Tg.

| target | train | test | P14 blend OOF | P14 GBM OOF | P14 GNN OOF | meaning |
|---|---|---|---|---|---|---|
| tg   | 4143 | 2763 | **0.902** | 0.898 | 0.877 | glass transition |
| egc  | 2028 | 1352 | **0.907** | 0.897 | 0.893 | chain bandgap |
| egb  |  337 |  224 | **0.925** | 0.884 | 0.924 | bulk bandgap |
| eps  |  229 |  153 | **0.801** | 0.736 | 0.749* | dielectric constant |
| nc   |  229 |  153 | **0.866** | 0.796 | 0.872 | refractive index |
| ei   |  222 |  148 | **0.824** | 0.808 | 0.800 | ionization energy |
| eea  |  221 |  147 | **0.913** | 0.880 | 0.907 | electron affinity |

Mean blend OOF ≈ **0.877**; current public LB **0.883** (P14, shipped). eps is the single worst target.

### 0.2 Physics, empirically confirmed on this dataset
- **egc ≈ ei − eea**: corr **0.995** on 10 shared polymers.
- **egb ≈ a·egc + b**: corr **0.926** on 82 shared.
- **eps ≈ a·nc² + b**: corr(nc², eps) **0.925**; a Ridge `eps=f(nc²)` alone gives OOF R² **0.848** on the 134 labeled pairs — the physics path alone beats the current eps blend (0.801) by **+0.047**.

### 0.3 THE decisive discovery — test–train polymer overlap
| test target | n | exact-smiles present in train | sibling-target labels present in train |
|---|---|---|---|
| eps | 153 | **148 (96.7%)** | 95 with `nc` |
| nc  | 153 | **148 (96.7%)** | 95 with `eps` |
| ei  | 148 | **142 (95.9%)** | 98 with `eea` |
| eea | 147 | **141 (95.9%)** | 98 with `ei` |
| egb | 224 | **180 (80.4%)** | 63 with `egc` |
| egc | 1352 | 72 (5.3%) | 14 with {ei, eea} |
| tg  | 2763 | 8 (0.3%) | – |

Same-target duplication is essentially zero (no leakage rows). **For the 5 small targets, ≈96% of the test polymers already appear in train under a *different* target.** So the generalization problem on the small heads is not "predict unseen chemical space," it is "recover the transferable part of the correlated physics and predict the target-specific residual." This is a fundamentally different problem from Tg/egc, and it caps the ceiling of any independent-head paradigm.

**This reframes the competition:** the 5 small targets are a *multi-view regression with highly correlated views*, not 5 independent low-data tasks. The current pipeline treats them as independent heads (GBM, GNN, ridge) with a bolted-on "sibling Ridge" (v16/v17) that was falsified offline-then-LB. The breakthrough is to make **cross-target transfer a first-class architectural object**, evaluated fold- honestly (see Task 5, §8).

---

## Task 1 — Gap analysis

### 1.1 What the current systems capture
2D structure (RDKit), 4 fingerprints (Morgan, MACCS, AP, TT), ~3,300 route/descriptor features, per-target z-scoring, 1/freq sample weighting, PI1M encoder init, per-target heads. Model vocabulary: GBDTs (lgb/xgb/cb) + GINE GNN, per-target Ridge blend. Honest GroupKFold on `canon`.

### 1.2 What they fail to capture (ranked by estimated residual impact)

1. **The cross-target identity is modelled as corrupted features, not as structure.**
   The GNN "twin" features are *other targets' LGBM predictions* — noisy on small targets, uncorrelated with the local head's errors, and only fold-safe by construction. The physics identities (egc=ei−eea, eps∝nc², egb∝egc) are near-exact and would let a 200-row head borrow signal that no independent head can learn alone. **Dominates.**

2. **3D/electronic signal is absent.** The small targets are fundamentally orbitals + dielectric. The pipeline carries no conformer features, no quantum descriptors, and only naive RDKit π-conjugation counts. Electronic-structure-grade signal would constrain exactly the targets with the weakest OOF.

3. **One representation family.** Fingerprint GBM + one GINE (atom/bond one-hot). No pretrained SMILES transformer, no graph transformer, no contrastive pretraining on PI1M — PI1M currently only initializes the encoder (v10/v14). **PI1M representation quality is the bottleneck for the small targets, because 200–330 rows cannot learn good chemistry alone.**

4. **Physics gates were measured on the wrong subsets.** v16's gain (+0.0118) was measured on the ~1,297-row subsection of *shared* polymers — clean physics, but not representative of the full LB (Tg/egc have ~0–5% overlap), so the benefit diluted. The correct gate applies the physics layer only where physics holds, evaluated selection-ensured.

5. **No level-2 model couples the 7 targets post-hoc.** The blend is per-target scalar on (GBM, GNN). v12 MoE tried x routing, not target-coupling; v16 tried a decoder but on the shared subset, not as a clean level-2 regressor over all 7 predictions × chemistry.

6. **PI1M pseudo-labelling (v8) failed** because the pseudo labels came from the same low-capacity family used downstream. Representation pretraining, however, was the single biggest win (v14: +0.006 LB over v13).

### 1.3 Single largest source of unexplained error
**The small-target family (eps, nc, ei, eea, egb) — jointly.** Both id-implicit: Tg/egc at ≈0.90; the deficit to a 0.90 mean comes almost entirely from eps (0.80) and ei (0.82). Their errors are (a) 200–330-sample noise, (b) missing orbital/dielectric signal, (c) physics only bolted onto eps/nc in v17. Fixing representation + physics coupling for this family is where practically all remaining headroom sits.

---

## Task 2 — Literature mining (2023–2026)

Evidence-graded shortlist (what matters for THIS data shape).

| Work | Core idea | Why here | Expected Δ (small targets) | Complexity |
|---|---|---|---|---|
| **PolyLLMem** (arXiv:2503.22962; ChemMater 2025) | Fuse an LLM embedding of polymer SMILES (Llama) with GNN; LoRA-adapt to the small set. Beats graph-only models without huge pretraining. | A frozen LLM gives a *pre-trained* chemical feature on 4,096 dims, one `transformers` call away — entirely orthogonal to the GBM/GNN features, and cheap. | +0.02–0.05 | Low–Med (fp16 7B on 1 GPU; cache once) |
| **MIPS / MMPolymer** (KDD'25; CIKM'24) | Pre-train polymer SMILES+graph with *infinite polymer* augmentation — expand `*`-chain to oligomers. | The `*`-decorated PSMILES here are oligomerizable: repeat-unit synthesis turns 200 rows into thousands of views without new data, matching the chain-extrapolated semantics of egc/ eg | +0.03–0.05 | Med |
| **Chemprop + rooted** (J.CIM 2024) | D-MPNN + rooted-SMILES augmentation + ensembles; SOTA on TDC. | Proven on 300-row regression tasks; both augmentation and ensembling built in. | +0.01–0.04 | Low–Med |
| **Graph Transformers (Graphormer/GPS, J.Cheminform 2025)** | GT on 2D with centrality+spatial; QM "atom-in-а-мolecule" pre-training sharpens electronic understanding. | The electronic targets are QM functions of electron density; QM-pretrained attention could beat the 4-layer GINE. | +0.03–0.06 | Med–High |
| **MolE** (NatCommun 2024) | DeBERTa-on-graphs, two-step pretraining (self-sup 850M + supervised 456k). | Proven on few-hundred-sample ADMET tasks. Direct template for 200–330 polymer rows. | +0.04 | Med (their weights) |
| **Meta-MGNN / FS-GNNCNN** (+ transport ML) | Treat each target as a task; gradient meta-learning so small targets borrow shared initialization. | The 5 small targets are exactly the few-shot setting. | +0.02–0.04 | Med–High |
| **Uni-Mol / EPT (3D)** | Structural pretraining on 3D conformers. | Only useful if we generate a conformer of the *repeat unit* (not the whole polymer) — heavy, flaky. | +0.01–0.03 | Med but tight |
| **Contrastive chemical CLR** (MolCLR, GraphCL, masked-atom SSL) | Graph augmentation-based contrast pretraining. | PI1M is already here; v10/v14 used only init. Contrastive upgrade of the same corpus lifts every head. | +0.03 on the small family | Low |
| **Evidential low-data** | Uncertainty-aware prediction. | Mean-R² metric → confidence heads are unnecessary. | – | Skip |

> **Verifiable verdict from this repo's own history:** the biggest real LB gains were (a) PI1M full pre-training (+0.006) and (b) honest ridge blending (+0.01). Every hand-made sibling/decoder/physics-gate **failed to transfer** (v15, v16 –0.009, v17 conditional). Conclusion: **do not bolt physics on as an imputation layer — encode physics in the architecture / loss and train end-to-end with honest CV.**

---

## Task 3 — Architecture discovery (≥ 10 novel candidates)

All notebook-runable and lean on verified repo infrastructure.

1. **TCOND — Target-Conditioned Graph Transformer.** One shared encoder + a learned target token concatenated into the pool; target-dependent head gating. All 7 targets in one model; Tg/egc rows transfer representation into the small heads. Uncommon for polymers.

2. **CTX-MT — Cross-Target Attention Level-2.** A small cross-attention over the *property-token sequence* for each molecule: `[eps, nc, ei, eea, egc, egb, tg]` → predict each property given the others' predictions (with a present/absent mask). Level-2 fusion over the 7 model predictions for the same SMILES.

3. **PhysNet — Physics-Forward GNN.** Electronic head returns `(ei, eea)`; forward computes `egc = ei − eea + b`; dielectric head computes `eps = a·nc² + b` simultaneously (eps/nc share a trunk); bandgap head `egb = c·egc + d`. Physics is *the forward pass*, never a post-hop imputation.

4. **Oligo — Multi-View Chain Graph Fusion (MIPS-style).** Expand `*...*` into n=1,2,4-mer graphs; three views → one embedding. Since egc is the *chain-extrapolated* gap, repeated-unit graphs literally encode the physics of polymerization.

5. **PI1M-SSL — Contrastive + masked-atom pretraining on PI1M**, replacing the v10/v14 init, then the existing fine-tune stack.

6. **RetrX — exact-match sibling retrieval.** For each small-target test row (96% overlap), fetch the sibling-target values for the *same SMILES* from train and feed them to a per-target residual model. Unlike v7 (fingerprint-nearest of target-blind), this is *exact-match sibling + physics* — far better signal.

7. **SibLin — Sibling Fusion Level-2.** Per target, a lightweight model `f(target | all 7 model predictions for the molecule)`, with missing-sibling masking so it fails cleanly when siblings are absent. (This is v16 done right: coupling at level-2, fold-clean.)

8. **MonotonePrior — soft univariate physics prior in blending.** Fit eps↔nc² and egb↔egc on the train, add as a **soft penalty** (not hard replacement) in the final blend where the target is small. Avoids the v15/v16 hard-imputation failure.

9. **EmbedCat — FiLM-conditioned dual-encoder.** ChemBERTa-SMILES embeddings concatenated with GNN pool; FiLM gates per-target heads. Adds a fully orthogonal modality cheaply.

10. **ProtoFS — Prototypical few-shot on pretrained pool.** Compute per-target prototypes; predict test by distance. Surgical few-shot for the 5 small targets.

11. **TwinCov — twin-mean centering.** Center each target on the sibling value (from the 96% overlap) and add a covariance/ranking loss across targets (eps > nc relationship implied). Cheap, orthogonal to any encoder.

12. **3D-ConXL — optional 3D: repeat-unit conformers (ETKDG) → Uni-Mol/DimeNet** as a second arm. Last resort — memory/march heavy.

**No-Go list (with evidence):** PI1M pseudo-labelling (v8 failed); fingerprint-nearest-neighbor retrieval (v7 degraded all targets); hard physics imputation on top of tuned heads (v15/v16/v17 fragile); chemistry-bucket MoE (v12 flat).

---

## Task 4 — Expected score analysis & ranking

| O    | Architecture | Δ small OOF | Δ LB (mean R²) | Risk | Compute | Eng effort | P(success) | Rank |
|---|---|---|---|---|---|---|---|---|
| 1 | **TC-Conditioned Transformer** | +0.03–0.05 | +0.008–0.017 | Med | 2–3 h GPU | 1.5–3 d | 0.7 | ★★★ |
| 2 | **Phys-Forward GNN** | +0.03–0.05 | +0.010–0.020 | Low–Med (only e.g. eps/nc; true physics) | 1–2 h | 1–2 d | 0.7 | ★★★ |
| 3 | **Sibling Fusion L2** | +0.02–0.04 | +0.006–0.015 | Med-High (v16 lesson: fold-correct) | 30 m | 1 d | 0.6 | ★★ |
| 4 | **PI1M Contrastive pre-train** | +0.02–0.04 | +0.008 | Low | 4–8 h GPU | 1 d | 0.8 | ★★ |
| 5 | **Exact-match retrieval** | +0.02–0.07 | +0.005–0.025 | Med (fold-safe leak guard) | 30 m–1 h | 1 d | 0.6 | ★★ |
| 6 | Chemprop/RoBERT embeddings | +0.01 | tie | Med | cheap | 0.5 d | 0.6 | ★ |
| 7 | Meta-learning | +0.01–0.02 | — | High (slow) | 1–2 d | 0.3 | — |
| 8 | 3D-equivariant | +0.01 | — | High (memory) | 2–3 d | 0.4 | — |

**Takeaway:** the winners are **(1) a target-conditioned transformer** and **(3) physics-in-the-forward-gNN** — and they compose into one model (§5). Sibling-level-2 fusion / exact-match retrieval are leaf-blend upgrades that must be fold-honest to avoid re-triggering the v15/v16 trap. Contrastive pretraining is a cheap all-round booster.

---

## Task 5 — Top recommendation

### OPT-A: "TranSPhysGraph" — Conditioned Graph Transformer with physics-coupled electronic head and contrastive pre-training

Fixes the single largest error (small-target family), exploits the 96% test-polymer overlap as a training signal (not a hack), and is notebook-safe.

### Architecture (data/feature flow)

```
PI1M (995k) ── contrative + masked-atom graph pretraining ──────────────┐
                                                                        │ encoder-init / finetune
SMILES ─► RDKit graph ─► 2 views [monomer, n=1, n>2-mer] ─► shared GT trunk
                                                               │
                        pooled embedding P (multi-head pool)   │
      ┌──┬──┬──┬──┬────────────────────────────────────────────┘
      │  │  │  │  │
   head tg  head egc  Electronic head                Optical head
             = ei − eea (+b)   ◄ eig ▸  ei             nc ──► eps = a·nc² + b
                  │           (learned linear)             (shared trunk with nc)
                  └───► egb = c·egc + d
```

- **Shared trunk** builds richness from Tg (4,143 rows) + egc (2,028 rows); its embedding vector is pooled once and re-used for all heads — this is the vanilla multi-task sharing done properly.
- **Electronic head** outputs `(ei, eea)` and *computes* `egc` in the forward pass via a learned linear map `egc = c1·ei + c2·eea + c3` (starts on (1, −1, 0)); `egb = m·egc + n` again.
- **Optical head** outputs `nc`, then `eps = a·nc² + b` with a/b learned. So in inference, better-nc knowledge directly improves eps without a separate model.
- **Tg head** stays independent (benchmark-anchored, distant physics).

### Training procedure
1. **Phase M (pre-train):** contrastive + masked-atom SSL on PI1M (the existing `pretrained_encoder.pt` is a valid init; contrastive-only head added).
2. **Phase F (fine-tune):** multi-task, per-target z-scored, `1/freq[t]` weighting, Smoothed-L1, AdamW, early-stopped per-fold — same shape as `mt_gnn_v2` but with the physics-forward heads.
3. **Fusion:** keep existing GBN; blend via the task per-target Ridge as today, then optionally the L2 sibling-fusion on top.

### Losses
- `L = Σ_t w_t · SmoothL1(pred_t, ȳ_t)` with `w_t = w_0 t/freq_t`
- tiny optional consistency term `(egc_pred − (ei−eea))²` only on rows that have all three labels (≤ dozens) — purely there to keep the map on-balance.
- The eps/nc coupling is *the forward pass*, not an extra loss.

### Fold-safe validation
- GroupKFold on `canon` — **identical to mt_gnn** (all rows of the same smiles in one fold). This kills the "same-SMILES sibling" leakage while *preserving* the legitimate use of other-target labels via the shared trunk.
- Pre-registered **gate**: ship only if `mean-OOF(A) ≥ OOF(P14)+0.008` **and** every target within `0.003` of P14.
- **pseudo-time-trial**: hold out a random 20% of the small-target polymers as if they were the 96%-overlap test set; if the gain collapses on this split-hold, the v15/v16 trap is being re-triggered — re-scope.
- **Ablations** (each one a notebook run, ~10): (a) remove physics-forward (independent heads); (b) remove multi-view/oligomer; (c) remove contrastive pruning; (d) remove target conditioning (back to vanilla multi-task).

### Failure modes & guard
- If physics-forward hurts ei/eea individually, **revert to independent `egc` head** (enforce by gate). The 0.995-correlation on 10 pairs is thin; the learned affine with Monotone prior is safer than a hard identity.
- Keep P14 **frozen** until gates pass — this is exactly the discipline that already safeguards v18.

---

## Task 6 — The breakthrough mandate

The single real barrier is prose: **200–330-row targets are underpowered because the chemical representation is 2D-fingerprint-centric and cross-target transfer is bolted on as features, while the physics is near-exact and the test manifold is 96% overlapping.**

Stop iterating "P14 + ε and gate". Instead:

> **Build the model as a target-conditioned, physics-forward graph model: one shared pre-trained (contrastive-PI1M) transformer paints the chemistry, the electronic head derives egc/egb from (ei, eea), and the optical head derives eps from n² — so each head inherits signal from every other target, and the 96% test-polymer overlap becomes learned physics, not a hack. Ship it only if the pre-registered gates and the overlap-simulation trial pass; otherwise keep P14.**

That is a single-notebook, one-week plan (pre-train cell + fine-tune cell + blend cell), with the sibling-level fusion and exact-match retrieval as fold-honest bonuses that protect the anchor P14.