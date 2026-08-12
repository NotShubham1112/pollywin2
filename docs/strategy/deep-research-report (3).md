# Executive Summary

Our current champion (P14) achieved **R² = 0.883** on the public leaderboard by blending a gradient-boosted (GBM) model with a graph neural network (MT-GNN) pre-trained on the PI1M corpus.  After systematic diagnosis, we know that further gains require **new, orthogonal signal** beyond these two models.  We evaluate six candidate strategies, each adding a distinct signal or technique.  For each, we detail the exact change, compute cost, leakage risk, expected gain, required artifacts, and a stop rule (pass/fail gate).  We then rank the candidates by expected payoff and time cost.  Our top two recommendations are:

- **Target-Specialist Blending (Inference Only)** – Re-weight the GBM and GNN predictions per target instead of one global blend.  This requires no retraining and minimal compute.  It targets the already-observed “weak” targets (EPS, NC) by giving more weight to their stronger arm.  In principle it can squeeze a few hundredths in R².  
- **Snapshot Ensembling (Model Averaging)** – Use saved GNN checkpoints (last few epochs) or minimal fine-tuning to create a small ensemble.  This has zero data leakage risk and is computationally very cheap (if models are saved).  Snapshot ensembles often yield marginal gains (∼0.001–0.005 R²) with almost no cost.

We **do not** recommend large new training experiments (e.g. massive pretraining or new architectures) given the contest deadline.  Pseudo-labeling carries high overfitting risk (as seen in v8 and literature on distribution shift) and should be approached only in a constrained form (if at all, and only after all else).  New descriptor streams (3D or Morgan-based GBMs) are promising orthogonal signals but would require significant feature engineering and tuning beyond our remaining time.  Instead, we focus on **exploiting** the existing models:

1. **Implement a target-specific inference blend.** This is almost free to try and targets the exact pattern we saw: EPS/NC get most gain from the GNN arm (others from GBM).  We **pre-register** that success is OOF R² ≥ 0.880 and any further lift on the public LB; failure is no OOF gain (or loss).  
2. **If submissions remain, try snapshot averaging.** E.g. average predictions from epochs 8, 9, 10 of the GNN.  Success: any measurable OOF gain; failure: none.

Below we compare all candidates in detail, with tables and references, and then give a step-by-step plan for our top choice.

| Experiment                         | Scope (change)                           | Compute (hrs) | Leakage/Overfit Risk | Expected LB lift | Artifacts Needed                          | Stop Rule (pre-registered)                     |
|------------------------------------|------------------------------------------|---------------|----------------------|------------------|--------------------------------------------|------------------------------------------------|
| **1. Extended GNN Pretraining**    | Increase GNN size (width/depth) and epochs, or switch to contrastive loss on PI1M (vs whatever v14 used).  | ~>4h (GPU)   | Minimal (only unlabeled) | +0.003–0.01*      | `mt_gnn_pretrained.pt`, `train.csv/test.csv`, PI1M | Success if OOF ↑≥0.005 or weakest target improves; otherwise stop. |
| **2. Safe Pseudo-Labeling**        | Generate test-labels for high-confidence targets (e.g. EPS/NC only), then fine-tune GBM/GNN with small weight (e.g. 10–20%).  | ~1–2h        | High (uses test data labels) | +0.002–0.01 (uncertain) | P14 model preds, train/test CSV, holdout split | Success if OOF & LB both improve; fail if LB drops or OOF gain disappears.  |
| **3. New Descriptor Stream**       | Add 3D conformer descriptors (e.g. SASA, radius of gyration) or Morgan fingerprints as inputs to a **new model** (e.g. GBM or small neural net).  | ~6–12h (CPU/GPU) | Low (only uses official data) | +0.003–0.008        | RDKit (3D conf, RDKIT QED/SASA), `train.csv/test.csv` | Stop if OOF gain <+0.003 or no LB gain after one trial.  |
| **4. Inference-Only Ensemble**     | **Target-specific blend:** assign different GBM/GNN weights per property; also try monotonic calibration (post-hoc scaling). No model retraining. | <1h          | None (no retraining) | +0.002–0.005       | P14 OOF predictions, submission script            | Success if OOF↑≥0.002; else stop and revert. |
| **5. Snapshot Ensembles**          | Average multiple checkpoints of GNN (or final GNN + GBM ensemble) to reduce variance.  (E.g. take epochs 7–10 or cyclic LR snapshots.) | <1h         | None | +0.001–0.003       | Saved model checkpoints, existing code          | Success if OOF↑≥0.001; otherwise skip. |
| **6. Hybrid (Pretrain + Pseudo)**  | Combine (1) and (2): e.g. full GNN pretrain + selective pseudo-label for best targets. | ~4–6h (extensive) | High (mixing test labels with new training) | +0.002–0.008    | Same as (1)+(2)                                 | Success only if **both** pretrain yields +0.003 *and* pseudo yields extra +0.002. High risk, so likely skip. |

*These expected gains are rough; actual improvement is diminishing as we approach the ceiling. Note “[+0.003–0.01]” indicates low confidence of such a gain; e.g. P14 pretraining gave +0.006.

## 1. Self-Supervised Pretraining Variants 

**Scope:** We already saw that scaling the GNN pretraining (20k→995k molecules, 5→10 epochs) produced a solid +0.008 OOF (＋0.006 LB).  To push further, we could enlarge the GNN architecture or train longer, or change the SSL objective.  For instance, *contrastive graph learning* (like GraphCL/MolCLR) has been shown to improve molecular GNNs: MolCLR trained on 10M molecules with augmentations and achieved state-of-art results on regression tasks after fine-tuning.  We could adapt this idea on PI1M. 

**Compute:** High. Extending epochs or network size (say double hidden dim or layers) may require >4 GPU-hours. Implementing a new SSL loss (e.g. InfoGraph or GraphCL) adds dev time.

**Risk:** Minimal for leakage (only uses unlabeled data).  Main risk is wasted time (diminishing returns). Overfitting is less a concern on large unlabeled pretrain.

**Expected LB Upside:** Probably small. P14’s pretrain already gave +0.006 LB. A bit more might come from a better objective or larger model, but likely <+0.01.  If the encoder truly learns new features, we should see a drop in the **GBM–GNN correlation** (currently ~0.95).  If not, gains will plateau.

**Artifacts:** We’d reuse `mt_gnn_pretrained.pt` updated, plus `train.csv`/`test.csv` for finetuning. No new data needed.

**Stop Rule:** *Success:* OOF R² increases ≥+0.005 over P14 and/or the weakest targets (EPS, NC) show an appreciable uplift. *Fail:* OOF gain <+0.003 and no drop in GBM–GNN correlation – indicating no new signal learned. In the latter case we abandon further pretrain tweaks.

## 2. Safe Pseudo-Labeling

**Scope:** Use the current best model to predict on the unlabeled test set, then train on those pseudo-labels.  However, we must be extremely conservative to avoid the “easy overfit” pitfall (as seen in our v8 trial).  E.g., we could pseudo-label **only the two weak targets** (EPS/NC) where the model is most confident, apply a high-confidence threshold, and add them with small weight (10–20%) to the training loss.  Another safe variant is **cross-fold pseudo-labeling**: hold out one fold of train as pseudo validation so that the model generating labels never sees them during its own training.

**Compute:** Low to moderate. Generating predictions on ~5k test polymers is trivial. Retraining (or fine-tuning) another epoch with pseudo data is maybe 1–2 GPU-hours.

**Risk:** High. If test distribution differs even slightly, pseudo-labels can mislead the model to overfit in-distribution (our v8 had this exactly: OOF →+0.017, LB →−0.019).  To mitigate, we must use *per-target thresholds* (only very high-confidence rows) and *holdout validation*.  But risk remains that we fit to public test labels.

**Expected LB Upside:** Uncertain. In principle, good pseudo-labeling can add up to +0.01, but more often it’s +0.002–0.005 or worse.  Given the lesson, we expect at most ~+0.002–0.003 if done perfectly for *one or two targets*. 

**Artifacts:** We need the model that will generate labels (P14 encoder and GBM).  Also the original folds so we can split for validation. Possibly a new version of the training script with pseudo-label data loader.

**Stop Rule:** *Success:* Pseudo-training yields an OOF gain ≥+0.005 on held-out pseudo-data and *no drop* in LB on a dry-run (if possible by submitting privately). *Fail:* Any LB drop or OOF gain that evaporates in the final. In practice, we should do a mini-test: e.g., only use a single fold of train and evaluate on the withheld fold before committing. If that fails, scrap it.

## 3. New Descriptor Streams

**Scope:** Introduce **entirely new features** that neither GBM nor GNN currently capture, then train a separate model.  Two promising orthogonal signals are:
- **3D Conformer Descriptors:** Generate a 3D conformer for each SMILES (e.g. with RDKit’s ETKDG) and compute descriptors like radius of gyration, asphericity, partial charge metrics, solvent-accessible surface area (SASA), etc.  These capture shape and electronic distribution not in 2D SMILES.  
- **Morgan Fold Features:** Compute Morgan fingerprints (circular substructure counts) and feed into a new GBM or neural net.  The existing GBM likely used many RDKit 2D descriptors; a separate Morgan-based model could catch subtle structural motifs.

After computing these features for *train+test*, one can train a lightweight model (e.g. CatBoost or small MLP) on just those features. The outputs would then be ensembled (stacked) with the current blend.

**Compute:** Medium to high. 3D conformer generation for ~1k+5k molecules might take several hours CPU (though maybe faster with GPU for certain tools). Feature calculation is modest once 3D conformers exist. Training a GBM on new descriptors is <1h. 

**Risk:** Low for data leakage (only uses `train.csv`/`test.csv`).  Main risk is wasted time: if the new stream correlates highly with existing models, gains will be negligible (as we saw with RDKit 2D descriptors vs GNN).  Overfitting is unlikely on large PI1M, but the small target data could suffer, so we should validate carefully.

**Expected LB Upside:** This could be substantial *if* truly orthogonal. Hard to predict, but perhaps **up to +0.005–0.008** if these features capture missed variance. However, it’s speculative and requires finishing before deadline. A rolled-out example: teams in material tasks often boost by ensembling a quantum/descriptors model with GNN. 

**Artifacts:** This needs none of the current code aside from data. We’ll use the official SMILES and possibly PI1M for consistency (though PI1M has no labels so it's only for GNN). We will need to run RDKit for 3D/descriptor code on all SMILES, and compute Morgan fingerprints via RDKit. Possibly we reuse `train_targets.csv` only for labels.

**Stop Rule:** *Success:* The new descriptor-model (stacked after GBM/GNN) shows **OOF R² improvement ≥+0.003** without hurting other targets. *Fail:* If OOF R² is within ±0.001 of current (or worse), or if LB doesn’t budge on a dry run, then drop this branch. (We also limit to one try, since time is short.)

## 4. Inference-Only Ensembles and Calibrations

**Scope:** With no retraining, tweak the final predictions. We already tried a global ridge blend. Now do **target-specific blends**: examine each target’s OOF and weight the GBM vs GNN accordingly.  For example, EPS and NC improved most via GNN (as we saw: EPS +0.026 OOF, NC +0.024) – these should lean heavier on the GNN predictions. Conversely, Tg/Ei etc. can lean heavier on the GBM. We can set these weights by inspecting OOF or via a small validation search.  

Additionally, consider simple **post-hoc calibration**: for each target, we could fit a monotonic isotonic or scaling to align the train-test distribution (if any drift is detected). For example, if the GNN tends to under-predict high values of Tg, we could apply a small additive offset (as was done in another polymer comp). But we must not tune directly on public test.

**Compute:** Very low. This is pure inference/post-processing. OOF predictions are available; we just recombine them.

**Risk:** None for leakage (we’re not using test labels). Calibration should only use training distribution. The only risk is overfitting those OOF numbers, but using monotonic transforms (like adding a constant *in proportion to train std*) is low-risk.

**Expected LB Upside:** Small but real. We expect a few thousandths of R² gain if done right. As a baseline, in v15 we saw that naive per-target ridge tuning didn’t help LB, but a *manual* target-special strategy (if well-chosen) can eke out a little more.  Even +0.002–0.005 is valuable in this late stage.

**Artifacts:** We need the OOF predictions of each arm per target (which we have). No new training needed. Possibly one script to recompute weights and produce submission.

**Stop Rule:** *Success:* OOF R² rises by ≥+0.002. *Fail:* OOF flat/↓ or undesirable target trade-offs. If it fails, we simply keep P14 as final. (Given zero cost, we can try and accept even small wins.)

## 5. Snapshot/Checkpoint Ensembles

**Scope:** Take advantage of **multiple model checkpoints** from the GNN’s training trajectory. For instance, if we have saved the GNN at epochs 8, 9, 10 (or used a cyclic learning-rate schedule), we can average their predictions. Alternatively, average the *model weights* (though prediction averaging is simpler). This uses the idea of *Snapshot Ensembles*: “train one network, converge to several local minima along its optimization path and save each model”. The result is effectively an ensemble at no extra training cost.

**Compute:** Very low. If checkpoints are already saved, it’s just reading and averaging. If not, we might need to re-run a quick 5–10 minute script to save intermediate weights (likely available). Then inference is trivial.

**Risk:** Zero leakage. It only uses our single model’s history on train/validation splits. Overfitting risk is less than single-model variance.

**Expected LB Upside:** Likely very small (order 0.001–0.003). Snapshot ensembles consistently give slight robustness gains in neural nets. It’s essentially variance reduction of the GNN arm.

**Artifacts:** We need the saved GNN checkpoints or the ability to save them. If not, we re-run final fine-tuning for a few epochs saving models. We already have the OOF preds for the final model, so we’d generate new OOF preds and submission from each snapshot. 

**Stop Rule:** *Success:* OOF R² increase ≥+0.001. *Fail:* no change. If unsuccessful, abandon. (Because it’s cheap, we can try it even if gain is small.)

## 6. Hybrid Approaches

**Scope:** Combine the above: e.g. after fully pretraining the GNN (as in P14/P15), then do *target-limited pseudo-labeling*. Or after contrastive pretraining, apply pseudo-label on top. Essentially, stack the strategies.

**Compute:** Very high. It compounds (1)+(2), so easily >6h.

**Risk:** Very high. Each part is already risky (especially pseudo-labeling). Combining them inherits all risks. There’s also extreme overfitting potential if pseudo-label is done on an even more powerful encoder without strong validation.

**Expected LB Upside:** Possibly slightly higher if both parts succeed additively, but more likely the two errors could amplify. For example, if contrastive pretrain didn’t lower GNN–GBM correlation, adding pseudo-label could still go wrong. Hard to justify with little time.

**Stop Rule:** Given limited time, we should **only** consider such hybrids *after* all other simpler options are exhausted and *only if* early results are clearly promising. Otherwise skip.

## Candidate Ranking 

| Priority | Experiment                     | Rationale                                                |
|----------|--------------------------------|----------------------------------------------------------|
| **1**    | Inference-Only (Target Blend)  | Cheapest, lowest risk, directly exploits known gains.    |
| **2**    | Snapshot Ensemble              | Very low cost/risk, small guaranteed variance reduction. |
| **3**    | Extended Pretraining (Contrastive / Larger) | Scales a proven mechanism (P14), moderate cost/risk.    |
| **4**    | New Descriptor Model           | Potentially adds true orthogonal signal, but high effort for uncertain gain. |
| **5**    | Safe Pseudo-Label (Targeted)   | High reward possible but high risk (v8 warning). If done, very narrowly. |
| **6**    | Hybrid (Pretrain + Pseudo)     | Too complex for remaining time; only if others plateau.  |

Given our final two submission slots, we **absolutely should use one** on **target-specialist inference blending** and **reserve** one for the snapshot ensemble (or vice versa). Both are quick to implement and do not risk our 0.883 anchor. Only if these both somehow fail (they should not, but if so) would we consider a controlled pseudo-label for, say, EPS with *very* high confidence. 

The evidence strongly suggests that any further gain comes from *fine margins*.  In fact, our analysis showed EPS/NC dominated the improvement, not any global de-correlation.  Thus **doing more of what helped** is our best bet: amplify the weak targets via ensemble tricks rather than reinventing a whole new model. 

## Implementation Plan for Top Recommendation: Target-Specialist Blend

**Timeline:** All steps fit in 4–6 hours of work (one working day).

1. **Prepare OOF predictions and targets.** Retrieve the saved out-of-fold (OOF) predictions for each model and each target from P14. We need the 7-dim arrays of (GBM, GNN) for each polymer in the train folds. Also gather the train target values for comparison.
   - *Files:* e.g. `v13_oof_blend.npz` or similar in vault, or regenerate via existing pipeline if needed.
   - *Check:* Plot or compute per-target R² for each arm to confirm current weights.

2. **Define target-specific weights.** Based on the lab report insights, we set initial weights:
   - **Tg, Egc, Ei:** lean towards **GBM** (e.g. weights [0.8 GBM, 0.2 GNN]).
   - **Egb, Eea, EPS, Nc:** lean towards **GNN** (e.g. [0.3 GBM, 0.7 GNN]).
   We should tune these on OOF: for each target, try a few weight pairs (e.g. (0.7,0.3), (0.6,0.4), ... ) and pick the best R² on OOF. Use the training folds only.
   - *Tests:* A brief grid search for each target (maybe ±0.1 around initial guess).
   - *Example:* EPS got +0.026 from GNN, so maybe try (0.2,0.8), (0.1,0.9) vs (0.3,0.7) and pick highest R².

3. **Apply weights and compute blended OOF.** Using the chosen weights, blend the OOF predictions and compute the new mean R² (and per-target R²). Verify against the old blend. Also check correlation of the new blended preds vs each component.
   - *Cite effect:* If done correctly, we should see R² ≥ 0.879 (target is +0.003 over 0.8769).
   - *Stop rule:* If OOF R² <0.877 (no gain), abort this path.

4. **Generate test-set predictions.** Run the notebook/script that creates final predictions with the new weights. (If the blend was done in Python, update the weight array; if in notebook, edit the cell.) Export `submission_v15.csv` (for example).

5. **Submit or dry-run.** Submit the new blend to Kaggle and note the public LB. We expect around **0.885–0.888** if our theory holds (EPS/NC improvements should mainly lift LB).
   - *Check:* Confirm that the patterns (EPS up, others stable) match our predictions. If LB disappoints (e.g. no change), then revert submission and re-evaluate weights (or skip and keep P14).

6. **Document results.** Save the chosen weights, OOF metrics, and final LB score in our records. Write a brief note in the logs (`docs/`) summarizing the change and outcome.

7. **(Optional) Quick calibration.** If time remains and we see slight biases (e.g. one target is systematically under/over predicting), apply a monotonic offset: e.g. if Tg predictions are too low by a constant factor on train, add that factor to submission (like `[2†L106-L113]`). Validate on OOF before applying. 

8. **Prepare fallback.** Regardless of outcome, our final submission #1 remains P14. We will only replace it if the new blend clearly beats it on the public LB. In either case, commit all code, weights, and results. 

This one-day plan prioritizes reproducibility: we use only our existing artifacts (OOF preds, model outputs) and standard utilities.  We avoid new training or data gathering except for a small tuning loop.  All steps should be scripted or noted so they can be re-run exactly. 

```mermaid
flowchart LR
    A[Current Best (0.883 LB)] --> B{Try Experiment}
    B -->|Target Blend| C[Compute OOF with new weights]
    C -->|OOF↑| D{Submit blend}
    C -->|OOF≰| E[Discard; keep 0.883]
    D -->|LB↑| F[Use blend as final]
    D -->|LB≰| E
    B -->|Snapshot Ensemble| G[Average last GNN preds]
    G -->|OOF↑| H{Submit ensemble}
    G -->|OOF≰| E
    H -->|LB↑| I[Possible final]
    H -->|LB≰| E
    B -->|Other| E
```

*Figure: Decision tree of final experiment choices. We attempt the top-priority “Target Blend” first. If it passes (both OOF and LB rise), we adopt it; if not, we try the Snapshot ensemble next, etc. If all fail, we fall back to the frozen 0.883 model.*


**Sources:** Prior work confirms these strategies.  Self-supervised GNN pretraining reliably boosts molecular predictions, especially when fine-tuned.  Contrastive pretraining (e.g. MolCLR) on millions of molecules achieves state-of-art results.  Snapshot ensembles have shown *“no additional training cost”* but lower error via cyclic LR checkpoints.  Pseudo-labeling can help but is known to risk overfitting under distribution shift.  3D descriptors capture conformational properties that 2D models miss.  We incorporate these insights into our plan, then decisively execute only the safest high-return options.