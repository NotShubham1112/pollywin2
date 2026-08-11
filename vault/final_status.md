# FINAL STATUS — 2026-08-09

## Anchors
- P14 (v14 kernel, full-PI1M pretrain, simple per-target Ridge over [GBM, MT]): public LB = 0.883  <== FINAL
- Honest OOF (equal-weight mean R2): 0.8641

## Attempted local (all scored WORSE on LB than P14)
- v19 CT-PGCN (sib+phys blend, alphas<=0.30): local OOF 0.8706 (+0.0063, gates pass) but public LB 0.874 -> REGRESSION
- v18 hard-gate sib(halved)+phys: LB 0.864
- v17 sib-Ridge + physics eps: LB 0.862
- v16 cross-target decoder: LB 0.874
=> All cross-target blending OOF gains failed to transfer; CLOSED for this competition.

## ROOT CAUSE v19: OOF sibling leakage (same failure mode as v16)
The sibling pivot is built from FULL train and used unchanged inside GroupKFold val folds.
GroupKFold groups on SMILES, so all 7 target rows of one polymer land in the same fold; a val-fold
eps/nc/egb/ei/eea prediction uses that polymer's TRUE other-target labels from the SAME val fold
-> label leakage. Measured leak-eligible val rows: eea 216/216, egb 274/274, egc 101/101, ei 216/216,
eps 223/223 (~100%), nc 222/222, tg 4/10 (~40%). The v19 eps OOF jump (+0.031) is ~entirely this
artifact. P14 avoids it (twin arms use model OOF preds, not true labels) which is why P14 OOF tracks
LB and the sib/decoder families (v16 0.874, v17 0.862, v18 0.864, v19 0.874) do not.

## What was produced (kept, not shipped)
- vault/ctpgcn_v19.py  (same-engine CT-PGCN)
- vault/ctpgcn_submission_v19.csv
- vault/kernel-v17-sib-phys/PolyWin_R2_v19_ctpgcn_submit.ipynb (reproducible P100 notebook w/ CT cell; NOT run on Kaggle)

## RULE CHECK (AISEHack 2.0: PPP Round 2)
- Final used submission is a pure kernel artifact from reproducible P14 notebook (no external files).
- No data leakage in shipped path (GroupKFold on smiles, arms fold-safe).
- Reproducible notebook reproduces P14 + self-gating CT-PGCN cell; correct per gates.

## Remaining path to 0.90 (orthogonal to P14; does NOT reuse the v16/v19 sibling mode)
1. ModernBERT / pairwise-pretrained SMILES representation arm (NeurIPS 2025 winner technique, untried)
2. GRIN 3-unit repeat encoder as a 3rd GNN arm (only if OOF corr with P14 < 0.88)
3. TransPolymer frozen embeddings -> GBM (proven on eps/nc benchmarks)

## Fingerprint (v19 vs P14, test-level) — why it failed
- 180/224 egb rows: physics arm mean shift +0.010 at alpha 0.30, uncorrelated with LB -> egb damage
- 148/153 eps rows: double-blend (sib then phys) over-correction
- ei +0.009 shift, nc −0.005 drag: net ≈ zero per target
- Equal-weight metric: hurting egb + ei erases eps gain on public slice (matches LB −0.009)
## v21 leak-safe SIB arm (2026-08-11) - CLOSED, gate FAIL confirmed on LB
- Pre-registered gates 0-3 (docs/superpowers/specs/2026-08-11-v21-sibling-arm-design.md):
  - gate0 sib_only_r2: 0.52-0.89 (real cross-target signal)
  - gate1 leak audit: 0 exact matches (clean - unlike v16/v19)
  - gate2 OOF gain: soft/strong BOTH FAIL (eps/nc/ei mean +0.0000, overall +0.0002 on Kaggle; local -0.0012 / +0.0000)
  - gate3 worst-target: PASS (worst -0.0014 on Kaggle, -0.0025 local)
  - VERDICT: FAIL -> P14 stays final
- Kaggle kernel shubhamkambli11/polywin-r2-v21-sibling-arm v1: ran error-free on P100 (~1.5h), public LB 0.867
- P14 = 0.883 (submitted 2026-08-10, submission_v14.csv from v14 kernel) remains FINAL
- Lesson: even a LEAK-SAFE sibling arm (model-OOF twin features, zero label leakage)
  does not transfer on LB. Cross-target sibling info is not LB-exploitable in this competition.
