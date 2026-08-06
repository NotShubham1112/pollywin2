# Notebook Submission Requirement

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Source: Rules §6.2, §7 (competition-specific, Round 2)

> ⚠️ **This is the single most important rule for our final submissions.** Round 2 overrides the general Kaggle rule that permits prediction-file uploads. **Every submission must be backed by a compliant Kaggle Notebook.**

---

## Notebook/Code-Only (§6.2.2)

- This is a **notebook/code-only competition**.
- **All submissions must be generated entirely within a Kaggle Notebook.**
- Although Kaggle permits uploading a prediction file directly, **prediction-only submissions that are not backed by a compliant Kaggle Notebook will be invalidated.**
- All stages of the pipeline must execute inside the notebook during a **single run**:
  data loading → train/val prep → model definition → training/fine-tuning → inference → submission file generation.
- **Manual intervention at any stage is not permitted.**

## Mandatory Notebook-Backed Submissions (§7.1)

For **every** submission:

1. The submission description **must include a link to the notebook** used to generate it.
2. The **default/pinned version of that notebook must correspond exactly to the version that generated the submitted results. This is mandatory.** Later versions may be created, but the default/pinned version must remain the one that produced the score.
3. The notebook may be private but **must be shared (view access) with all competition hosts**:
   - `Rohit Batra IITM`
   - `Rahulsundar`
   - `LaksmanN`
   - `VIJITH P`
   - `shreyasri0301`

Submissions without a linked notebook, without proper sharing, or with a mismatched default/pinned version **will be invalidated**.

## Reproducibility & Post-Competition Validation (§7.2)

- After the competition, hosts will **execute the default/pinned version** of the notebook linked to our selected best submission (from the two chosen final submissions).
- To remain eligible:
  - The notebook must run **end-to-end without manual intervention**.
  - Execution must complete within **Kaggle's compute and time limits**.
  - **The reproduced results must match the submitted results**.
- Participants must explicitly set and document all relevant **random seeds**.
- Failure to reproduce → **invalidation regardless of leaderboard position**.

## No External Data / No Uploaded Artifacts (§6.2.1, §6.2.4)

- Use **only the official Competition Data** (train.csv, test.csv, PI1M.csv, sample_submission.csv).
- **Prohibited:** external/private datasets, attaching external datasets to the notebook, using data generated or collected outside notebook execution.
- **Prohibited:** uploading pretrained model weights/checkpoints, embeddings, feature files, cached tensors, or processed datasets. **All model weights and artifacts must be produced during the notebook run.**

## Submission File

- File must be named **`submission.csv`** (the submission requirement noted on the Requirements page).

---

## Impact on Our Pipeline (measured 2026-08-07)

- Our standing **v11 blend (0.852 LB) was produced locally** (`gnn_moe_blend.py`/`gnn_v10_blend.py`), submitted as a **prediction-only CSV with no notebook link** → **at risk of invalidation** under §6.2.2/§7.1.
- v8, v10, v12 CSV submissions carry the same risk. Only v5 claims a "live Kaggle run."
- **Conclusion:** the next submission must be a self-contained Kaggle notebook that trains everything inside one run (v6 stack + v10 GNN + blend + v13 specialist), pins that exact version, shares it with the hosts, and links it in the description. Local scripts remain valid only for smoke testing / offline analysis, never for the final submission.

See: [[Submission Rules]] · [[Leaderboard and Evaluation]] · [[Disqualification and Conduct]]

#rules #submission #notebook #compliance
