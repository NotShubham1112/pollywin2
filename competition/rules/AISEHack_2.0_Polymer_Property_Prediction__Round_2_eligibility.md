**Title:** AISEHack 2.0 — Eligibility: No Private Artifacts / External Datasets / Pre-Trained Weights

> **Note:** This is an excerpt from the **official AISEHack 2.0 website** (competition
> eligibility/requirements page). It is **NOT** part of the Kaggle competition-page
> boilerplate captured in the other files in this folder, and must not be confused
> with the Kaggle General Competition Rules.

---

| Field | Value |
|---|---|
| **Source** | AISEHack 2.0 official rules webpage |
| **Retrieved** | 16 August 2026 |
| **Relevant requirement** | **"No private artifacts, external datasets, or pre-trained weights allowed."** |

---

## How the submitted P14 notebook satisfies this requirement

Verified against the shipped `notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb` and its source-of-truth builder `src/notebook_builders/build_v14_kaggle_nb.py`:

1. **Data is read from the competition input only.** `PI1M.csv` is located via
   `find_input("/kaggle/input", "PI1M.csv")` which searches only the competition
   data mount (`/kaggle/input/ppp-round-2/`). No external dataset paths exist in the notebook.
2. **The encoder is trained from scratch inside the notebook.** `PretrainedEncoder`
   is instantiated fresh and self-supervised on PI1M in-kernel (masked atom/bond
   reconstruction). No pre-trained weights are downloaded or uploaded.
3. **The checkpoint is an execution artifact, not a dependency.** The trained encoder
   is written to `/kaggle/working/pretrained_encoder.pt` — the notebook's own output.
4. **No `/kaggle/input/*.pt` fallback.** The only `torch.load` reads the notebook's
   own `/kaggle/working/` output, guarded by `os.path.exists`.
5. **Graceful absence handling.** If the checkpoint does not exist, `pretrained_state`
   is `None` and the MT-GNN still trains from scratch — it never silently loads a
   local/private weight.

The model-generation process therefore runs end-to-end from allowed competition data
(`train.csv`, `test.csv`, `PI1M.csv`) entirely within the Kaggle environment.