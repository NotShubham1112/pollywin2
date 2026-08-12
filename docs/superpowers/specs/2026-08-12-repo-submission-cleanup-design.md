# Repo Submission Cleanup — Design

> AISEHack 2.0 Round 2 (ppp-round-2), final submission prep. Team Cosmic.
> Date: 2026-08-12.

## Goal

Prepare the `pollywin2` GitHub repo as the accessible code-repository deliverable
required by the AISEHack 2.0 R2 submission rules: clean, readable, rule-compliant,
with a full README and the P14 final notebook marked as the locked submission.

## Decided (user-approved 2026-08-12)

- Final submission: **P14** (`PolyWin_R2_v14_p1m_pretrain.ipynb`, public LB 0.883).
  v21/v22 gates failed; P14 stays locked.
- Kaggle notebook link: `https://www.kaggle.com/code/shubhamkambli11/polywin-r2-v14-p1m-pretrain`
- Team name: **Cosmic**
- Layout: **hybrid** — group what is safely moveable, keep the test-bound
  generator+notebook group at root so `tests/` keep rebuilding notebooks.

## Changes

1. **`.gitignore`** add scratch/agent dirs so `git add -A` stays clean:
   `.venv_intel/`, `.kilo/`, `.superpowers/`, `.firecrawl/`, `.claude/`,
   `sandbox-opencode/`, `hack_skills/`, `session-*.md`, `codex.md`, `smoke*.txt`.
2. **`git rm --cached`** tracked scratch: `session-ses_*.md` (4 files), `smoke_full.txt`,
   `smoke_out.txt` — keep on disk, stop tracking.
3. **Rename `New folder/` → `v22/`** via `git mv`. `v22/tests/*` resolve modules via
   `parents[1]`, so module imports keep working unchanged.
4. **Move research docs** → `docs/`: `polymer_research_plan.md`,
   `polymer_modeling_strategy.md`, `deep-research-report (3).md`.
5. **Keep at root** (test-bound): all `build_*.py`, version notebooks, `tests/`,
   `vault/`, `official_dataset/`, v20/v22 gating scripts, `mt_gnn_v2.py`.
6. **README.md** — team, final notebook + link, metric, architecture, version
   history table, repro instructions, rules compliance, layout map.
7. Commit in two commits (cleanup, README) and push to `origin/main`.

## Explicit non-goals

- No `notebooks/` move of the version notebooks (breaks 16 rebuild tests tonight).
- No git history rewrite, no force push, no repo rename.
- No deletion of any working file on disk.