"""Execute the v22 SMOKE notebook locally.

nbclient forces the kernel's CWD to the notebook's own directory, so a bare
`nbconvert --execute` cannot see `official_dataset/` (relative to repo root).
This driver overrides the resources path so the notebook resolves inputs the
same way the harness does (repo root), writes the executed notebook next to
the source, and keeps ./New folder/PolyWin_R2_v22_bert_arm_smoke.ipynb
the bit-stable deliverable.
"""
import os
import pathlib

import nbformat
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "New folder" / "PolyWin_R2_v22_bert_arm_smoke.ipynb"
DST = ROOT / "New folder" / "PolyWin_R2_v22_bert_arm_smoke_run.ipynb"

os.environ["SMOKE"] = "1"

with SRC.open(encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

resources = {"metadata": {"path": str(ROOT)}}
ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
nb, resources = ep.preprocess(nb, resources=resources)

with DST.open("w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("WROTE", DST)