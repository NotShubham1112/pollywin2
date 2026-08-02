"""Tests for the v5 notebook generator (build_pipeline_nb.py).

Each test rebuilds the notebook from the generator and asserts on the combined
source of all code cells. Run with `python tests/test_pipeline_nb.py` or pytest.
"""
import pathlib, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "AISEHack_Round2_Pipeline.ipynb"
GEN = REPO / "build_pipeline_nb.py"

def _build():
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True)
    nb = nbformat.read(str(NB), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    return code, md

def test_cell1_gpu_bootstrap():
    code, _ = _build()
    assert "def ensure_torch_cuda" in code
    assert "GLOBAL_FOLDS = 5 if SMOKE else 10" in code
    assert "pipeline_out_smoke" in code

def test_cell3_folds_persistence():
    code, _ = _build()
    assert "FOLDS_CSV" in code
    assert "loaded folds.csv" in code
    assert "dedup[\"fold\"] = folds" in code

def test_aux_physics_cell():
    code, _ = _build()
    assert "AUX_TASKS" in code
    assert "def aux_physics_scores" in code
    assert "Chem.AtomHasConjugatedBond" in code

def test_harness_electronic_targets_and_artifact_helper():
    code, _ = _build()
    assert "ELECTRONIC_TARGETS = [\"egc\",\"egb\",\"eps\",\"nc\",\"ei\",\"eea\"]" in code
    assert "def save_oof_artifact" in code

def test_efn_replaces_mtnn():
    code, _ = _build()
    assert "class EFN" in code
    assert "def efn_fit_predict" in code
    assert "save_oof_artifact(\"efn\"" in code
    assert "train_multitask" not in code
    assert "MultiTaskNN" not in code

if __name__ == "__main__":
    import traceback
    failed = 0
    for _n, _fn in list(globals().items()):
        if _n.startswith("test_") and callable(_fn):
            try:
                _fn(); print("PASS", _n)
            except Exception as _e:
                failed += 1
                print("FAIL", _n, "->", _e)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
