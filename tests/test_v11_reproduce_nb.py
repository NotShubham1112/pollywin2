"""Tests for the v11-reproduce notebook generator stage (build_v12_kaggle_nb.py).

The same generator emits both notebooks; POLYWIN_STAGE=v11 must produce
PolyWin_R2_v11_reproduce.ipynb with the bucket-MoE cells omitted and a forced
v11-blend submission. Run with `python tests/test_v11_reproduce_nb.py` or pytest.
"""
import os, pathlib, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "PolyWin_R2_v11_reproduce.ipynb"
GEN = REPO / "build_v12_kaggle_nb.py"

def _build():
    env = dict(os.environ, POLYWIN_STAGE="v11")
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    nb = nbformat.read(str(NB), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    return code, md

def test_v11_header_and_stage():
    code, md = _build()
    assert "PolyWin R2 — v11 blend reproduce" in md
    assert "notebook-backed submission" in md
    assert 'STAGE = "v11"' in code
    assert "GLOBAL_FOLDS = 3 if SMOKE else 10" in code
    assert "BUCKET_FIT_CAP" in code

def test_v11_shared_pipeline_core():
    code, _ = _build()
    assert "def build_features(smiles_list, canon_list=None):" in code
    assert "def get_splits(tt):" in code
    assert "def gbm_fit_predict(tt, make_model, Xtr_full, Xte_full, use_folds=True):" in code
    assert "class GINEEncoder(nn.Module):" in code
    assert "v11_blend_oof = {}" in code
    assert "grid = np.linspace(0.0, 1.0, 21)" in code
    assert "V11_W[tt] = float(np.mean(w_acc))" in code
    assert "v11 reference blend" in code

def test_v11_bucket_moe_omitted():
    """The bucket-MoE cells are dropped entirely; only dead v12 branches (never
    taken when STAGE == "v11") may retain v12-only strings."""
    code, _ = _build()
    assert "KMeans" not in code
    assert "run_bucket_moe" not in code
    assert "ROUTER_COLS" not in code
    assert "cluster_assignment" not in code
    assert "BUCKET_KS" not in code

def test_v11_submission_forced_blend():
    code, _ = _build()
    assert 'if STAGE == "v11":' in code
    assert "FINAL_TE[tt] = v11_blend_te[tt]" in code
    assert "REPRODUCE-V11: submission = v11 blend (forced)" in code
    assert "submission saved:" in code
    assert "==== PIPELINE COMPLETE ====" in code
    assert "23_v11_reproduce.png" in code

def test_v11_runtime_guard_forces_blend():
    """The forced v11 blend must be the first branch so STAGE == "v11" short-circuits
    before any v12 bucket-MoE logic is reachable."""
    code, _ = _build()
    i = code.index('if STAGE == "v11":')
    j = code.index("FINAL_TE[tt] = v11_blend_te[tt]", i)
    assert j > i
    else_idx = code.find("else:", j)
    assert "REPRODUCE-V11" in code[i:else_idx]

def test_v11_all_cells_compile():
    """Regression: a SyntaxError in any emitted cell would break nbconvert."""
    import ast
    env = dict(os.environ, POLYWIN_STAGE="v11")
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    nb = nbformat.read(str(NB), as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)

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
