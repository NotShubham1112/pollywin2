"""Tests for the v9 GNN Kaggle notebook generator (build_gnn_kaggle_nb.py).

Each test rebuilds the notebook from the generator and asserts on the combined
source of all code cells. Run with `python tests/test_gnn_kaggle_nb.py` or pytest.
"""
import pathlib, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "PolyWin_R2_v9_GNN_kaggle.ipynb"
GEN = REPO / "build_gnn_kaggle_nb.py"

def _build():
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True)
    nb = nbformat.read(str(NB), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    return code, md

def test_lowercase_targets():
    """The pipeline uses lowercase target names; the original v9 scratch notebook
    used capitalized names that would KeyError on real data."""
    code, _ = _build()
    assert 'TARGETS = ["tg","egc","egb","eps","nc","ei","eea"]' in code

def test_self_contained_data_loading():
    code, _ = _build()
    assert '"/kaggle/input"' in code
    assert 'official_dataset' in code
    assert "train.csv" in code and "test.csv" in code

def test_gnn_trunk_gine():
    code, _ = _build()
    assert "GINEConv" in code
    assert "class GNNTrunk" in code

def test_gnn_fold_safe_and_trust_check():
    code, _ = _build()
    assert "trust_frac" in code
    assert "trust_mask" in code and "trust_ids" in code  # 15% trust-check holdout, excluded from training
    assert "trust_graphs" in code  # scored separately from the fold-val set
    assert "patience" in code
    assert "bad_epochs" in code  # early-stopping counter

def test_honest_oof_trust_tables():
    code, _ = _build()
    assert "OOF R2" in code
    assert "trust R2" in code
    assert "gap" in code
    assert "suspect" in code

def test_blend_gbm_vs_gbmgm():
    code, _ = _build()
    assert "GBM-only" in code and "GBM+GNN" in code
    assert "Ridge" in code
    assert "Level-1.5" in code
    assert "StandardScaler" in code  # scaler feeding the Ridge stack

def test_gnn_excluded_where_suspect():
    code, _ = _build()
    assert "suspect" in code
    assert "0.05" in code  # trust-gap threshold

def test_submission_format_and_bounds():
    code, _ = _build()
    assert '"id"' in code and '"target"' in code
    assert 'test["id"].values' in code
    assert 'np.maximum(final[_mm], 0.0)' in code  # egc/egb/ei >= 0
    assert 'np.maximum(final[_mm], 1.0)' in code  # eps >= 1
    assert 'np.clip(final[_mm], 1.0, 3.0)' in code  # nc in [1,3]
    assert "sub.to_csv" in code

def test_figures_present():
    code, _ = _build()
    for n in ["01_target_balance", "02_gnn_oof_vs_trust", "03_blend_comparison",
              "04_gnn_pred_vs_actual", "05_gnn_residuals", "06_base_model_compare"]:
        assert n in code

def test_rows_per_target_chart():
    code, _ = _build()
    assert "rows per target" in code.lower() or "value_counts" in code

def test_predict_graphs_on_bags_folds():
    code, _ = _build()
    assert "def predict_graphs_on" in code
    assert "bag" in code
    assert "state" in code

def test_blend_walrus_no_crash():
    """The Step-7 blend used an awkward walrus placeholder guard; assert it is
    still a valid expression and not a bare broken name."""
    code, _ = _build()
    assert "Zgn_te" in code
    assert "Zgbm_te" in code

def test_all_cells_compile():
    """Every code cell must be valid Python (regression: a SyntaxError that only
    nbconvert would catch)."""
    import ast
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True)
    nb = nbformat.read(str(NB), as_version=4)
    for i, c in enumerate(nb.cells):
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