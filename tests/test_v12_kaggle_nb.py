"""Tests for the v12 Chemistry Bucket MoE notebook generator (build_v12_kaggle_nb.py).

Each test rebuilds the notebook from the generator and asserts on the combined
source of all code cells. Run with `python tests/test_v12_kaggle_nb.py` or pytest.
"""
import pathlib, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "PolyWin_R2_v12_bucket_moe.ipynb"
GEN = REPO / "build_v12_kaggle_nb.py"

def _build():
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True)
    nb = nbformat.read(str(NB), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    return code, md

def test_v12_header_and_data():
    code, md = _build()
    assert "Chemistry Bucket MoE" in md
    assert 'TARGETS = ["tg","egc","egb","eps","nc","ei","eea"]' in code
    assert "def canon_key(smiles):" in code
    assert 'groupby(["canon", "target_type"], as_index=False)["target"].median()' in code
    assert "GroupKFold" in code
    assert '"/kaggle/input"' in code
    assert "GLOBAL_FOLDS = 3 if SMOKE else 10" in code
    assert "BUCKET_FIT_CAP" in code

def test_v12_bootstrap_flags_and_pkgs():
    code, _ = _build()
    assert 'os.environ.get("POLYWIN_SMOKE", "0") == "1"' in code
    assert "PRETRAIN_EPOCHS" in code and "PRETRAIN_SAMPLE" in code
    assert "def ensure_pkg(pkg, import_name=None):" in code
    assert "torch_geometric" in code and "lightgbm" in code
    assert "def find_input(base, name):" in code
    assert "def parse_mol(smiles):" in code

def test_v12_feature_factory():
    code, _ = _build()
    assert "DESC_NAMES = [d[0] for d in Descriptors.descList]" in code
    assert "def polymer_physics(mol):" in code
    assert "def build_features(smiles_list, canon_list=None):" in code
    assert 'Xtr, ok_tr = build_features(dedup["smiles"].tolist())' in code
    assert 'Xtr.to_pickle(os.path.join(WORK, "Xtr.pkl"))' in code
    assert "X_all = X_all.replace([np.inf, -np.inf], np.nan)" in code
    assert '"ring_count","rigidity","flexibility","halogen_density"' in code

def test_v12_harness_and_gbm():
    code, _ = _build()
    assert "MODEL_COLS = list(Xtr.columns)" in code
    assert "def get_splits(tt):" in code
    assert "def record(name, tt, oof, te_pred):" in code
    assert "def gbm_fit_predict(tt, make_model, Xtr_full, Xte_full, use_folds=True):" in code
    assert "def make_lgb():" in code
    assert "moe_gbm_chk.parquet" in code
    assert 'oof_store[(r["key"], r["target"])] = r["oof"]' in code
    assert "SMOKE: loaded" in code

def test_v12_stacking():
    code, _ = _build()
    assert "def store_key(b, tt):" in code
    assert "L15_OOF = {}; L15_TE = {}" in code
    assert "CROSS_MAP = {" in code
    assert "def reliability_features(tt, models):" in code
    assert "def cross_oof_features(tt):" in code
    assert "FINAL_OOF = {}; FINAL_TE = {}" in code
    assert "Ridge(alpha=10.0)" in code
    assert "StandardScaler" in code

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
