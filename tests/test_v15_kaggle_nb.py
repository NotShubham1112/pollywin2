"""Tests for the v15 EPS/NC Focus notebook generator stage
(build_v15_kaggle_nb.py).

v15 = v14 + exactly ONE functional change: the fine-tune graph sample weight for
rows whose target_type is eps or nc is multiplied by an extra focus factor
(TGT_FOCUS). Everything else must remain BIT-IDENTICAL to v14 (full-PI1M
pretrain, seeds, folds, GNN config, GBM stack, ridge blend). No pseudo-labeling.

"Bit-identical" is verified CELL-LEVEL on the two source-sliced cells (CORE_A =
graph feats + GINE + MT-GNN; CORE_B = fold OOF + GBM stack): v15's CORE_A must
equal v14's CORE_A with ONLY the weight line boosted, and CORE_B must be an
exact string match.

Run with `python tests/test_v15_kaggle_nb.py` or pytest.
"""
import ast, os, pathlib, re, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN15 = REPO / "build_v15_kaggle_nb.py"
GEN14 = REPO / "build_v14_kaggle_nb.py"
NB15 = REPO / "PolyWin_R2_v15_epsnc_focus.ipynb"
NB14 = REPO / "PolyWin_R2_v14_p1m_pretrain.ipynb"

PLAIN_LINE = "g.w = torch.tensor([1.0 / freq[row.target_type]], dtype=torch.float)"
BOOST_LINE = ("g.w = torch.tensor([1.0 / freq[row.target_type] "
              "* TGT_FOCUS.get(row.target_type, 1.0)], dtype=torch.float)")


def _cell_list(nb_path):
    nb = nbformat.read(str(nb_path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _core_cells(cells):
    """Return (core_a, core_b) — the two source-sliced cells by unique markers."""
    a = next(c for c in cells if "# Graph featurization" in c
             and "# Twin source:" not in c)
    b = next(c for c in cells if "# Twin source:" in c and "lgb_test_te" in c)
    return a, b


def _build(generator, nb_path):
    env = dict(os.environ)
    subprocess.run([sys.executable, str(generator)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    cells = _cell_list(nb_path)
    code = "\n".join(cells)
    md = "\n".join(c.source for c in nbformat.read(str(nb_path), as_version=4).cells
                   if c.cell_type == "markdown")
    return code, md, cells


def _build15():
    return _build(GEN15, NB15)


def _build14():
    return _build(GEN14, NB14)


def test_v15_header_and_config():
    code, md, _ = _build15()
    assert "v15" in md and "EPS/NC FOCUS" in md
    assert 'TGT_FOCUS = {"eps": 2.0, "nc": 2.0}' in code

def test_v15_boost_line_present_plain_line_absent():
    code, _, _ = _build15()
    assert BOOST_LINE in code
    assert PLAIN_LINE not in code

def test_v15_core_a_is_v14_plus_single_boost():
    _, _, c15 = _build15()
    _, _, c14 = _build14()
    a15, _ = _core_cells(c15)
    a14, _ = _core_cells(c14)
    assert a14.count(PLAIN_LINE) == 1
    assert a15 == a14.replace(PLAIN_LINE, BOOST_LINE)

def test_v15_core_b_identical_to_v14():
    _, _, c15 = _build15()
    _, _, c14 = _build14()
    _, b15 = _core_cells(c15)
    _, b14 = _core_cells(c14)
    assert b15 == b14

def test_v15_no_pseudo_and_no_arch_change():
    code, _, _ = _build15()
    assert not re.search(r"\bpseudo[_-]?label", code, re.IGNORECASE)
    assert "class GINEEncoder(nn.Module):" in code
    assert "class MTGNN(nn.Module):" in code
    assert "PRETRAIN_EPOCHS = 10" in code
    assert "GNN_SEEDS = os.environ.get(\"GNN_SEEDS\", \"42,999,2025\")" in code

def test_v15_submission_names():
    code, _, _ = _build15()
    assert "submission_v15.csv" in code
    assert "blend_oof_test.npz" in code

def test_v15_all_cells_compile():
    for src in _cell_list(NB15):
        ast.parse(src)


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