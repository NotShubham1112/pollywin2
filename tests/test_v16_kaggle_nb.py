"""Tests for the v16 Cross-Target Decoder notebook generator
(build_v16_kaggle_nb.py).

v16 = v14 + ONE functional stage: a cross-target decoder cell (physics_arm +
learned_arm, embedded VERBATIM from decoder_v16.py) plus a widened 4-arm
per-target Ridge blend [GBM, MT-GNN, phys, learned]. Everything else —
full-PI1M pretrain, GNN config, GBM stack, base-arm OOFs — must remain
BIT-IDENTICAL to v14 (CORE_A / CORE_B cell-level checks).

Run with `python tests/test_v16_kaggle_nb.py` or pytest.
"""
import ast, os, pathlib, re, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN16 = REPO / "build_v16_kaggle_nb.py"
GEN14 = REPO / "build_v14_kaggle_nb.py"
NB16 = REPO / "PolyWin_R2_v16_cross_target_decoder.ipynb"
NB14 = REPO / "PolyWin_R2_v14_p1m_pretrain.ipynb"


def _cell_list(nb_path):
    nb = nbformat.read(str(nb_path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _build(generator, nb_path):
    env = dict(os.environ)
    subprocess.run([sys.executable, str(generator)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    cells = _cell_list(nb_path)
    code = "\n".join(cells)
    md = "\n".join(c.source for c in nbformat.read(str(nb_path), as_version=4).cells
                   if c.cell_type == "markdown")
    return code, md, cells


def _build16():
    return _build(GEN16, NB16)


def _build14():
    return _build(GEN14, NB14)


def test_v16_header_and_config():
    code, md, _ = _build16()
    assert "v16" in md and "CROSS-TARGET DECODER" in md.upper()
    assert "submission_v16.csv" in code
    assert "blend_oof_test16.npz" in code
    assert "v16_blend_report.csv" in code
    assert "pipeline_out_v16" in code


def test_v16_cores_identical_to_v14():
    _, _, c16 = _build16()
    _, _, c14 = _build14()
    a16 = next(c for c in c16 if "# Graph featurization" in c
               and "# Twin source:" not in c)
    b16 = next(c for c in c16 if "# Twin source:" in c and "lgb_test_te" in c)
    a14 = next(c for c in c14 if "# Graph featurization" in c
               and "# Twin source:" not in c)
    b14 = next(c for c in c14 if "# Twin source:" in c and "lgb_test_te" in c)
    assert a16 == a14
    assert b16 == b14


def test_v16_decoder_cell_verbatim_from_module():
    src = (REPO / "decoder_v16.py").read_text(encoding="utf-8").rstrip("\n")
    _, _, c16 = _build16()
    assert any(c.rstrip("\n") == src for c in c16), "decoder module must be one verbatim cell"


def test_v16_decoder_present_and_between_core_b_and_blend():
    code, _, cells = _build16()
    assert "TARGETS_DEC" in code
    assert "PHYS_RECIPE" in code
    assert "physics_arm" in code
    assert "learned_arm" in code
    i_a = next(i for i, c in enumerate(cells) if "# Graph featurization" in c)
    i_b = next(i for i, c in enumerate(cells)
               if "# Twin source:" in c and "lgb_test_te" in c)
    i_dec = next(i for i, c in enumerate(cells) if "# Twin source:" not in c
                 and "TARGETS_DEC" in c and "physics_arm" in c and "learned_arm" in c)
    i_blend = next(i for i, c in enumerate(cells) if "blend_oof_test16.npz" in c)
    assert i_a < i_b < i_dec < i_blend


def test_v16_four_arm_outputs():
    code, _, _ = _build16()
    assert "oof_phys" in code and "oof_learn" in code
    assert "test_phys" in code and "test_learn" in code
    assert "w_PH" in code and "w_LEARN" in code
    assert "blends_tr" in code and "blends_te" in code


def test_v16_no_pseudo_and_no_arch_change():
    code, _, _ = _build16()
    assert not re.search(r"\bpseudo[_-]?label", code, re.IGNORECASE)
    assert "class GINEEncoder(nn.Module):" in code
    assert "class MTGNN(nn.Module):" in code
    assert "PRETRAIN_EPOCHS = 10" in code
    assert "GNN_SEEDS = os.environ.get(\"GNN_SEEDS\", \"42,999,2025\")" in code


def test_v16_all_cells_compile():
    for src in _cell_list(NB16):
        ast.parse(src)


def test_v16_physics_calls_fold_safe():
    code, _, _ = _build16()
    assert "group=G" in code
    assert "global_folds=GLOBAL_FOLDS" in code


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