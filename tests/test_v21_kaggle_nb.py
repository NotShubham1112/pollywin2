"""Tests for the v21 leak-safe sibling arm notebook builder
(build_v21_kaggle_nb.py).

v21 = P14 (bit-identical level-0: CORE_A graph feats + GINE pretrain + CORE_B
fold OOF + GBM trio stack) + ONE additive change: a leak-safe sibling Ridge arm
(SIB) as a THIRD column in P14's per-target Ridge blend, plus an in-notebook
gate report (gates 0-3).

"Bit-identical" is verified CELL-LEVEL on the two source-sliced cells: v21's
CORE_A and CORE_B must be exact string matches of v14's (both are extracted from
the same mt_gnn_v2.py markers). The SIB arm cell + replaced 3-arm blend cell are
the only functional changes; v14 cell 6 (the leaky sibling pivot + physics eps)
must be absent.

Run with `python tests/test_v21_kaggle_nb.py` or pytest.
"""
import ast, os, pathlib, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN21 = REPO / "src/notebook_builders/build_v21_kaggle_nb.py"
GEN14 = REPO / "src/notebook_builders/build_v14_kaggle_nb.py"
NB21 = REPO / "notebooks/v21_sibling_arm/PolyWin_R2_v21_sibling_arm.ipynb"
NB21_SMOKE = REPO / "PolyWin_R2_v21_sibling_arm_smoke.ipynb"
NB14 = REPO / "notebooks/v14_p14_baseline/PolyWin_R2_v14_p1m_pretrain.ipynb"

# v14 cell-6 leaky markers that v21 must NOT contain (design §4): the full-train
# true-label pivot table + physics-eps cell re-introduces the exact leak v21
# exists to remove.
LEAKY_MARKERS = ("train_pivot", "get_sibs", "submission_v17_final.csv",
                 "physics imputation", "A_phys")

# Input-dependency refs that would break self-containment on Kaggle. Note the
# notebook DOES write to `vault/pipeline_out_v21` on local runs (v14 fork
# behavior, output only) — that is not an input dependency and is allowed.
# The only permitted URL is the torch cu121 wheel index in the v13/v14
# CUDA-repair cell (a package-install URL, NOT a data dependency).
FORBIDDEN = ("superblend_oof.npz", "huggingface", "transformers",
             "wget", "curl")
ALLOWED_URL = "https://download.pytorch.org/whl/cu121"


def _cell_list(nb_path):
    nb = nbformat.read(str(nb_path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _core_cells(cells):
    """Return (core_a, core_b) — the two source-sliced cells by unique markers."""
    a = next(c for c in cells if "# Graph featurization" in c
             and "# Twin source:" not in c)
    b = next(c for c in cells if "# Twin source:" in c and "lgb_test_te" in c)
    return a, b


def _build21(smoke=False):
    env = dict(os.environ)
    if smoke:
        env["SMOKE"] = "1"
    else:
        env.pop("SMOKE", None)
    for k in ("GNN_SEEDS", "PRETRAIN_EPOCHS", "PRETRAIN_SAMPLE"):
        env.pop(k, None)
    nb_path = NB21_SMOKE if smoke else NB21
    subprocess.run([sys.executable, str(GEN21)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    cells = _cell_list(nb_path)
    code = "\n".join(cells)
    md = "\n".join(c.source for c in nbformat.read(str(nb_path), as_version=4).cells
                   if c.cell_type == "markdown")
    return code, md, cells


def _build14():
    env = dict(os.environ)
    for k in ("GNN_SEEDS", "PRETRAIN_EPOCHS", "PRETRAIN_SAMPLE"):
        env.pop(k, None)
    subprocess.run([sys.executable, str(GEN14)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    return _cell_list(NB14)


def test_v21_sib_cell_present():
    code, _, _ = _build21()
    assert "def build_sib_arm" in code
    assert "sib_only_r2" in code
    assert "def build_feats" in code
    # pre-registered alpha grid, same as P14
    assert "ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]" in code


def test_v21_blend_has_sib_column_and_w_sib():
    code, _, _ = _build21()
    assert "np.column_stack([oof_gbm_global, oof_mt_global, sib_oof])" in code
    assert "np.column_stack([test_gbm_global, test_mt_global, sib_test])" in code
    assert "w_sib" in code and "w_SIB" in code


def test_v21_core_a_core_b_bit_identical_to_v14():
    _, _, c21 = _build21()
    c14 = _build14()
    a21, b21 = _core_cells(c21)
    a14, b14 = _core_cells(c14)
    assert a21 == a14
    assert b21 == b14


def test_v21_leaky_cell_absent():
    code, md, _ = _build21()
    blob = code + "\n" + md
    for m in LEAKY_MARKERS:
        assert m not in blob, f"leaky v14 cell-6 marker present: {m}"


def test_v21_gates_pre_registered():
    code, md, _ = _build21()
    blob = code + "\n" + md
    # thresholds must be pre-registered and NOT softened
    assert "SOFT_DELTA = 0.0015" in code
    assert "STRONG_DELTA = 0.003" in code
    assert "WORST_TOL = 0.003" in code
    assert "+0.0015" in blob and "+0.003" in blob
    assert "\u22120.003" in blob          # unicode minus in the markdown gate table
    # gate 1 leak audit count
    assert "gate_1_leak_audit" in code and "leak_count" in code
    # tier strings
    low = blob.lower()
    assert "soft" in low and "strong" in low
    # final verdict line
    assert "GATE: " in code and "PASS" in code and "P14 stays final" in code


def test_v21_submission_unchanged():
    code, _, _ = _build21()
    assert "submission_v21.csv" in code
    assert '"id": tef["id"].values, "target": final_te' in code
    assert "index=False" in code
    assert "final_te = np.zeros(len(tef))" in code


def test_v21_all_cells_compile():
    _, _, c21 = _build21()
    for src in c21:
        ast.parse(src)
    _, _, c21s = _build21(smoke=True)
    for src in c21s:
        ast.parse(src)


def test_v21_smoke_subset():
    code, _, _ = _build21(smoke=True)
    assert "GLOBAL_FOLDS = 2" in code
    assert "PRETRAIN_SAMPLE = 2000" in code
    assert "PRETRAIN_EPOCHS = 1" in code
    assert "MAX_EPOCHS = 4" in code
    assert '"1"' in code          # GNN_SEEDS default -> "1"


def test_v21_full_build_is_full():
    code, _, _ = _build21(smoke=False)
    assert "GLOBAL_FOLDS = 5" in code
    assert "PRETRAIN_SAMPLE = 2000000" in code
    assert "PRETRAIN_EPOCHS = 10" in code
    assert '"42,999,2025"' in code


def test_v21_forbidden_refs():
    import re
    code, md, _ = _build21()
    blob = code + "\n" + md
    for f in FORBIDDEN:
        assert f not in blob, f"forbidden ref present: {f}"
    urls = set(re.findall(r"https?://[^\s\"')]+", blob))
    assert urls <= {ALLOWED_URL}, f"unexpected urls: {urls - {ALLOWED_URL}}"


def test_v21_sib_source_is_model_oof_not_labels():
    code, _, _ = _build21()
    # SIB features must come ONLY from the CORE_B twin source (model OOF preds),
    # never from true train labels.
    assert "twin_scores, lgb_test_te = leak_safe_oof_scores()" in code
    assert "twin_scores" in code and "lgb_test_te" in code
    assert "train_pivot" not in code


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
