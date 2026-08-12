"""Tests for the v20 self-contained Kaggle notebook builder
(build_v20_kaggle_nb.py).

The v20 notebook mirrors the v14 P14 kernel structure (IMPORTS -> DATA ->
CORE_A -> GINE pretrain on PI1M -> CORE_B -> arms/blend/gate/submission)
but:
  * RECOMPUTES the P14 arms (oof_gbm/oof_mt/test_gbm/test_mt) IN-KERNEL from
    train.csv/test.csv/PI1M.csv ONLY (no superblend_oof.npz, no externals),
  * inlines the v20 module sources (codec/encoder/arm_cv/blend/gate_report),
  * writes submission.csv ONLY if the pre-registered gate passes, else prints
    GATE=FAIL -> P14 stays final.

Run with pytest or `python tests/test_v20_nb.py`.
"""
import os, pathlib, re, sys, tempfile
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import build_v20_kaggle_nb as B

GATE_STRINGS = ("mean_delta >= 0.003", "THR_MEAN = 0.003", "ALPHA_CAP = 0.30",
                "thr_mean=0.003", "alpha_cap=0.30")
FORBIDDEN = ("huggingface", "transformers", "wget", "curl", "http://", "https://",
             "superblend_oof.npz", "competition/data/raw", "vault/pipeline_out_pretrain",
             "D:\\", "C:\\")


def _build(tmp, smoke=False):
    out = pathlib.Path(tmp) / ("v20_smoke.ipynb" if smoke else "v20_full.ipynb")
    nb = B.build(str(out), smoke=smoke)
    assert out.exists()
    nb2 = nbformat.read(str(out), as_version=4)
    code = "\n".join(c.source for c in nb2.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb2.cells if c.cell_type == "markdown")
    return nb, nb2, code, md


def test_v20_nb_minimal_shape(tmp_path):
    nb, _, code, md = _build(tmp_path)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 10
    assert "v20" in md.lower() and "GATE" in md
    assert "blend_3d" in code
    assert '"id"' in code and '"target"' in code        # P14 writer header
    assert "torch.manual_seed" in code
    assert "np.random.seed" in code
    assert "V20_SEED" in code


def test_v20_nb_kaggle_input_only(tmp_path):
    _, _, code, _ = _build(tmp_path)
    # data is read ONLY from the competition input (root base + every mount layout)
    assert "train.csv" in code and "test.csv" in code and "PI1M.csv" in code
    assert "INP = \"/kaggle/input\"" in code
    assert "ppp-round-2" in code          # probes the <slug> and competitions/<slug> mounts
    assert "competitions" in code
    assert "find_input" in code


def test_v20_nb_no_forbidden_refs(tmp_path):
    _, _, code, md = _build(tmp_path)
    blob = code + "\n" + md
    low = blob.lower()
    for f in FORBIDDEN:
        assert f not in blob, f"forbidden ref present: {f}"
    assert "find_input(INP" in code


def test_v20_nb_recompute_p14_arms(tmp_path):
    _, _, code, _ = _build(tmp_path)
    # the P14 arms must be RECOMPUTED in-kernel from CORE_A/CORE_B
    assert "stack_oof[t] = oof; stack_test[t] = te_pred" in code
    assert "mt_oof[" in code and "mt_test" in code
    assert "oof_gbm_global = np.full(len(X), np.nan" in code
    assert "oof_mt_global = np.full(len(X), np.nan" in code
    assert "test_gbm_global" in code and "test_mt_global" in code
    assert "assert not np.isnan(oof_gbm_global).any()" in code


def test_v20_nb_v20_sources_inlined(tmp_path):
    _, _, code, _ = _build(tmp_path)
    for s in ("def build_tokenizer", "def tokenize_batch", "class MaskEncoder",
              "def pool_embeddings", "def pretrain_encoder",
              "def compute_trf_arm", "def blend_3d", "def compute_gate_report",
              "def write_submission", "def _p14_2arm_oof"):
        assert s in code, f"v20 source not inlined: {s}"


def test_v20_nb_gate_and_submission(tmp_path):
    _, _, code, _ = _build(tmp_path)
    for g in GATE_STRINGS:
        assert g in code
    assert "GATE: " in code and "PASS" in code and "FAIL" in code
    assert "GATE=FAIL -> P14 stays final" in code          # fail path
    assert "write_submission(" in code
    assert "submission.csv" in code
    assert "index=False" in code
    assert "report['pass']" in code
    # end-of-notebook globals per the brief
    assert "oof_v20, test_pred, GATE" in code or "oof_v20" in code
    assert "test_pred" in code and "GATE" in code
    # pre-registered threshold values (do not soften)
    assert "THR_MEAN = 0.003" in code and "THR_WORST = 0.003" in code


def test_v20_nb_all_cells_compile(tmp_path):
    import ast
    _, nb, _, _ = _build(tmp_path)
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)


def test_v20_nb_roundtrip_and_validate(tmp_path):
    _, _, code, _ = _build(tmp_path)
    out = pathlib.Path(tmp_path) / "v20_full.ipynb"
    back = nbformat.read(str(out), as_version=4)
    assert back.nbformat == 4
    # structural validity
    nbformat.validate(back)
    assert len([c for c in back.cells]) == len(back.cells)
    assert any(c.cell_type == "markdown" for c in back.cells)


def test_v20_nb_smoke_build_is_small(tmp_path):
    _, nb, code, _ = _build(tmp_path, smoke=True)
    # smoke must shorten pretrain so a CPU exec is feasible
    assert "GLOBAL_FOLDS = 2" in code
    assert "PRETRAIN_SAMPLE = 2000" in code
    assert "V20_PI_COUNT = 300" in code
    assert "V20_EPOCHS = 1" in code
    assert "GNN_SEEDS = \"1\"" in code
    nbformat.validate(nb)


def test_v20_nb_full_build_is_full(tmp_path):
    _, nb, code, _ = _build(tmp_path, smoke=False)
    assert "GLOBAL_FOLDS = 5" in code
    assert "PRETRAIN_SAMPLE = 2000000" in code
    assert "V20_D = 256" in code
    assert "V20_LAYERS = 4" in code
    assert "GNN_SEEDS = \"42,999,2025\"" in code


if __name__ == "__main__":
    import traceback
    failed = 0
    import tempfile as _tf
    _td = _tf.mkdtemp(prefix="v20nb_")
    for _n, _fn in list(globals().items()):
        if _n.startswith("test_") and callable(_fn):
            try:
                _fn(pathlib.Path(_td) / _n)
                print("PASS", _n)
            except Exception as _e:
                failed += 1
                print("FAIL", _n, "->", _e)
                traceback.print_exc()
    sys.exit(1 if failed else 0)