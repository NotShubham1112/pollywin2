import ast
import os
import pathlib
import re
import subprocess
import sys

import nbformat

MOD = pathlib.Path(__file__).resolve().parents[1]
REPO = MOD.parent
GEN22 = MOD / "build_v22_kaggle_nb.py"
GEN14 = REPO / "build_v14_kaggle_nb.py"
NB22 = MOD / "PolyWin_R2_v22_bert_arm.ipynb"
NB22_SMOKE = MOD / "PolyWin_R2_v22_bert_arm_smoke.ipynb"
NB14 = REPO / "PolyWin_R2_v14_p1m_pretrain.ipynb"

LEAKY_MARKERS = ("train_pivot", "get_sibs", "submission_v17_final.csv",
                 "physics imputation", "A_phys")
FORBIDDEN = ("superblend_oof.npz", "huggingface", "transformers", "wget",
             "curl")
ALLOWED_URL = "https://download.pytorch.org/whl/cu121"


def _cell_list(nb_path):
    nb = nbformat.read(str(nb_path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _core_cells(cells):
    a = next(c for c in cells if "# Graph featurization" in c
             and "# Twin source:" not in c)
    b = next(c for c in cells if "# Twin source:" in c and "lgb_test_te" in c)
    return a, b


def _build22(smoke=False):
    env = dict(os.environ)
    if smoke:
        env["SMOKE"] = "1"
    else:
        env.pop("SMOKE", None)
    for k in ("GNN_SEEDS", "PRETRAIN_EPOCHS", "PRETRAIN_SAMPLE"):
        env.pop(k, None)
    nb_path = NB22_SMOKE if smoke else NB22
    subprocess.run([sys.executable, str(GEN22)], cwd=str(REPO), check=True,
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


def test_v22_sources_inlined():
    code, _, _ = _build22()
    for s in ("def learn_bpe", "def tokenize_batch", "class BertEncoder",
              "def pretrain_mlm", "def pool_embeddings",
              "def compute_bert_arm", "def blend_narm_oof",
              "def _p14_2arm_oof", "def gate_report", "def gate_1_leak_audit",
              "def write_submission"):
        assert s in code, f"v22 source not inlined: {s}"
    assert "PROTECTED = (\"*\", \"(\", \")\")" in code


def test_v22_blend_has_bert_column():
    code, _, _ = _build22()
    assert "np.column_stack([oof_gbm_global, oof_mt_global, oof_bert])" in code
    assert "w_bert" in code and "w_BERT" in code
    assert "bert_only_r2" in code


def test_v22_gates_pre_registered():
    code, md, _ = _build22()
    blob = code + "\n" + md
    assert "SOFT_DELTA = 0.0015" in code
    assert "STRONG_DELTA = 0.003" in code
    assert "WORST_TOL = 0.003" in code
    assert "EPS_NC_EI = (\"eps\", \"nc\", \"ei\")" in code
    assert "gate_1_leak_audit" in code and "leak_count" in code
    assert "GATE: " in code and "PASS" in code and "P14 stays final" in code


def test_v22_core_a_core_b_bit_identical_to_v14():
    _, _, c22 = _build22()
    c14 = _build14()
    a22, b22 = _core_cells(c22)
    a14, b14 = _core_cells(c14)
    assert a22 == a14
    assert b22 == b14


def test_v22_leaky_cell_absent():
    code, md, _ = _build22()
    blob = code + "\n" + md
    for m in LEAKY_MARKERS:
        assert m not in blob, f"leaky v14 cell-6 marker present: {m}"


def test_v22_forbidden_refs():
    code, md, _ = _build22()
    blob = code + "\n" + md
    for f in FORBIDDEN:
        assert f not in blob, f"forbidden ref present: {f}"
    urls = set(re.findall(r"https?://[^\s\"')]+", blob))
    assert urls <= {ALLOWED_URL}, f"unexpected urls: {urls - {ALLOWED_URL}}"


def test_v22_submission_unchanged():
    code, _, _ = _build22()
    assert "submission_v22.csv" in code
    assert '"id": tef["id"].values, "target": final_te' in code
    assert "index=False" in code
    assert "final_te = np.zeros(len(tef))" in code


def test_v22_all_cells_compile():
    _, _, c22 = _build22()
    for src in c22:
        ast.parse(src)
    _, _, c22s = _build22(smoke=True)
    for src in c22s:
        ast.parse(src)


def test_v22_smoke_subset():
    code, _, _ = _build22(smoke=True)
    assert "GLOBAL_FOLDS = 2" in code
    assert "PRETRAIN_SAMPLE = 2000" in code
    assert "PRETRAIN_EPOCHS = 1" in code
    assert "V22_PI_COUNT = 2000" in code
    assert "V22_D = 32" in code
    assert "V22_LAYERS = 2" in code
    assert 'GNN_SEEDS = "1"' in code


def test_v22_full_build_is_full():
    code, _, _ = _build22(smoke=False)
    assert "GLOBAL_FOLDS = 5" in code
    assert "PRETRAIN_SAMPLE = 2000000" in code
    assert "V22_D = 384" in code
    assert "V22_LAYERS = 6" in code
    assert "V22_HEADS = 8" in code
    assert "V22_PI_COUNT = -1" in code
    assert "V22_VOCAB = 4000" in code
    assert 'GNN_SEEDS = "42,999,2025"' in code


def test_v22_self_contained_inputs():
    code, md, _ = _build22()
    blob = code + "\n" + md
    assert "find_input(INP" in code
    assert "PI1M.csv" in code and "train.csv" in code and "test.csv" in code
    assert 'INP = "/kaggle/input"' in code
    assert "competitions" in code
