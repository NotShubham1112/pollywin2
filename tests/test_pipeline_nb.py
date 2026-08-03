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
    assert "def get_torch_device" in code
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
    assert "{a for ring in ri.AtomRings() for a in ring}" in code

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

def test_tgnn_cell():
    code, _ = _build()
    assert "class TgNN" in code
    assert "def tgnn_fit_predict" in code
    assert "save_oof_artifact(\"tgnn\"" in code

def test_gnn_removed():
    code, _ = _build()
    assert "class GINConv" not in code
    assert "PolymerGNN" not in code
    assert "GNN branch archived" in code

def test_stacking_levels():
    code, _ = _build()
    assert "BASE_MODELS_ELEC" in code
    assert "BASE_MODELS_TG" in code
    assert "CROSS_MAP" in code
    assert "FINAL_OOF" in code
    assert "l15_ridge.parquet" in code
    assert "final_meta.parquet" in code

def test_cross_te_features_matches_oof_cols():
    code, _ = _build()
    oof_block = code.split("def cross_oof_features")[1].split("def cross_te_features")[0]
    te_block = code.split("def cross_te_features")[1]
    assert 'cols += [f"cross_{ct}", f"cross_{ct}_miss"]' in oof_block
    assert 'cols += [f"cross_{ct}", f"cross_{ct}_miss"]' in te_block

def test_gbm_artifact_save():
    code, _ = _build()
    assert "model_oof" in code
    assert "save_oof_artifact(name, model_oof[name], model_te[name])" in code

def test_fig08_in_notebook():
    code, _ = _build()
    assert "cross_target_corr.csv" in code
    assert "vault/figures" not in code
    assert "FINAL_OOF.get" in code

def test_submission_uses_final():
    code, _ = _build()
    assert "final[m_te] = FINAL_TE[tt][m_te]" in code

def test_artifact_helper_train_test_separate():
    """save_oof_artifact must not pack train-aligned (dedup_index/oof) and
    test-aligned (test_pred) arrays into one DataFrame: dedup and test row counts
    differ per target, which raised 'All arrays must be the same length' on the
    first smoke run."""
    nb = nbformat.read(str(NB), as_version=4)
    src = next(c.source for c in nb.cells
               if c.cell_type == "code" and "def save_oof_artifact" in c.source)
    assert '"subset": "train"' in src
    assert '"subset": "test"' in src
    oof_at = src.find('"oof": np.asarray(oof_map[tt])')
    te_at = src.find('"test_pred": np.asarray(te_map[tt])[m_te]')
    assert oof_at != -1 and te_at != -1
    assert '"subset"' in src[oof_at:te_at]  # separate DataFrame block starts between them

def test_run_nns_gate():
    code, md = _build()
    assert "RUN_NNS = False" in code
    assert "if RUN_NNS:" in code
    assert 'BASE_MODELS_ELEC = ["lgb","cat","xgb","hgb","efn"] if RUN_NNS else ["lgb","cat","xgb","hgb"]' in code
    assert 'BASE_MODELS_TG = ["lgb","cat","xgb","hgb","tgnn"] if RUN_NNS else ["lgb","cat","xgb","hgb"]' in code
    assert "archived" in md

def test_header_v7_architecture():
    code, md = _build()
    assert "Retrieval Memory" in md
    assert "ablation" in md.lower()

def test_layer3_global_sim():
    code, _ = _build()
    assert 'S_tr = (1.0 - cdist(retr_tr, retr_tr, metric="jaccard")).astype(np.float32)' in code
    assert 'S_te = (1.0 - cdist(retr_te, retr_tr, metric="jaccard")).astype(np.float32)' in code
    assert "def topk_from_sim" in code
    assert "K_RETR" in code
    assert "def wmean_sq" in code

def test_layer3_pool_column_grid():
    code, _ = _build()
    assert "RETR_COLS_A" in code and "RETR_COLS_B" in code and "RETR_COLS_C" in code
    assert "assert len(RETR_COLS_A) + len(RETR_COLS_B) + len(RETR_COLS_C) == 57" in code
    assert '"g_exact_twin"' in code
    assert '"st_tgt_wmean_sq"' in code
    assert '"ct_{t}_count"' in code
    assert '"ct_{t}_wmean_sq"' in code
    assert 'TARGETS = ["tg","egc","egb","eps","nc","ei","eea"]' in code

def test_layer3_fold_safety_persistence():
    code, _ = _build()
    assert "S_m[:, folds == f] = -1.0" in code
    assert "cand_tt" in code
    assert "retrieval_audit.csv" in code
    assert "retrieval_test_features" in code

def test_retrieval_persistence():
    code, _ = _build()
    assert "Xtr_retr.parquet" in code
    assert "Xte_retr.parquet" in code
    assert "Xtr_full.pkl" in code

def test_pool_c_gets_2d_sim10():
    code, _ = _build()
    assert "sim10 = np.take_along_axis(S_m, idx10, axis=1)" in code
    assert "build_pool_c(sim10, idx10, target_vals)" in code
    assert "build_pool_c(g[\"g_top10_sim\"]" not in code

def test_ablation_cell():
    code, _ = _build()
    assert "ablation_lgb.csv" in code
    assert "ablation_density.csv" in code
    assert "retrieval_gain_share" in code
    assert '("base", BASE_COLS), ("full", list(Xtr.columns)), ("retr", RETR_ALL_COLS)' in code
    assert "importance_full" in code
    assert "rmse_metric(Y[sel]" in code

def test_figures_10_14_and_overlap():
    code, _ = _build()
    for n in ["10_similarity_dist.png", "11_neighbor_density.png", "12_exact_twin_freq.png",
              "13_retrieval_gain_vs_density.png", "14_sim_vs_oof_error.png"]:
        assert n in code
    assert "Shared-polymer count" in code

def test_figures_15_20():
    code, _ = _build()
    for n in ["15_ablation_base_full_retr.png", "16_retrieval_delta.png", "17_oof_pred_corr.png",
              "18_retrieval_feat_importance.png", "19_pool_contribution.png", "20_lb_progression.png"]:
        assert n in code
    assert 'pivot(index="target", columns="arm", values="rmse").reindex(TARGETS)' in code

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
