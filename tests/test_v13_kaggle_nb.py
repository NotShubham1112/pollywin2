"""Tests for the v13 Small-Five Specialist notebook generator stage
(build_v12_kaggle_nb.py).

The same generator emits three notebooks; POLYWIN_STAGE=v13 must produce
PolyWin_R2_v13_specialist.ipynb with the v13 leakage/specialist/blend/decision
cells and no v12 bucket-MoE cells. Run with `python tests/test_v13_kaggle_nb.py`
or pytest.
"""
import os, pathlib, re, subprocess, sys
import nbformat

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "PolyWin_R2_v13_specialist.ipynb"
GEN = REPO / "build_v12_kaggle_nb.py"

def _build():
    env = dict(os.environ, POLYWIN_STAGE="v13")
    subprocess.run([sys.executable, str(GEN)], cwd=str(REPO), check=True,
                   capture_output=True, text=True, env=env)
    nb = nbformat.read(str(NB), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    return code, md, nb

def _stage_literal(nb):
    return any('STAGE = "v13"' in "".join(c.source)
               for c in nb.cells if c.cell_type == "code")

def test_v13_header_and_stage():
    code, md, nb = _build()
    assert "Small-Five Specialist" in md
    assert "Question" in md and "Protocol (honest)" in md
    assert "fold-safe leakage" in md
    assert "physics-residual" in md or "physics residuals" in md
    assert "USE_SPECIALIST = mean_specialist_blend_oof >= mean_v11_blend_oof" in md
    assert _stage_literal(nb)

def test_v13_bootstrap_shared_arm():
    code, md, nb = _build()
    assert 'STAGE = "v13"' in code
    assert "GLOBAL_FOLDS = 3 if SMOKE else 10" in code
    assert "PRETRAIN_EPOCHS" in code and "PRETRAIN_SAMPLE" in code
    assert "def canon_key(smiles):" in code
    assert "def get_splits(tt):" in code
    assert "class GINEEncoder(nn.Module):" in code
    assert "def pretrain(epochs=PRETRAIN_EPOCHS" in code
    assert "pretrained_encoder.pt" in code

def test_v13_leakage_fold_safe():
    code, _, _ = _build()
    assert 'SMALL_FIVE = ["eps", "nc", "ei", "eea", "egb"]' in code
    assert "def build_pivot(df):" in code
    assert "FULL_PIVOT = build_pivot(dedup)" in code
    assert 'FOLD_PIVOTS = {f: build_pivot(dedup[dedup["fold"] != f]) for f in range(folds.max() + 1)}' in code
    assert "def impute_value(tt, known):" in code
    assert 'return known["ei"] - known["eea"]' in code
    assert 'return known["egc"] - 0.10' in code
    assert 'return known["nc"] ** 2' in code
    assert "def pivot_known(canon, pivot):" in code
    assert "def leak_vec(canon, tt, pivot):" in code
    assert "vec += [known[ct], 1.0]" in code
    assert "vec.append(imp if not np.isnan(imp) else TMEAN[tt])" in code

def test_v13_leakage_only_baseline():
    code, _, _ = _build()
    assert "CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, l2_leaf_reg=3.0" in code
    assert "leak_oof[tt] = oof; leak_te[tt] = te_pred" in code
    assert "v13_leak_only_compare.csv" in code
    assert "for tt in SMALL_FIVE:" in code

def test_v13_physics_residuals_scaled():
    code, _, _ = _build()
    assert "PHYS_W = 0.075" in code
    assert "def physics_residual_loss(model, pred, batch):" in code
    assert "def real(t, i):" in code
    assert "target_stats[t]" in code
    assert "/ 5.0) ** 2" in code
    assert "/ 10.0) ** 2" in code
    assert "/ 2.0) ** 2" in code
    assert "phys_delta" in code

def test_v13_specialist_heads():
    code, _, _ = _build()
    assert "N_EXTRA = 2 * (len(TARGETS) - 1) + 2" in code
    assert 'LOSS_W = {"tg": 1.0, "egc": 1.0, "egb": 1.5, "eps": 3.0, "nc": 2.5, "ei": 3.0, "eea": 2.0}' in code
    assert "class SpecialistModel(nn.Module):" in code
    assert "self.big_heads = nn.ModuleDict({" in code
    assert "self.small_heads = nn.ModuleDict({" in code
    assert 'for t in SMALL_FIVE}' in code
    assert "def load_encoder(self, state_dict):" in code

def test_v13_specialist_graphs_and_train():
    code, _, _ = _build()
    assert "def build_spec_graphs(smiles_list, target_idx, y_vals, w_vals, canon_ids, row_ids):" in code
    assert "g.row_id = rid" in code
    assert "spec_train_graphs = build_spec_graphs(" in code
    assert "spec_test_graphs = build_spec_graphs(" in code
    assert "def train_specialist(init_state, epochs=MINI_EPOCHS" in code
    assert "leak_vec(row.canon, tt, pivot)" in code
    assert "g.extra = torch.tensor(leak_vec(test.loc[g.row_id, \"canon\"], tt, FULL_PIVOT)" in code
    assert "F.mse_loss(pred_n, y_n, reduction=\"none\") * batch.w * lw" in code
    assert "loss = main + phys_w * phys" in code
    assert "model(gb)[0, ti].item() * std_ + mean_" in code

def test_v13_specialist_skip_and_outputs():
    code, _, _ = _build()
    assert "spec_oof = spec_oof_nl = spec_te = spec_te_nl = None" in code
    assert "Specialist SKIPPED (no pretrained encoder available)" in code
    assert "spec_oof, spec_oof_nl, spec_te, spec_te_nl = train_specialist(init_state=_init)" in code
    assert "v13_specialist_compare.csv" in code

def test_v13_blend_seven_candidates_and_floor():
    code, _, _ = _build()
    assert "from scipy.optimize import nnls" in code
    assert "def cand_arrays(tt):" in code
    assert 'oofs["specialist"] = spec_oof[get_splits(tt)[0]]' in code
    assert 'oofs["specialist_no_leak"] = spec_oof_nl[get_splits(tt)[0]]' in code
    assert 'oofs["leakage_only"] = leak_oof[tt]' in code
    assert 'oofs["imputed"] = imputed_oof[tt]' in code
    assert 'oofs["stack"] = FINAL_OOF[tt]' in code
    assert 'oofs["gnn"] = gnn_oof_df' in code
    assert 'oofs["v11_blend"] = v11_blend_oof[tt]' in code
    assert "w, _ = nnls(Ztr[fin], y_tt[tr_l][fin])" in code
    assert "v13_blend_oof[tt] = oof; v13_blend_te[tt] = fold_te" in code
    assert 'if spec_oof is not None:' in code

def test_v13_fold_safe_imputation():
    code, _, _ = _build()
    assert "imputed_oof = {}; imputed_te = {}" in code
    assert "pivot = FOLD_PIVOTS[f]" in code
    assert "impute_value(tt, known)" in code
    assert "for tt in SMALL_FIVE:" in code

def test_v13_decision_and_submission():
    code, _, _ = _build()
    assert "USE_SPECIALIST = mean_v13 >= mean_v11" in code
    assert "v13_compare = pd.DataFrame(v13_rows)" in code
    assert "v13_compare.to_csv(os.path.join(WORK, \"v13_compare.csv\"), index=False)" in code
    assert "FINAL_TE[tt] = v13_blend_te[tt]" in code
    assert "FINAL_TE[tt] = v11_blend_te[tt]" in code
    assert "submission = v11 blend (floor)" in code
    assert "submission saved:" in code
    assert "==== PIPELINE COMPLETE ====" in code

def test_v13_figure():
    code, _, _ = _build()
    assert "25_v13_specialist.png" in code
    assert "v13 Small-Five Specialist vs stack / GNN / v11 blend" in code
    assert "def savefig(fig, name):" in code

def test_v13_no_trained_gate_no_bucket():
    code, _, _ = _build()
    assert not re.search(r"\bgate\b", code, re.IGNORECASE)
    assert "BUCKET_KS" not in code
    assert "ROUTER_COLS" not in code
    assert "v12_bucket_compare.csv" not in code
    assert "KMeans(n_clusters=K, random_state=42, n_init=10)" not in code

def test_v13_all_cells_compile():
    import ast
    _, _, nb = _build()
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
