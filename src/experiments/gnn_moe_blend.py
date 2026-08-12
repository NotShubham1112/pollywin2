"""MoE-style per-target blend: v6 stack (rebuilt on cached BASE features) + GNN arm.

Motivation (from scaffold validator): GNN random-OOF (0.847) overstates
generalisation — honest shift-weighted test R2 ~0.843, on par with the standing
stack. But GNN is *stronger* than the stack exactly where the stack is weak and
shift ~0 (egb/eea/eps). This script builds the per-target expert blend:

  final_tt = w_tt * stack_tt + (1 - w_tt) * gnn_tt

with w_tt chosen **fold-safely** per target (tuned on the 9 training folds'
(stack, gnn) OOF pairs, evaluated on the held-out fold) -> an honest blend OOF
free of weight-tune optimism. Mirrors the pipeline's stacking folds exactly
(shared global GroupKFold on `canon`, 10 folds).

Steps:
  1. rebuild v6 stack OOF + test on cached Xtr.pkl/Xte.pkl (BASE cols = all cols,
     no retrieval) -> 4 GBMs per target + L1.5 Ridge + L2 meta (reliability +
     cross-target features), exactly like build_pipeline_nb.py Layer 4/9.
  2. load GNN OOF/test (gnn_oof.csv / gnn_test.csv), align on dedup/test index.
  3. per-target fold-safe weight search -> blend OOF R2 vs stack-only vs gnn-only.
  4. emit gnn_moe_compare.csv + gnn_moe_test.csv + blended submission.

Usage (Miniconda3 python):
  python gnn_moe_blend.py
"""

import os, time
import numpy as np
import pandas as pd
import lightgbm as lgbm
import xgboost as xgbm
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error as rmse_metric

WORK = r"vault\pipeline_out"
GNN_DIR = os.path.join(WORK, "gnn_arm")
OUT = GNN_DIR

TARGETS = ["tg", "egc", "egb", "eps", "nc", "ei", "eea"]
CROSS_MAP = {
    "eps": ["nc", "egc", "egb", "eea"],
    "nc": ["eps", "egb", "egc", "ei"],
    "egc": ["egb", "eea", "nc", "eps", "ei"],
    "egb": ["egc", "nc", "eea", "eps", "ei"],
    "ei": ["egc", "egb", "nc"],
    "eea": ["egc", "egb", "eps"],
    "tg": [],
}
SEED = 42


def make_lgb():
    return lgbm.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
                              reg_alpha=0.3, reg_lambda=1.0, min_child_samples=10,
                              random_state=SEED, verbose=-1, n_jobs=-1)


def make_cat():
    return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                             l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                             allow_writing_files=False)


def make_xgb():
    return xgbm.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                             subsample=0.85, colsample_bytree=0.7, reg_alpha=0.3,
                             reg_lambda=1.0, random_state=SEED, verbosity=0, n_jobs=-1)


def make_hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         random_state=SEED, l2_regularization=1.0)


def main():
    t0 = time.time()
    Xtr = pd.read_pickle(os.path.join(WORK, "Xtr.pkl"))
    Xte = pd.read_pickle(os.path.join(WORK, "Xte.pkl"))
    dedup = pd.read_pickle(os.path.join(WORK, "dedup.pkl"))
    test = pd.read_pickle(os.path.join(WORK, "test.pkl"))
    folds = dedup["fold"].to_numpy()
    Y = dedup["target"].values
    MODEL_COLS = list(Xtr.columns)  # BASE = v6 cols (Xtr.pkl has no retrieval cols)
    print(f"Xtr {Xtr.shape} Xte {Xte.shape} | {len(MODEL_COLS)} BASE cols | folds {folds.max()+1}")

    def get_splits(tt):
        m = (dedup["target_type"] == tt).values
        idx = np.where(m)[0]
        splits = []
        for f in range(folds.max() + 1):
            fold_mask = (folds[m] == f)
            va = idx[fold_mask]
            tr = idx[~fold_mask]
            if len(va) > 0 and len(tr) > 0:
                splits.append((tr, va))
        return m, idx, splits

    # ---- rebuild stack: 4 GBMs per target (GroupKFold OOF) ----
    oof_store, test_store = {}, {}
    CHK = os.path.join(OUT, "moe_gbm_chk.parquet")
    if os.path.exists(CHK):
        chk = pd.read_parquet(CHK)
        for _, r in chk.iterrows():
            oof_store[(r["key"], r["target"])] = r["oof"]
            test_store[(r["key"], r["target"])] = r["test_pred"]
        print(f"loaded {len(chk)} GBM checkpoints from {CHK}")
    else:
        print("Rebuilding GBM experts (v6 BASE cols)...")
        all_parts = []
        for tt in TARGETS:
            m, idx, splits = get_splits(tt)
            parts = []
            for name, mk in [("lgb", make_lgb), ("cat", make_cat),
                             ("xgb", make_xgb), ("hgb", make_hgb)]:
                oof = np.zeros(m.sum())
                te_pred = np.zeros(len(Xte))
                for tr, va in splits:
                    mdl = mk()
                    mdl.fit(Xtr.iloc[tr][MODEL_COLS], Y[tr])
                    oof[np.where(m)[0].searchsorted(va)] = mdl.predict(Xtr.iloc[va][MODEL_COLS])
                    te_pred += mdl.predict(Xte[MODEL_COLS]) / len(splits)
                oof_store[(name + "_" + tt, tt)] = oof
                test_store[(name + "_" + tt, tt)] = te_pred
                parts.append(pd.DataFrame({"key": name + "_" + tt, "target": tt,
                                           "oof": [oof], "test_pred": [te_pred]}))
            all_parts.extend(parts)
            # persist cumulative table so a mid-run kill keeps every finished target
            pd.concat(all_parts, ignore_index=True).to_parquet(CHK, index=False)
            print(f"  {tt}: {time.time()-t0:.0f}s elapsed, chk={len(all_parts)} rows", flush=True)

    # ---- L1.5 Ridge stack ----
    L15_OOF, L15_TE = {}, {}
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        cols = [b + "_" + tt for b in ("lgb", "cat", "xgb", "hgb")]
        Z = np.column_stack([oof_store[(c, tt)] for c in cols])
        Zte = np.column_stack([test_store[(c, tt)] for c in cols])
        pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
        oof = np.zeros(m.sum()); te_pred = np.zeros(len(Zte))
        for tr, va in splits:
            tr_l, va_l = pos[tr], pos[va]
            sr = StandardScaler().fit(Z[tr_l])
            meta = Ridge(alpha=10.0).fit(sr.transform(Z[tr_l]), Y[idx][tr_l])
            oof[va_l] = meta.predict(sr.transform(Z[va_l]))
            te_pred += meta.predict(sr.transform(Zte)) / len(splits)
        L15_OOF[tt] = oof; L15_TE[tt] = te_pred

    # ---- L2 meta (reliability + cross-target), fold-safe ----
    def reliability(Z):
        return np.column_stack([Z.mean(1), Z.std(1), Z.max(1), Z.min(1)])

    def cross_oof_features(tt, dedup_, L15_OOF_):
        m_tt = (dedup_["target_type"] == tt).values
        feats, cols = [], []
        for ct in CROSS_MAP[tt]:
            m_ct = (dedup_["target_type"] == ct).values
            c2o = dict(zip(dedup_.loc[m_ct, "canon"].values, L15_OOF_[ct]))
            vals = np.array([c2o.get(c, np.nan) for c in dedup_.loc[m_tt, "canon"].values],
                            dtype=np.float32)
            miss = np.isnan(vals).astype(np.float32)
            vals = np.nan_to_num(vals, nan=float(np.nanmean(L15_OOF_[ct])))
            feats += [vals, miss]; cols += [f"cross_{ct}", f"cross_{ct}_miss"]
        if not feats:
            return None, None
        return np.column_stack(feats), cols

    def cross_te_features(tt):
        feats = []
        for ct in CROSS_MAP[tt]:
            feats.append(np.asarray(L15_TE[ct], dtype=np.float32))
            feats.append(np.zeros(len(test), dtype=np.float32))
        if not feats:
            return None
        return np.column_stack(feats)

    FINAL_OOF, FINAL_TE = {}, {}
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        cols = [b + "_" + tt for b in ("lgb", "cat", "xgb", "hgb")]
        Z1 = np.column_stack([oof_store[(c, tt)] for c in cols])
        Zrel = reliability(Z1)
        Zcr, ccr = cross_oof_features(tt, dedup, L15_OOF)
        Z2 = np.column_stack([Z1, Zrel] + ([Zcr] if Zcr is not None else []))
        Zte1 = np.column_stack([test_store[(c, tt)] for c in cols])
        Zte_rel = reliability(Zte1)
        Zte_cr = cross_te_features(tt)
        Zte2 = np.column_stack([Zte1, Zte_rel] + ([Zte_cr] if Zte_cr is not None else []))
        pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
        oof = np.zeros(m.sum()); te_pred = np.zeros(len(Zte2))
        for tr, va in splits:
            tr_l, va_l = pos[tr], pos[va]
            sr = StandardScaler().fit(Z2[tr_l])
            meta = Ridge(alpha=10.0).fit(sr.transform(Z2[tr_l]), Y[idx][tr_l])
            oof[va_l] = meta.predict(sr.transform(Z2[va_l]))
            te_pred += meta.predict(sr.transform(Zte2)) / len(splits)
        FINAL_OOF[tt] = oof; FINAL_TE[tt] = te_pred
    print(f"stack rebuilt in {time.time()-t0:.0f}s")

    # ---- load GNN OOF / test, align on dedup / test row index ----
    gnn_oof = pd.read_csv(os.path.join(GNN_DIR, "gnn_oof.csv")).set_index("row_id")
    gnn_test = pd.read_csv(os.path.join(GNN_DIR, "gnn_test.csv")).set_index("row_id")

    # ---- fold-safe per-target weight search ----
    grid = np.linspace(0.0, 1.0, 21)
    blend_oof = {tt: np.full(int((dedup["target_type"] == tt).sum()), np.nan) for tt in TARGETS}
    blend_te = {tt: np.zeros(len(Xte)) for tt in TARGETS}
    best_w = {}
    print("\nPer-target fold-safe blend weight search:")
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        stack_oof = FINAL_OOF[tt]          # length = n rows of target tt
        gnn_vals = gnn_oof["gnn_oof"].reindex(dedup.index[idx]).to_numpy()
        y_tt = Y[idx]
        pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
        # per-fold: tune w on other folds, apply to val fold (honest)
        fold_te = np.zeros(len(Xte))
        w_acc = []
        for tr, va in splits:
            tr_l, va_l = pos[tr], pos[va]
            best_w_here, best_r = 0.0, -np.inf
            for w in grid:
                pred = w * stack_oof[tr_l] + (1 - w) * gnn_vals[tr_l]
                fin = ~np.isnan(pred) & ~np.isnan(y_tt[tr_l])
                if fin.sum() < 5:
                    continue
                r = r2_score(y_tt[tr_l][fin], pred[fin])
                if r > best_r:
                    best_r, best_w_here = r, w
            blend_oof[tt][va_l] = (best_w_here * stack_oof[va_l]
                                   + (1 - best_w_here) * gnn_vals[va_l])
            gnn_te = gnn_test["gnn_test"].reindex(test.index).to_numpy()
            fold_te += (best_w_here * FINAL_TE[tt] + (1 - best_w_here) * gnn_te) / len(splits)
            w_acc.append(best_w_here)
        blend_te[tt] = fold_te
        best_w[tt] = float(np.mean(w_acc)) if w_acc else np.nan
    print("  (per-fold tuned weights averaged per target)")

    # fallback: simple single-weight per target tuned on full OOF (for reporting)
    w_report = {}
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        stack_oof = FINAL_OOF[tt]
        gnn_vals = gnn_oof["gnn_oof"].reindex(dedup.index[idx]).to_numpy()
        y_tt = Y[idx]
        best_w_full, best_r = 0.0, -np.inf
        for w in grid:
            pred = w * stack_oof + (1 - w) * gnn_vals
            fin = ~np.isnan(pred) & ~np.isnan(y_tt)
            if fin.sum() < 5:
                continue
            r = r2_score(y_tt[fin], pred[fin])
            if r > best_r:
                best_r, best_w_full = r, w
        w_report[tt] = best_w_full

    # ---- report table ----
    rows = []
    print(f"\n{'target':<6} {'stackOOF':>10} {'gnnOOF':>10} {'blendOOF':>10} {'w(stack)':>9}")
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        y_tt = Y[idx]
        s_oof = FINAL_OOF[tt]
        g_oof = gnn_oof["gnn_oof"].reindex(dedup.index[idx]).to_numpy()
        b_oof = blend_oof[tt]
        r_s = r2_score(y_tt, s_oof)
        r_g = r2_score(y_tt[~np.isnan(g_oof)], g_oof[~np.isnan(g_oof)])
        r_b = r2_score(y_tt, b_oof)
        rows.append((tt, r_s, r_g, r_b, w_report[tt]))
        print(f"{tt:<6} {r_s:>10.4f} {r_g:>10.4f} {r_b:>10.4f} {w_report[tt]:>9.2f}")

    n_s = np.mean([r[1] for r in rows]); n_g = np.mean([r[2] for r in rows])
    n_b = np.mean([r[3] for r in rows])
    print(f"\nmean: stack {n_s:.4f} | gnn {n_g:.4f} | blend {n_b:.4f}")

    pd.DataFrame(rows, columns=["target", "stack_oof_r2", "gnn_oof_r2",
                                "blend_oof_r2", "w_stack"]).round(4)\
        .to_csv(os.path.join(OUT, "gnn_moe_compare.csv"), index=False)

    # ---- blended test submission (uses fold-averaged blend_te) ----
    sub = pd.DataFrame({"id": test["id"].values, "target": np.nan}, index=test.index)
    for tt in TARGETS:
        sub.loc[test["target_type"] == tt, "target"] = blend_te[tt][test["target_type"] == tt]
    assert sub["target"].notna().all(), "missing blend test preds"
    sub.to_csv(os.path.join(WORK, "gnn_moe_submission.csv"), index=False)
    print(f"\nwrote {OUT}/gnn_moe_compare.csv and {WORK}/gnn_moe_submission.csv "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()