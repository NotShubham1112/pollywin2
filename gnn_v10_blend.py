"""v11 diagnostics + blend: v6/v8 stack (rebuilt on cached GBM OOFs) + v10 PRETRAINED GNN.

Answers the one question: does the pretrained GNN add orthogonal signal to the stack?

Diagnostics (per target):
  1. OOF prediction correlation  corr(stack_oof, gnn_oof)
  2. ERROR correlation           corr(y - stack, y - gnn)   <- the ensemble signal
  3. Blend sweep on OOF          argmax_w R2(w*stack + (1-w)*gnn)

Then a fold-safe per-target weight search -> blended OOF / blended test submission.

This mirrors gnn_moe_blend.py but points at the v10 PRETRAINED GNN cache and reports
the full correlation/blend tables.
"""

import os, time, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error as rmse_metric

WORK = r"vault\pipeline_out"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-dir", default=r"vault\kernel-v10-output")
    ap.add_argument("--out", default=r"vault\pipeline_out\gnn_arm")
    ap.add_argument("--tag", default="v10")
    args = ap.parse_args()

    t0 = __import__("time").time()
    Xtr = pd.read_pickle(os.path.join(WORK, "Xtr.pkl"))
    Xte = pd.read_pickle(os.path.join(WORK, "Xte.pkl"))
    dedup = pd.read_pickle(os.path.join(WORK, "dedup.pkl"))
    test = pd.read_pickle(os.path.join(WORK, "test.pkl"))
    folds = dedup["fold"].to_numpy()
    Y = dedup["target"].values
    print(f"Xtr {Xtr.shape} Xte {Xte.shape} | folds {folds.max() + 1}", flush=True)

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

    # ---- load cached full-run GBM base OOFs (4 models x 7 targets) ----
    CHK = os.path.join(WORK, "gnn_arm", "moe_gbm_chk.parquet")
    chk = pd.read_parquet(CHK)
    oof_store = {(r["key"], r["target"]): r["oof"] for _, r in chk.iterrows()}
    test_store = {(r["key"], r["target"]): r["test_pred"] for _, r in chk.iterrows()}
    print(f"loaded {len(chk)} GBM checkpoints", flush=True)

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

    def cross_oof_features(tt):
        m_tt = (dedup["target_type"] == tt).values
        feats, cols = [], []
        for ct in CROSS_MAP[tt]:
            m_ct = (dedup["target_type"] == ct).values
            c2o = dict(zip(dedup.loc[m_ct, "canon"].values, L15_OOF[ct]))
            vals = np.array([c2o.get(c, np.nan) for c in dedup.loc[m_tt, "canon"].values],
                            dtype=np.float32)
            miss = np.isnan(vals).astype(np.float32)
            vals = np.nan_to_num(vals, nan=float(np.nanmean(L15_OOF[ct])))
            feats += [vals, miss]; cols += [f"cross_{ct}", f"cross_{ct}_miss"]
        return (np.column_stack(feats), cols) if feats else (None, None)

    def cross_te_features(tt):
        feats = []
        for ct in CROSS_MAP[tt]:
            feats.append(np.asarray(L15_TE[ct], dtype=np.float32))
            feats.append(np.zeros(len(test), dtype=np.float32))
        return np.column_stack(feats) if feats else None

    FINAL_OOF, FINAL_TE = {}, {}
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        cols = [b + "_" + tt for b in ("lgb", "cat", "xgb", "hgb")]
        Z1 = np.column_stack([oof_store[(c, tt)] for c in cols])
        Zrel = reliability(Z1)
        Zcr, _ = cross_oof_features(tt)
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
    print(f"stack rebuilt (v10 cols, GBM-only experts)", flush=True)

    # ---- load GNN (v10 pretrained) cache, aligned on dedup/test index ----
    gnn_oof = pd.read_csv(os.path.join(args.gnn_dir, "gnn_oof.csv")).set_index("row_id")
    gnn_test = pd.read_csv(os.path.join(args.gnn_dir, "gnn_test.csv")).set_index("row_id")
    print(f"GNN cache {args.gnn_dir}: oof {len(gnn_oof)} test {len(gnn_test)}", flush=True)

    # =====================================================================
    # DIAGNOSTICS: OOF corr, ERROR corr (the ensemble signal), full-oof blend
    # =====================================================================
    grid = np.linspace(0.0, 1.0, 101)
    rows = []
    print(f"\n{'target':<6} {'stackOOF':>9} {'gnnOOF':>9} {'predCorr':>9} "
          f"{'errCorr':>9} {'bestBlend':>9} {'w(stack)':>9}")
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        y_tt = Y[idx]
        s = FINAL_OOF[tt]
        g = gnn_oof["gnn_oof"].reindex(dedup.index[idx]).to_numpy().copy()
        fin = ~np.isnan(g)
        g[~fin] = np.nan
        r_s = r2_score(y_tt, s)
        r_g = r2_score(y_tt[fin], g[fin])
        pc = np.corrcoef(s, g)[0, 1]            # prediction correlation
        es = y_tt - s; eg = y_tt - g
        mfin = fin & ~np.isnan(y_tt)
        ec = np.corrcoef(es[mfin], eg[mfin])[0, 1]  # error correlation
        best_r, best_w = -np.inf, np.nan
        for w in grid:
            pred = w * s + (1 - w) * g
            f = ~np.isnan(pred) & ~np.isnan(y_tt)
            r = r2_score(y_tt[f], pred[f])
            if r > best_r:
                best_r, best_w = r, w
        rows.append((tt, r_s, r_g, pc, ec, best_r, best_w))
        print(f"{tt:<6} {r_s:>9.4f} {r_g:>9.4f} {pc:>9.3f} {ec:>9.3f} "
              f"{best_r:>9.4f} {best_w:>8.2f}")

    n_s = np.mean([r[1] for r in rows]); n_g = np.mean([r[2] for r in rows])
    n_b = np.mean([r[5] for r in rows])
    print(f"\nmean: stack {n_s:.4f} | gnn {n_g:.4f} | best-blend {n_b:.4f}")

    diag = pd.DataFrame(rows, columns=["target", "stack_oof_r2", "gnn_oof_r2",
                                       "pred_corr", "err_corr", "best_blend_r2",
                                       "w_stack"]).round(4)
    diag.to_csv(os.path.join(args.out, f"v{args.tag}_blend_diag.csv"), index=False)
    print(f"wrote {args.out}/v{args.tag}_blend_diag.csv")

    # =====================================================================
    # FOLD-SAFE blend weight search -> honest blend OOF + test submission
    # =====================================================================
    grid21 = np.linspace(0.0, 1.0, 21)
    blend_oof = {tt: np.full(int((dedup["target_type"] == tt).sum()), np.nan) for tt in TARGETS}
    blend_te = {tt: np.zeros(len(Xte)) for tt in TARGETS}
    best_w = {}
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        stack_oof = FINAL_OOF[tt]
        gnn_vals = gnn_oof["gnn_oof"].reindex(dedup.index[idx]).to_numpy()
        y_tt = Y[idx]
        pos = np.full(len(dedup), -1, dtype=int); pos[idx] = np.arange(len(idx))
        fold_te = np.zeros(len(Xte)); w_acc = []
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

    print(f"\n{'target':<6} {'stackOOF':>9} {'gnnOOF':>9} {'blendOOF':>9} {'w(stack)':>9}")
    bro = []
    for tt in TARGETS:
        m, idx, splits = get_splits(tt)
        y_tt = Y[idx]
        s = FINAL_OOF[tt]
        g = gnn_oof["gnn_oof"].reindex(dedup.index[idx]).to_numpy()
        b = blend_oof[tt]
        r_s = r2_score(y_tt, s); r_g = r2_score(y_tt, g); r_b = r2_score(y_tt, b)
        bro.append((tt, r_s, r_g, r_b, best_w[tt]))
        print(f"{tt:<6} {r_s:>9.4f} {r_g:>9.4f} {r_b:>9.4f} {best_w[tt]:>9.2f}")
    n_s2 = np.mean([r[1] for r in bro]); n_b2 = np.mean([r[3] for r in bro])
    print(f"\nmean: stack {n_s2:.4f} | blend {n_b2:.4f} | gain {n_b2 - n_s2:+.4f}")

    pd.DataFrame(bro, columns=["target", "stack_oof", "gnn_oof",
                               "blend_oof", "w_stack"]).round(4)\
        .to_csv(os.path.join(args.out, f"v{args.tag}_blend_compare.csv"), index=False)

    sub = pd.DataFrame({"id": test["id"].values, "target": np.nan}, index=test.index)
    for tt in TARGETS:
        sub.loc[test["target_type"] == tt, "target"] = blend_te[tt][test["target_type"] == tt]
    assert sub["target"].notna().all()
    sub.to_csv(os.path.join(WORK, f"gnn_moe_v{args.tag}_submission.csv"), index=False)
    print(f"\nwrote {WORK}/gnn_moe_v{args.tag}_submission.csv ({len(sub)} rows)")


if __name__ == "__main__":
    main()