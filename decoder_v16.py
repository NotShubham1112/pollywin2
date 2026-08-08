"""v16 Cross-Target Decoder — canonical, unit-testable source of truth.

The v16 Kaggle notebook embeds the decoder as one verbatim cell (the same
source-slice pattern as mt_gnn_v2.py -> CORE_A/CORE_B). Keeping it here as a
Python module lets the pure logic be unit-tested (tests/test_decoder_v16.py)
and lets vault/compare_v16.py reuse the physics math offline. Only train
labels are ever read (no test leakage). All folds use GroupKFold(n_splits=
GLOBAL_FOLDS) on `canon`, identical to P14.
"""
import numpy as np
import pandas as pd

GLOBAL_FOLDS = 5
SEED = 42

TARGETS_DEC = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
TARGET_IDX_DEC = {t: i for i, t in enumerate(TARGETS_DEC)}

# Physics recipes: target -> (kind, srcs)
#   "subtract": out = src0 - src1         (egc = ei - eea)
#   "linear":   out = a * (feature(src)) + b, fitted from train pairs
#               egb = a * egc + b    | eps = a * nc^2 + b
PHYS_RECIPE = {
    "egc": ("subtract", ("ei", "eea")),
    "egb": ("linear", ("egc",)),
    "eps": ("linear", ("nc2",)),
}


def build_pivot_df(canon_arr, tgt_arr, val_arr):
    """Pivot table: index=canon, columns=TARGETS_DEC, values=target (NaN if absent)."""
    df = pd.DataFrame({"canon": canon_arr, "target_type": tgt_arr, "value": val_arr})
    piv = df.dropna(subset=["value"]).pivot_table(
        index="canon", columns="target_type", values="value", aggfunc="first")
    return piv.reindex(columns=TARGETS_DEC)


def sibling_feature(canon_list, pivot):
    """(n,7) float64 — for every row, the canon's 7 train-mediated sibling
    values in TARGETS_DEC column order (NaN where a target is absent)."""
    out = np.full((len(canon_list), 7), np.nan, dtype=np.float64)
    for i, c in enumerate(canon_list):
        if c in pivot.index:
            out[i] = pivot.loc[c].values
    return out


def _fit_linear(x, y):
    """Least-squares slope/intercept fit. Needs >= 3 points, else identity."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return 1.0, 0.0
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def _aug_feature(sib):
    """Append derived feature columns to a (n,7) sibling matrix.
    Returns (aug, name->col) with 'nc2' = nc**2 appended at index 7."""
    nc2 = np.full(len(sib), np.nan, dtype=np.float64)
    ok = np.isfinite(sib[:, TARGET_IDX_DEC["nc"]])
    nc2[ok] = sib[ok, TARGET_IDX_DEC["nc"]] ** 2
    aug = np.column_stack([sib, nc2])
    names = {t: i for i, t in enumerate(TARGETS_DEC)}
    names["nc2"] = 7
    return aug, names


def physics_arm(sib_tr, sib_te, tr_tgt=None, group=None, global_folds=GLOBAL_FOLDS):
    """Return (phys_tr, phys_te) as (n_tr,7) and (n_te,7) float64 arrays in
    TARGETS_DEC column order. Missing/NaN stays NaN (caller falls back).

    Fold-safe: when `group` (array of per-row canon/labels) is given, the
    linear-recipe pair fits exclude the held-out fold's own pairs for the OOF
    arms (GroupKFold(global_folds) on group), and test uses a fit on all train
    pairs (train labels only). When group is None, a single all-train fit is
    used for both (unit-test convenience; still train-only).
    """
    n_tr, n_te = len(sib_tr), len(sib_te)
    out_tr = np.full((n_tr, 7), np.nan, dtype=np.float64)
    out_te = np.full((n_te, 7), np.nan, dtype=np.float64)

    aug_tr, names_tr = _aug_feature(sib_tr)
    aug_te, names_te = _aug_feature(sib_te)

    # per-fold membership of training rows (None when not fold-safe)
    if group is not None and global_folds > 1:
        from sklearn.model_selection import GroupKFold
        _cv = GroupKFold(n_splits=min(global_folds, len(np.unique(group))))
        fold_id = np.empty(n_tr, dtype=int)
        for _g, (_, vk) in enumerate(_cv.split(np.zeros(n_tr), np.zeros(n_tr), group)):
            fold_id[vk] = _g
    else:
        fold_id = np.zeros(n_tr, dtype=int)

    for tcol, (kind, srcs) in PHYS_RECIPE.items():
        ti = TARGET_IDX_DEC[tcol]
        if kind == "subtract":
            s0, s1 = (TARGET_IDX_DEC[s] for s in srcs)
            tr_ok = np.isfinite(sib_tr[:, s0]) & np.isfinite(sib_tr[:, s1])
            te_ok = np.isfinite(sib_te[:, s0]) & np.isfinite(sib_te[:, s1])
            out_tr[tr_ok, ti] = sib_tr[tr_ok, s0] - sib_tr[tr_ok, s1]
            out_te[te_ok, ti] = sib_te[te_ok, s0] - sib_te[te_ok, s1]
        else:  # linear
            fcol = names_tr[srcs[0]]
            # pairs where the source feature AND the destination value are known
            pair_ok_tr = np.isfinite(aug_tr[:, fcol]) & np.isfinite(sib_tr[:, ti])
            dst = sib_tr[pair_ok_tr, ti]
            srcv = aug_tr[pair_ok_tr, fcol]

            def _apply(a, b, aug):
                ok = np.isfinite(aug[:, fcol])
                vals = np.full(len(aug), np.nan, dtype=np.float64)
                vals[ok] = a * aug[ok, fcol] + b
                return vals

            if fold_id.max() == 0 or group is None:
                a, b = _fit_linear(srcv, dst)
                out_tr[:, ti] = _apply(a, b, aug_tr)
                out_te[:, ti] = _apply(a, b, aug_te)
            else:
                # fold-safe: per-fold coefficients on other-fold pairs
                a_all, b_all = _fit_linear(srcv, dst)
                out_te[:, ti] = _apply(a_all, b_all, aug_te)
                for k in range(fold_id.max() + 1):
                    keep = (fold_id != k) & pair_ok_tr
                    a_k, b_k = _fit_linear(aug_tr[:, fcol][keep],
                                           sib_tr[:, ti][keep])
                    m = fold_id == k
                    vals = np.full(n_tr, np.nan, dtype=np.float64)
                    vals[m] = a_k * aug_tr[m, fcol] + b_k
                    out_tr[:, ti] = np.where(m, vals, out_tr[:, ti])
    return out_tr, out_te


def learned_arm(canon_tr, tgt_tr, Y_tr, pivot, canon_te, global_folds=GLOBAL_FOLDS,
                seed=SEED, alpha=10.0, min_sibs=2):
    """Fold-safe learned cross-target arm. Returns (lo_tr, lo_te) as (n_tr,7)
    and (n_te,7) float64 arrays in TARGETS_DEC column order.

    For each target t: per-row features = the canon's sibling values for the
    other 6 targets (from the train-only pivot), standardized per fold. A
    per-target Ridge(alpha) is fit on the target-t rows of GroupKFold training
    folds and validated on the held-out canon-fold rows (a canon never crosses
    folds, so a held-out polymer's own target labels never enter its Ridge).
    Test inference averages the per-fold models on the full-train pivot.
    Rows whose canon has < `min_sibs` known siblings stay NaN (caller falls
    back to the target mean).
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n_tr, n_te = len(canon_tr), len(canon_te)
    sib_tr = sibling_feature(canon_tr, pivot)
    sib_te = sibling_feature(canon_te, pivot)
    lo_tr = np.full((n_tr, 7), np.nan, dtype=np.float64)
    lo_te = np.full((n_te, 7), np.nan, dtype=np.float64)
    group_tr = np.asarray(canon_tr)

    for t in TARGETS_DEC:
        ti = TARGET_IDX_DEC[t]
        idx_t = np.where(tgt_tr == t)[0]
        if len(idx_t) < 1:
            continue
        keep_cols = [j for j in range(7) if j != ti]
        Ftr = sib_tr[:, keep_cols]
        Fte = sib_te[:, keep_cols]
        ok_tr = np.isfinite(Ftr).sum(axis=1) >= min_sibs
        ok_te = np.isfinite(Fte).sum(axis=1) >= min_sibs
        n_groups = len(np.unique(group_tr[idx_t]))
        if n_groups < 2:
            continue
        n_splits = max(1, min(global_folds, n_groups))
        cv = GroupKFold(n_splits=n_splits)
        te_acc = np.zeros(n_te)
        te_cnt = np.zeros(n_te)
        idx_t_arr = np.asarray(idx_t)
        for trk, vk in cv.split(idx_t_arr, Y_tr[idx_t], group_tr[idx_t]):
            # trk/vk are positional in idx_t; map to global rows
            g_trk = idx_t[trk]
            g_vk = idx_t[vk]
            fit_ok = g_trk[ok_tr[g_trk]]
            if len(fit_ok) < 1:
                continue
            sc = StandardScaler().fit(Ftr[fit_ok])
            m = Ridge(alpha=alpha).fit(sc.transform(Ftr[fit_ok]), Y_tr[fit_ok])
            vok = g_vk[ok_tr[g_vk]]
            if len(vok) > 0:
                lo_tr[vok, ti] = m.predict(sc.transform(Ftr[vok]))
            if ok_te.any():
                te_acc[ok_te] += m.predict(sc.transform(Fte[ok_te]))
                te_cnt[ok_te] += 1
        lo_te[:, ti] = np.where(te_cnt > 0, te_acc / np.maximum(te_cnt, 1), np.nan)
    return lo_tr, lo_te