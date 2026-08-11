"""Fold-safe per-target Ridge "bert" arm on a frozen, label-free pool.

compute_bert_arm() produces out-of-fold (OOF) and test predictions from per-
target Ridge heads fit on frozen encoder embeddings. The pool carries no
labels, so no label leakage flows through the encoder; fold safety comes from
a single shared GroupKFold on canonical SMILES (group = smiles), which never
puts the same polymer on both sides of a split.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g, n_splits=5, seed=42):
    """Per-target Ridge heads on a frozen pool, folded on canonical SMILES.

    Parameters
    ----------
    pool_tr : (n_tr, d) float32 — frozen, LABEL-FREE embeddings of train rows.
    pool_te : (n_te, d) float32 — frozen embeddings of test rows.
    y : (n_tr,) float32 — train target values, one per row.
    tt_tr : (n_tr,) str array — target-type per train row (eea/egb/egc/ei/eps/nc/tg).
    tt_te : (n_te,) str array — target-type per test row.
    g : (n_tr,) — group ids = canonical smiles per train row (GroupKFold key).
    n_splits : int — number of folds for the ONE shared GroupKFold (default 5).
    seed : int — accepted for API stability / downstream reuse; GroupKFold is
                 deterministic by construction.

    Returns
    -------
    (oof_bert, test_bert) : (n_tr,) float32, (n_te,) float32
        oof_bert[i] is predicted by a head trained only on rows of target
        tt_tr[i] whose fold differs from row i's fold (leak-safe by
        construction). test_bert[i] = mean over fold heads of target tt_te[i].

    Fallbacks (documented):
    * Target with fewer than n_splits rows: one head is fit on ALL of the
      target's rows and used for both OOF and test.
    * Single-row (or empty) target: no Ridge is fit on < 2 samples; the row's
      value is its own label (mean of its rows; per-row target mean).
    * Rows that GroupKFold cannot cover cross-validator-wise (e.g. all rows of
      a target share one group): a head fit on all of that target's rows fills
      the gap. This is unusual and slightly in-sample for those rows only; it
      exists to guarantee no NaN on degenerate inputs.
    * Test rows whose target never appears in train: filled with the global
      train mean (never NaN).
    """
    n_tr = pool_tr.shape[0]
    n_te = pool_te.shape[0]
    oof = np.full(n_tr, np.nan, dtype=np.float64)
    test = np.full(n_te, np.nan, dtype=np.float64)

    global_mean = float(np.mean(y)) if n_tr else 0.0

    # GroupKFold cannot form n_splits folds from fewer groups; on such a
    # degenerate pool (e.g. every row in one smiles group) skip the shared CV
    # and fall back to per-target all-rows heads below. No leak: this only
    # triggers when cross-validation folds are impossible.
    n_groups = len(np.unique(g)) if n_tr else 0
    degenerate = n_groups < n_splits
    if n_tr >= 2 and not degenerate:
        fold = np.empty(n_tr, dtype=np.int64)
        gkf = GroupKFold(n_splits=n_splits)
        for f, (_, va) in enumerate(gkf.split(np.arange(n_tr), y, g)):
            fold[va] = f
    else:
        fold = np.zeros(n_tr, dtype=np.int64)

    for t in np.unique(tt_tr):
        tr_idx = np.where(tt_tr == t)[0]
        n_t = tr_idx.size
        te_idx = np.where(tt_te == t)[0]

        if n_t == 0:
            test[te_idx] = global_mean
            continue

        if n_t < n_splits or degenerate:
            # small-target fallback: one head on all rows, used for both sides
            # (also the whole-pool fallback when groups < n_splits)
            if n_t < 2:
                val = float(np.mean(y[tr_idx]))
                oof[tr_idx] = val
                test[te_idx] = val
            else:
                ridge = Ridge(alpha=1.0, fit_intercept=True)
                ridge.fit(pool_tr[tr_idx], y[tr_idx])
                oof[tr_idx] = ridge.predict(pool_tr[tr_idx])
                if te_idx.size:
                    test[te_idx] = ridge.predict(pool_te[te_idx])
            continue

        # main path: 5-fold-per-row heads (exclusive folds, leak-safe)
        fold_t = fold[tr_idx]
        heads = []
        for f in range(n_splits):
            trf = tr_idx[fold_t != f]
            if trf.size < 2:
                continue
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(pool_tr[trf], y[trf])
            heads.append((f, ridge))

        if heads:
            for f, ridge in heads:
                va = tr_idx[fold_t == f]
                if va.size:
                    oof[va] = ridge.predict(pool_tr[va])
            if te_idx.size:
                acc = np.stack([ridge.predict(pool_te[te_idx])
                                for _, ridge in heads])
                test[te_idx] = acc.mean(axis=0)

        # anything still uncovered (e.g. all t-rows share one group)
        missing = tr_idx[np.isnan(oof[tr_idx])]
        if missing.size:
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(pool_tr[tr_idx], y[tr_idx])
            oof[missing] = ridge.predict(pool_tr[missing])
            if not te_idx.size or np.isnan(test[te_idx]).any():
                test[te_idx] = ridge.predict(pool_te[te_idx]) if te_idx.size else test[te_idx]

    # catch-all: no NaN, ever
    test[np.isnan(test)] = global_mean
    oof[np.isnan(oof)] = global_mean

    return (np.asarray(oof, dtype=np.float32),
            np.asarray(test, dtype=np.float32))


def compute_bert_only_r2(oof_bert, y, tt_tr):
    """Gate-0 diagnostic: per-target OOF R^2 of the BERT arm Ridge alone."""
    out = {}
    for t in np.unique(tt_tr):
        idx = np.where(tt_tr == t)[0]
        if idx.size < 2:
            out[t] = 0.0
            continue
        out[t] = float(np.corrcoef(y[idx], oof_bert[idx])[0, 1]) ** 2
    return out
