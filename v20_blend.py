"""Per-target 3-arm Ridge blend (P14 fold_safe_blend with a 3rd arm).

blend_3d is the P14 production blend (vault/final_synthesis.py:84-101,
fold_safe_blend) run on a 3-column arm matrix: same GroupKFold on smiles
groups, same alpha grid, same inner alpha selection by OOF r2_score, then
refit at the best alpha. It is called ONCE PER TARGET: rows are already
filtered to that target before this function is invoked.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ALPHAS = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]


def blend_3d(M_tr, y, g, alphas=ALPHAS, n_splits=5):
    """Blend 3 arm OOF predictions (cols = gbm, mt, trf) per-target.

    Returns
    -------
    oof : np.ndarray (n,)
        Blended out-of-fold prediction per row, in input order.
    coefs_mean : np.ndarray (3,)
        Mean of the refit fold coefficients (one per arm).
    """
    M_tr = np.asarray(M_tr, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)

    # Never crash on degenerate input: a single row (or empty) yields the row's
    # own label as OOF and zero coefficients.
    if n < 2:
        return y.copy(), np.zeros(3)

    # Non-finite arms: replace NaN with column mean, then any residual
    # non-finite (all-NaN column) with 0.0. Verbatim from P14 fold_safe_blend.
    M = np.where(np.isfinite(M_tr), M_tr, np.nanmean(M_tr, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)

    # Too few smiles groups to form n_splits folds: GroupKFold would crash.
    # Fall back to a single Ridge at the first alpha on all rows.
    if len(np.unique(g)) < n_splits:
        lr = Ridge(alpha=alphas[0]).fit(M, y)
        return lr.predict(M), lr.coef_

    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))

    # Inner alpha selection: OOF r2_score over the full per-fold prediction.
    best, besta = -np.inf, alphas[0]
    for a in alphas:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, besta = r, a

    # Refit at the best alpha; collect fold coefficients.
    oof = np.zeros(n)
    coefs = []
    for tr, vk in cv:
        lr = Ridge(alpha=besta).fit(M[tr], y[tr])
        oof[vk] = lr.predict(M[vk])
        coefs.append(lr.coef_)
    return oof, np.mean(coefs, axis=0)