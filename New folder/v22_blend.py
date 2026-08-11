"""Per-target n-arm Ridge blend (P14 fold_safe_blend generalized to k arms)."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]


def blend_narm_oof(M, y, g, alphas=ALPHA_GRID, n_splits=5):
    """Fold-safe n-arm blend per-target.

    Returns (oof, best_alpha, coefs_mean). coefs_mean has length M.shape[1].
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    k = M.shape[1] if M.ndim == 2 else 0
    if n < 2:
        return y.copy(), float(alphas[0]), np.zeros(k)
    M = np.where(np.isfinite(M), M, np.nanmean(M, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        lr = Ridge(alpha=alphas[0]).fit(M, y)
        return lr.predict(M), float(alphas[0]), lr.coef_
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, besta = -np.inf, alphas[0]
    for a in alphas:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, besta = r, a
    oof = np.zeros(n)
    coefs = []
    for tr, vk in cv:
        lr = Ridge(alpha=besta).fit(M[tr], y[tr])
        oof[vk] = lr.predict(M[vk])
        coefs.append(lr.coef_)
    return oof, float(besta), np.mean(coefs, axis=0)


def _p14_2arm_oof(M2, y, g, n_splits=5):
    """Fold-safe 2-arm (gbm, mt) OOF alpha scan — P14 reference protocol."""
    M2 = np.asarray(M2, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    if n < 2:
        return y.copy()
    M = np.where(np.isfinite(M2), M2, np.nanmean(M2, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        return Ridge(alpha=ALPHA_GRID[0]).fit(M, y).predict(M)
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, out = -np.inf, np.zeros(n)
    for a in ALPHA_GRID:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, out = r, o.copy()
    return out
