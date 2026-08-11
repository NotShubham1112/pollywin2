import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from v22_blend import ALPHA_GRID, _p14_2arm_oof, blend_narm_oof


def make_blend_data(n_groups=12, per_group=6, seed=0, n_arms=3):
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    g = np.repeat(np.arange(n_groups), per_group)
    base = rng.normal(size=n)
    cols = [base + rng.normal(scale=0.05, size=n) for _ in range(n_arms - 1)]
    cols.append(rng.normal(scale=1e-6, size=n))
    M = np.column_stack(cols)
    y = base + rng.normal(scale=0.01, size=n)
    return M, y, g


def ridge_blend_ncol(M, y, g, alphas=ALPHA_GRID, n_splits=5):
    M = np.where(np.isfinite(M), M, np.nanmean(M, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, besta = -np.inf, alphas[0]
    for a in alphas:
        o = np.zeros(len(y))
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, besta = r, a
    oof = np.zeros(len(y))
    for tr, vk in cv:
        oof[vk] = Ridge(alpha=besta).fit(M[tr], y[tr]).predict(M[vk])
    return oof, besta


def test_default_grid_and_signature():
    params = inspect.signature(blend_narm_oof).parameters
    assert tuple(params["alphas"].default) == tuple(ALPHA_GRID)
    assert params["n_splits"].default == 5


def test_returns_oof_alpha_coefs():
    M, y, g = make_blend_data()
    oof, alpha, coefs = blend_narm_oof(M, y, g)
    assert oof.shape == (len(y),)
    assert coefs.shape == (3,)
    assert np.isfinite(oof).all() and np.isfinite(coefs).all()
    assert alpha in ALPHA_GRID


def test_generates_4_arms():
    M, y, g = make_blend_data(n_arms=4)
    oof, alpha, coefs = blend_narm_oof(M, y, g)
    assert coefs.shape == (4,)
    r = np.corrcoef(oof, y)[0, 1]
    assert r >= 0.999


def test_matches_reference_fold_safe_math():
    rng = np.random.default_rng(11)
    n = 60
    g = np.repeat(np.arange(10), 6)
    base = rng.normal(size=n)
    M = np.column_stack([base + rng.normal(scale=0.1, size=n),
                         base + rng.normal(scale=0.1, size=n),
                         rng.normal(scale=1e-6, size=n)])
    y = base + rng.normal(scale=0.05, size=n)
    oof, alpha, _ = blend_narm_oof(M, y, g)
    ref_oof, ref_alpha = ridge_blend_ncol(M, y, g)
    assert alpha == ref_alpha
    assert np.allclose(oof, ref_oof, atol=1e-8)


def test_zero_col_preserves_2arm_reference():
    """blend_narm_oof on [a, b, 0] must match _p14_2arm_oof on [a, b]."""
    rng = np.random.default_rng(21)
    n = 60
    g = np.repeat(np.arange(10), 6)
    base = rng.normal(size=n)
    M2 = np.column_stack([base + rng.normal(scale=0.05, size=n),
                          base + rng.normal(scale=0.05, size=n)])
    y = base + rng.normal(scale=0.01, size=n)
    oof3, _, _ = blend_narm_oof(np.column_stack([M2, np.zeros(n)]), y, g)
    ref = _p14_2arm_oof(M2, y, g)
    assert np.allclose(oof3, ref, atol=1e-6)


def test_nonfinite_handling():
    M, y, g = make_blend_data(seed=3)
    M2 = np.where(M == M, M, np.nan)
    M2[0, 0] = np.nan
    oof, alpha, coefs = blend_narm_oof(M2, y, g)
    assert np.isfinite(oof).all() and np.isfinite(coefs).all()


def test_small_group_and_single_row_fallback():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(3, 3))
    y = rng.normal(size=3)
    g = np.ones(3, dtype=int)
    oof, alpha, coefs = blend_narm_oof(M, y, g, n_splits=5)
    assert alpha == ALPHA_GRID[0]
    assert np.allclose(oof, Ridge(alpha=ALPHA_GRID[0]).fit(M, y).predict(M), atol=1e-8)
    M1, y1 = np.array([[1.0, 2.0, 3.0]]), np.array([2.5])
    oof1, alpha1, coefs1 = blend_narm_oof(M1, y1, np.array([0]), n_splits=5)
    assert np.array_equal(oof1, y1)
    assert np.array_equal(coefs1, np.zeros(3))
    assert alpha1 == ALPHA_GRID[0]


def test_deterministic():
    M, y, g = make_blend_data(seed=7)
    o1, a1, c1 = blend_narm_oof(M, y, g)
    o2, a2, c2 = blend_narm_oof(M, y, g)
    assert np.array_equal(o1, o2) and a1 == a2 and np.array_equal(c1, c2)
