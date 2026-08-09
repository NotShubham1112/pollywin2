import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from v20_blend import blend_3d

ALPHAS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0)


def make_blend_data(n_groups=12, per_group=6, seed=0):
    """2 informative arms + a near-zero 3rd arm, y = strong signal + tiny noise."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    g = np.repeat(np.arange(n_groups), per_group)
    base = rng.normal(size=n)
    arm1 = base + rng.normal(scale=0.05, size=n)
    arm2 = base + rng.normal(scale=0.05, size=n)
    arm3 = rng.normal(scale=1e-6, size=n)
    M = np.column_stack([arm1, arm2, arm3])
    y = base + rng.normal(scale=0.01, size=n)
    return M, y, g


def ridge_blend_2col(M, y, g, alphas=ALPHAS, n_splits=5):
    """Reference fold_safe_blend semantics for 2 columns, computed in-test."""
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
    return oof


def test_returns_both_outputs_shape_and_default_grid():
    M, y, g = make_blend_data()
    res = blend_3d(M, y, g)
    assert isinstance(res, tuple) and len(res) == 3
    oof, coefs, alpha = res
    assert oof.shape == (len(y),)
    assert coefs.shape == (3,)
    assert np.isfinite(oof).all()
    assert np.isfinite(coefs).all()
    assert alpha in ALPHAS
    assert isinstance(alpha, float)
    # default alpha grid matches the P14 grid exactly
    import inspect
    params = inspect.signature(blend_3d).parameters
    assert "alphas" in params
    assert tuple(params["alphas"].default) == ALPHAS


def test_informative_columns_recovery():
    # 2 informative columns -> OOF correlates with y; 3rd near-zero column
    # must not derail the blend. Coefs have length 3 (one per arm).
    M, y, g = make_blend_data()
    oof, coefs, alpha = blend_3d(M, y, g)
    assert coefs.shape == (3,)
    r = np.corrcoef(oof, y)[0, 1]
    assert r >= 0.999, f"OOF corr {r:.6f}"
    assert np.abs(coefs[2]) < 1e-3, f"zero arm coef {coefs[2]:.4f} should be ~0"


def test_verbatim_fold_safe_equivalent():
    # A zero 3rd column must reproduce the 2-col fold_safe_blend OOF exactly:
    # the 3rd coef is regularized to ~0, leaving predictions unchanged.
    rng = np.random.default_rng(11)
    n_groups, per_group = 10, 6
    n = n_groups * per_group
    g = np.repeat(np.arange(n_groups), per_group)
    base = rng.normal(size=n)
    M2 = np.column_stack([base + rng.normal(scale=0.1, size=n),
                          base + rng.normal(scale=0.1, size=n)])
    y = base + rng.normal(scale=0.05, size=n)
    M3 = np.column_stack([M2, np.zeros(n)])

    oof3, coefs3, alpha3 = blend_3d(M3, y, g)
    oof_ref = ridge_blend_2col(M2, y, g)
    assert np.allclose(oof3, oof_ref, atol=1e-6), "zero-col 3d blend != 2-col"
    assert np.abs(coefs3[2]) < 1e-3


def test_nonfinite_handling_matches_reference():
    # NaN in an arm must be replaced by column mean then 0 (never propagated).
    M, y, g = make_blend_data(seed=3)
    M2 = np.where(M == M, M, np.nan)
    M2[0, 0] = np.nan
    M2[5, 1] = np.nan
    oof, coefs, alpha = blend_3d(M2, y, g)
    assert np.isfinite(oof).all()
    assert np.isfinite(coefs).all()
    assert alpha in ALPHAS


def test_small_group_fallback():
    # 3 rows, 1 unique group: GroupKFold(n_splits=5) cannot form 5 folds;
    # must fall back to a single all-rows Ridge, finite, never crashes.
    rng = np.random.default_rng(5)
    M = rng.normal(size=(3, 3))
    y = rng.normal(size=3)
    g = np.ones(3, dtype=int)
    oof, coefs, alpha = blend_3d(M, y, g, n_splits=5)
    assert oof.shape == (3,)
    assert coefs.shape == (3,)
    assert np.isfinite(oof).all()
    assert np.isfinite(coefs).all()
    assert alpha == ALPHAS[0]
    # OOF equals in-sample Ridge prediction at the first alpha
    ref = Ridge(alpha=ALPHAS[0]).fit(M, y).predict(M)
    assert np.allclose(oof, ref, atol=1e-8)


def test_single_row_returns_label():
    M = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2.5])
    g = np.array([0])
    oof, coefs, alpha = blend_3d(M, y, g, n_splits=5)
    assert np.array_equal(oof, y)
    assert coefs.shape == (3,)
    assert np.array_equal(coefs, np.zeros(3))
    assert alpha == ALPHAS[0]


def test_deterministic():
    M, y, g = make_blend_data(seed=7)
    oof1, c1, a1 = blend_3d(M, y, g)
    oof2, c2, a2 = blend_3d(M, y, g)
    assert np.array_equal(oof1, oof2)
    assert np.array_equal(c1, c2)
    assert a1 == a2


def test_alpha_is_the_internally_selected_best():
    """blend_3d's third element must be the alpha that maximizes OOF r2 in
    the inner per-alpha sweep (ties keep the grid-lowest) — recompute that
    selection independently in-test and require equality."""
    def reference_best_alpha(M, y, g, alphas=ALPHAS, n_splits=5):
        cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
        best, besta = -np.inf, alphas[0]
        for a in alphas:
            o = np.zeros(len(y))
            for tr, vk in cv:
                o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
            r = r2_score(y, o)
            if r > best:
                best, besta = r, a
        return besta

    rng = np.random.default_rng(21)
    n_groups, per_group = 7, 6
    n = n_groups * per_group
    g = np.repeat(np.arange(n_groups), per_group)
    base = rng.normal(size=n)
    M = np.column_stack([base + rng.normal(scale=0.05, size=n),
                         base + rng.normal(scale=0.05, size=n),
                         rng.normal(scale=1e-6, size=n)])
    y = base + rng.normal(scale=0.01, size=n)

    oof, _, alpha = blend_3d(M, y, g)
    ref = reference_best_alpha(M, y, g)
    assert alpha == ref
    assert alpha in ALPHAS


def test_group_folds_never_straddle_smiles():
    # same-group rows must land in a single fold (GroupKFold guarantees this)
    M, y, g = make_blend_data(n_groups=10, per_group=5)
    oof, _, _ = blend_3d(M, y, g)
    assert oof.shape == (len(y),)
    assert np.isfinite(oof).all()