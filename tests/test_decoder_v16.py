"""Unit tests for decoder_v16.py — the canonical v16 Cross-Target Decoder.

Covers the physics-imputed arm (pivot builder, sibling feature matrix,
least-squares pair fit, physics_arm OOF/test layout) and (Task 2) the
fold-safe learned arm. Run with `python tests/test_decoder_v16.py` or pytest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoder_v16 import (
    TARGETS_DEC, TARGET_IDX_DEC, PHYS_RECIPE,
    build_pivot_df, sibling_feature, _fit_linear, physics_arm,
)


def _flat():
    tr_c = np.array(["A", "A", "B", "C", "C", "D"])
    tr_t = np.array(["egc", "ei", "egb", "eps", "nc", "nc"])
    tr_v = np.array([2.0, 5.0, 1.5, 3.0, 4.0, 6.0])
    return tr_c, tr_t, tr_v


def test_pivot_shape_and_values():
    tr_c, tr_t, tr_v = _flat()
    piv = build_pivot_df(tr_c, tr_t, tr_v)
    assert list(piv.index) == ["A", "B", "C", "D"]
    assert list(piv.columns) == TARGETS_DEC
    assert piv.loc["A", "egc"] == 2.0 and piv.loc["A", "ei"] == 5.0
    assert np.isnan(piv.loc["C", "egc"])  # polymer C has no egc label


def test_sibling_feature_aligns_to_columns():
    tr_c, tr_t, tr_v = _flat()
    piv = build_pivot_df(tr_c, tr_t, tr_v)
    sib = sibling_feature(["A", "C", "ZZ"], piv)
    assert sib.shape == (3, 7)
    assert sib[0, TARGET_IDX_DEC["egc"]] == 2.0
    assert sib[0, TARGET_IDX_DEC["ei"]] == 5.0
    assert np.isnan(sib[2]).all()       # unknown canon is all-NaN


def test_physics_subtract_and_linear():
    tr_c, tr_t, tr_v = _flat()
    piv = build_pivot_df(tr_c, tr_t, tr_v)
    # A: <2,5,?,?,?,?,?> -> egc_phys(A) = ei-eea(na) => NaN (no eea)
    # C: has eps=3, nc=4 -> eps_phys = f(nc^2) from train pairs
    tr_sib = sibling_feature(tr_c, piv)
    te_sib = sibling_feature(["A", "C"], piv)
    p_tr, p_te = physics_arm(tr_sib, te_sib, tr_t)
    assert np.isnan(p_te[0, TARGET_IDX_DEC["egc"]])   # A has no eea
    # C: eps fits from (eps,nc^2) pairs -> any real number, not NaN
    assert np.isfinite(p_te[1, TARGET_IDX_DEC["eps"]])
    assert np.isnan(p_te[1, TARGET_IDX_DEC["ei"]])    # C has ei=NaN
    assert p_tr.shape == (len(tr_c), 7) and p_te.shape == (2, 7)


def test_physics_subtract_exact():
    # egc = ei - eea exactly
    c = np.array(["P", "P", "P"])
    t = np.array(["ei", "eea", "egc"])
    v = np.array([7.0, 2.0, 999.0])   # egc label is a decoy; physics wins
    piv = build_pivot_df(c, t, v)
    sib = sibling_feature(["P"], piv)
    p_tr, p_te = physics_arm(sib, sib, t)
    assert p_te[0, TARGET_IDX_DEC["egc"]] == 5.0


def test_fit_linear_identity():
    x = np.array([1.0, 2.0, 3.0]); y = np.array([2.0, 4.0, 6.0])
    a, b = _fit_linear(x, y)
    assert np.allclose(a, 2.0) and np.allclose(b, 0.0)


def test_fit_linear_insufficient_points_falls_back():
    x = np.array([1.0, 2.0]); y = np.array([5.0, 5.0])
    a, b = _fit_linear(x, y)
    assert a == 1.0 and b == 0.0   # <3 pts -> identity fallback


def _mk_full_siblings(n_poly=60, noise=0.05, seed=7):
    """Every polymer carries all 7 targets, each a linear fn of a latent z
    (+ tiny noise) -> the learned arm should reconstruct each target from the
    other 6 siblings almost perfectly."""
    rng = np.random.RandomState(seed)
    z = rng.rand(n_poly)
    w = {"eea": 1.0, "egb": 1.7, "egc": 2.3, "ei": 3.1, "eps": 4.2, "nc": 5.1, "tg": 6.2}
    canon, tgt, Y = [], [], []
    for p in range(n_poly):
        for t in TARGETS_DEC:
            canon.append(f"p{p:03d}")
            tgt.append(t)
            Y.append(3.0 + w[t] * z[p] + noise * rng.randn())
    return (np.array(canon, dtype=str), np.array(tgt, dtype=str),
            np.array(Y, dtype=np.float64))


def test_learned_arm_recovers_siblings():
    from decoder_v16 import learned_arm
    canon, tgt, Y = _mk_full_siblings()
    piv = build_pivot_df(canon, tgt, Y)
    # drop ~1/3 of train rows as 'test' canon set (new polymers)
    te_idx = np.arange(len(canon))[::3]
    te_canon = canon[te_idx].copy()
    lo_tr, lo_te = learned_arm(canon, tgt, Y, piv, te_canon, global_folds=5, seed=42)
    assert lo_tr.shape == (len(canon), 7) and lo_te.shape == (len(te_idx), 7)
    # each target's OOF arm is only defined on its own target rows
    assert np.isnan(lo_tr).sum() == len(canon) * 6
    # eps is linearly reconstructible from the other 6 sibling sensors
    i_eps = TARGET_IDX_DEC["eps"]
    m = tgt == "eps"
    assert np.isfinite(lo_tr[m, i_eps]).all()
    r2 = 1.0 - np.mean((Y[m] - lo_tr[m, i_eps]) ** 2) / np.var(Y[m])
    assert r2 > 0.90


def test_learned_arm_nan_on_sibling_less_polys():
    from decoder_v16 import learned_arm
    # only ADD single-target polymers -> their canon has no siblings in pivot,
    # so the learned arm must REPORT NaN (caller falls back to target mean)
    canon = np.array(["P1", "P2", "P3"], dtype=str)
    tgt = np.array(["eps"] * 3, dtype=str)
    Y = np.array([1.0, 2.0, 3.0])
    piv = build_pivot_df(np.array(["P1", "P2", "P3", "Q"], dtype=str),
                         np.array(["eps", "eps", "eps", "eps"], dtype=str),
                         np.array([1.0, 2.0, 3.0, 9.0]))
    lo_tr, _ = learned_arm(canon, tgt, Y, piv, np.array(["NEW"], dtype=str), global_folds=2, seed=1)
    assert np.isnan(lo_tr).all()


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
