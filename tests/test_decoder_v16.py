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
