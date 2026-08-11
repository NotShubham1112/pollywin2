"""Unit tests for the v21 local gate harness (vault/r2_sibling_validate.py).

Synthetic data only (3 targets x ~40 rows, 2 folds) so the tests run in
seconds. Mirrors the style of tests/test_v20_arm_cv.py (alignment/shape) and
tests/test_v20_gate.py (pure gate-boundary checks).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vault"))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from r2_sibling_validate import (
    ALPHA_GRID,
    build_feats,
    build_sib_arm,
    blend_driver,
    gate_1_leak_audit,
    gate_report,
    recompute_twin,
    ridge_oof,
)

SYN_TARGETS = ("aa", "bb", "cc")


def make_syn_data(n_per=40, n_te=40, targets=SYN_TARGETS, seed=0, groups_per=5):
    """3-target synthetic bundle: twin/test matrices + aligned labels/groups."""
    rng = np.random.default_rng(seed)
    n_tr = n_per * len(targets)
    twin = rng.standard_normal((n_tr, len(targets))).astype(np.float64)
    lgb_te = rng.standard_normal((n_te, len(targets))).astype(np.float64)
    T = np.repeat(np.array(list(targets)), n_per)
    T_te = np.array([targets[i % len(targets)] for i in range(n_te)])
    G = np.array([f"G{i // groups_per:04d}" for i in range(n_tr)])
    Y = (twin[:, 0] * 0.7 - twin[:, 1] * 0.3 + twin[:, 2] * 0.2).astype(np.float64)
    idx_of_target = {t: np.where(T == t)[0] for t in targets}
    target_mean = {t: float(Y[idx_of_target[t]].mean()) for t in targets}
    return {
        "twin": twin, "lgb_te": lgb_te, "Y": Y, "T": T, "T_te": T_te, "G": G,
        "idx_of_target": idx_of_target, "target_mean": target_mean,
    }


# ---------------------------------------------------------------------------
# build_feats
# ---------------------------------------------------------------------------
def test_build_feats_column_count_miss_flag_impute():
    targets = ("aa", "bb", "cc")
    mean = {"aa": 1.0, "bb": 2.0, "cc": 3.0}
    twin = np.array([[np.nan, 0.5, np.nan],
                     [1.0, np.nan, 3.0]])
    feats = build_feats(twin, ("bb", "cc"), targets, mean)
    # 2 cols per sibling target (value + miss flag), no self column
    assert feats.shape == (2, 4)
    # bb: row0 0.5 (not missing), row1 NaN -> TARGET_MEAN["bb"]=2.0 + miss flag
    np.testing.assert_allclose(feats[:, 0], [0.5, 2.0])
    np.testing.assert_allclose(feats[:, 1], [0.0, 1.0])
    # cc: row0 NaN -> 3.0 + miss flag, row1 3.0 (not missing)
    np.testing.assert_allclose(feats[:, 2], [3.0, 3.0])
    np.testing.assert_allclose(feats[:, 3], [1.0, 0.0])


def test_build_feats_no_self_column():
    targets = ("aa", "bb", "cc")
    mean = {t: 0.0 for t in targets}
    twin = np.zeros((5, 3))
    feats = build_feats(twin, ("bb", "cc"), targets, mean)
    assert feats.shape == (5, 4)  # self ("aa") excluded -> only bb, cc
    feats2 = build_feats(twin, ("aa", "bb", "cc"), targets, mean)
    assert feats2.shape == (5, 6)


# ---------------------------------------------------------------------------
# ridge_oof
# ---------------------------------------------------------------------------
def test_ridge_oof_shape_finite_alpha():
    rng = np.random.default_rng(0)
    n, d, n_te = 60, 4, 20
    X = rng.standard_normal((n, d))
    Xte = rng.standard_normal((n_te, d))
    G = np.array([f"G{i // 5:03d}" for i in range(n)])
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(n)
    cv = list(GroupKFold(n_splits=5).split(X, y, G))
    oof, te, a = ridge_oof(X, Xte, y, cv, [0.1, 10.0])
    assert oof.shape == (n,) and te.shape == (n_te,)
    assert np.isfinite(oof).all() and np.isfinite(te).all()
    assert a in (0.1, 10.0)


def test_ridge_oof_oof_matches_manual_fold_math():
    """OOF must be the fold-computed predictions at the returned alpha, written
    at each row's original position (fold val indices)."""
    rng = np.random.default_rng(1)
    n, d = 48, 3
    X = rng.standard_normal((n, d))
    Xte = rng.standard_normal((10, d))
    G = np.array([f"P{i // 4:03d}" for i in range(n)])
    y = 1.5 * X[:, 2] - 0.5 * X[:, 0] + 0.05 * rng.standard_normal(n)
    cv = list(GroupKFold(n_splits=4).split(X, y, G))
    oof, _, a = ridge_oof(X, Xte, y, cv, ALPHA_GRID)
    manual = np.zeros(n)
    for tr, vk in cv:
        manual[vk] = Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[vk])
    assert np.allclose(oof, manual, rtol=1e-8, atol=1e-8)


def test_ridge_oof_permutation_alignment():
    """Re-running on a consistently-shuffled copy, then un-shuffling, must
    reproduce the un-shuffled OOF (fixed alpha so only BLAS reduction order
    varies -> float tolerance, not bit-exact)."""
    rng = np.random.default_rng(11)
    n, d = 48, 3
    X = rng.standard_normal((n, d))
    Xte = rng.standard_normal((12, d))
    G = np.array([f"Q{i // 4:03d}" for i in range(n)])
    y = 1.5 * X[:, 2] - 0.5 * X[:, 0] + 0.05 * rng.standard_normal(n)
    cv = list(GroupKFold(n_splits=3).split(X, y, G))
    oof, _, a = ridge_oof(X, Xte, y, cv, [1.0])
    perm = rng.permutation(n)
    cv2 = list(GroupKFold(n_splits=3).split(X[perm], y[perm], G[perm]))
    oof2, _, _ = ridge_oof(X[perm], Xte, y[perm], cv2, [1.0])
    assert np.allclose(oof, oof2[np.argsort(perm)], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# recompute_twin (verbatim mirror of leak_safe_oof_scores)
# ---------------------------------------------------------------------------
def test_recompute_twin_shapes_and_deterministic():
    targets = SYN_TARGETS
    target_idx = {t: i for i, t in enumerate(targets)}
    rng = np.random.default_rng(0)
    n_tr, n_te, d = 90, 30, 8
    Xs = rng.standard_normal((n_tr, d)).astype(np.float32)
    Xtes = rng.standard_normal((n_te, d)).astype(np.float32)
    T = np.array([targets[i % 3] for i in range(n_tr)])
    G = np.array([f"C{i // 3:03d}" for i in range(n_tr)])
    Y = rng.standard_normal(n_tr).astype(np.float32)
    idx_of_target = {t: np.where(T == t)[0] for t in targets}

    kw = dict(Xs=Xs, Xtes=Xtes, Y=Y, G=G, idx_of_target=idx_of_target,
              TARGET_IDX=target_idx, GLOBAL_FOLDS=2, EARLY_HOLDOUT=0.15,
              SEED=42, n_estimators=50)
    s1, t1 = recompute_twin(**kw)
    s2, t2 = recompute_twin(**kw)
    assert s1.shape == (n_tr, 3) and t1.shape == (n_te, 3)
    assert s1.dtype == np.float32 and t1.dtype == np.float32
    assert np.isfinite(s1).all() and np.isfinite(t1).all()
    assert np.array_equal(s1, s2) and np.array_equal(t1, t2)


# ---------------------------------------------------------------------------
# build_sib_arm
# ---------------------------------------------------------------------------
def test_build_sib_arm_shapes():
    d = make_syn_data()
    sib_oof, sib_test, sib_only_r2 = build_sib_arm(
        d["twin"], d["lgb_te"], SYN_TARGETS, d["target_mean"], d["Y"], d["T"],
        d["G"], d["T_te"], d["idx_of_target"], GLOBAL_FOLDS=2,
        alpha_grid=ALPHA_GRID)
    assert sib_oof.shape == (len(d["Y"]),)
    assert sib_test.shape == (len(d["lgb_te"]),)
    assert set(sib_only_r2.keys()) == set(SYN_TARGETS)
    assert np.isfinite(sib_oof).all()
    assert np.isfinite(sib_test).all()
    # per-target placement: OOF populated exactly at each target's rows
    for t in SYN_TARGETS:
        assert np.isfinite(sib_oof[d["idx_of_target"][t]]).all()
    # every test row is covered by exactly one target's arm
    for t in SYN_TARGETS:
        idx_te = np.where(d["T_te"] == t)[0]
        assert np.isfinite(sib_test[idx_te]).all()


# ---------------------------------------------------------------------------
# gate_1_leak_audit
# ---------------------------------------------------------------------------
def _make_audit_data(seed=3):
    rng = np.random.default_rng(seed)
    targets = SYN_TARGETS
    n_tr = 6 * len(targets)
    T = np.array([targets[i % len(targets)] for i in range(n_tr)])
    G = np.array([f"P{i // 2:02d}" for i in range(n_tr)])  # 2 rows / polymer
    Y = rng.integers(0, 20, size=n_tr).astype(np.float64)  # exactly f32-representable
    twin = rng.standard_normal((n_tr, len(targets))).astype(np.float32)
    idx_of_target = {t: np.where(T == t)[0] for t in targets}
    trf = pd.DataFrame({"target": Y, "target_type": T, "canon": G})
    return twin, trf, idx_of_target


def test_gate1_zero_on_leak_safe_features():
    twin, trf, idx = _make_audit_data()
    assert gate_1_leak_audit(twin, trf, idx, folds=2) == 0


def test_gate1_positive_when_leak_injected():
    twin, trf, idx = _make_audit_data()
    # polymer P00 = rows {0,1}; row0 target aa, row1 target bb.
    assert trf["target_type"][0] == "aa" and trf["target_type"][1] == "bb"
    assert trf["canon"][0] == trf["canon"][1]
    twin[0, 1] = np.float32(trf["target"].values[1])  # aa-row twin==bb label
    assert gate_1_leak_audit(twin, trf, idx, folds=2) > 0


# ---------------------------------------------------------------------------
# blend_driver
# ---------------------------------------------------------------------------
def test_blend_driver_shapes_and_keys():
    rng = np.random.default_rng(5)
    targets = SYN_TARGETS
    n_tr = 90
    oof_g = rng.normal(size=n_tr)
    oof_m = rng.normal(size=n_tr)
    sib = rng.normal(size=n_tr)
    T = np.array([targets[i % 3] for i in range(n_tr)])
    G = np.array([f"B{i // 3:03d}" for i in range(n_tr)])
    Y = rng.normal(size=n_tr)
    r2_p14, r2_v21, alphas, w_sib = blend_driver(
        oof_g, oof_m, sib, Y, T, G, targets, n_splits=2)
    assert set(r2_p14.keys()) == set(targets)
    assert set(r2_v21.keys()) == set(targets)
    assert set(alphas.keys()) == set(targets)
    assert set(w_sib.keys()) == set(targets)
    assert all(np.isfinite(list(r2_p14.values())))
    assert all(np.isfinite(list(r2_v21.values())))
    assert all(np.isfinite(list(alphas.values())))
    assert all(np.isfinite(list(w_sib.values())))


# ---------------------------------------------------------------------------
# gate_report
# ---------------------------------------------------------------------------
def _p14_zero():
    return {"eps": 0.0, "nc": 0.0, "ei": 0.0}


def test_gate_report_keys_contract():
    r = gate_report(_p14_zero(), {t: 0.001 for t in _p14_zero()})
    assert {"gate0", "gate1", "gate2_soft", "gate2_strong", "gate3",
            "pass"} <= set(r.keys())


def test_gate_report_soft_boundary_equality_passes():
    p14 = _p14_zero()
    r = gate_report(p14, {t: 0.0015 for t in p14}, leak_count=0)
    assert r["gate2_soft"] is True
    assert r["gate2_strong"] is False
    assert r["pass"] is True


def test_gate_report_soft_boundary_below_fails():
    p14 = _p14_zero()
    r = gate_report(p14, {t: 0.0014 for t in p14}, leak_count=0)
    assert r["gate2_soft"] is False
    assert r["pass"] is False


def test_gate_report_strong_boundary_equality_passes():
    p14 = _p14_zero()
    r = gate_report(p14, {t: 0.003 for t in p14}, leak_count=0)
    assert r["gate2_soft"] is True
    assert r["gate2_strong"] is True


def test_gate_report_worst_boundary():
    p14 = _p14_zero()
    r = gate_report(p14, {"eps": -0.003, "nc": 0.0, "ei": 0.0}, leak_count=0)
    assert r["gate3"] is True
    r2 = gate_report(p14, {"eps": -0.0031, "nc": 0.0, "ei": 0.0}, leak_count=0)
    assert r2["gate3"] is False
    assert r2["pass"] is False


def test_gate_report_fails_on_any_leak():
    p14 = _p14_zero()
    r = gate_report(p14, {t: 0.003 for t in p14}, leak_count=1)
    assert r["gate1"] == 1
    assert r["pass"] is False


def test_gate_report_gate0_diagnostic_passthrough():
    p14 = _p14_zero()
    sib = {"eps": 0.0, "nc": 0.01, "ei": 0.0, "eea": 0.5}
    r = gate_report(p14, {t: 0.001 for t in p14}, sib_only_r2=sib, leak_count=0)
    assert r["gate0"] == sib


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
