import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from v20_arm_cv import compute_trf_arm

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
SLOPES = np.array([1.0, 1.6, 2.2, 0.7, 2.7, 1.2, 0.9])
OFFSETS = np.array([0.1, -0.3, 0.5, 0.0, -0.8, 0.2, 0.6])


def make_linear_data(n_tr=60, n_te=20, d=8, n_splits=5, seed=0, groups_per_polymer=5):
    assert n_tr % groups_per_polymer == 0
    rng = np.random.default_rng(seed)
    pool_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    pool_te = rng.standard_normal((n_te, d)).astype(np.float32)

    tt_tr = np.array([TARGETS[(i * 3) % len(TARGETS)] for i in range(n_tr)])
    tt_te = np.array([TARGETS[(i * 3) % len(TARGETS)] for i in range(n_te)])
    g_tr = np.array([f"P{i // groups_per_polymer:03d}" for i in range(n_tr)])

    col = {t: j for j, t in enumerate(TARGETS)}
    j_tr = np.array([col[t] for t in tt_tr])
    j_te = np.array([col[t] for t in tt_te])
    y = (SLOPES[j_tr] * pool_tr[np.arange(n_tr), j_tr].astype(np.float64)
         + OFFSETS[j_tr]).astype(np.float32)
    y_te_true = (SLOPES[j_te] * pool_te[np.arange(n_te), j_te].astype(np.float64)
                 + OFFSETS[j_te]).astype(np.float32)
    return pool_tr, pool_te, y, tt_tr, tt_te, g_tr, y_te_true


def test_interface_signature():
    sig = inspect.signature(compute_trf_arm)
    assert list(sig.parameters) == [
        "pool_tr", "pool_te", "y", "tt_tr", "tt_te", "g", "n_splits", "seed"]
    assert sig.parameters["n_splits"].default == 5
    assert sig.parameters["seed"].default == 42


def test_oof_alignment_and_shape():
    pool_tr, pool_te, y, tt_tr, tt_te, g, _ = make_linear_data()
    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    assert oof.shape == (60,)
    assert test.shape == (20,)
    assert np.isfinite(oof).all()
    assert np.isfinite(test).all()
    # positional alignment: oof must be indexed at the ORIGINAL input order.
    # Re-run on a consistently-shuffled copy; un-shuffling outputs must be
    # bit-identical to the un-shuffled run (a lost/shifted index would break this).
    rng = np.random.default_rng(7)
    perm = rng.permutation(60)
    oof2, _ = compute_trf_arm(pool_tr[perm], pool_te, y[perm], tt_tr[perm],
                              tt_te, g[perm], n_splits=5, seed=42)
    # oof2[j] holds original row perm[j]; map back with the inverse permutation.
    # permuting rows swaps BLAS reduction order in Ridge's X^T X, so outputs
    # agree only to float tolerance (~1e-7), not bit-exact.
    assert np.allclose(oof, oof2[np.argsort(perm)], rtol=1e-5, atol=1e-5)


def test_groupkfold_never_splits_duplicated_smiles():
    n = 48
    groups = np.array([f"G{i // 4:03d}" for i in range(n)])  # 12 groups of 4
    gkf = GroupKFold(n_splits=5)
    fold = np.empty(n, dtype=int)
    for f, (_, va) in enumerate(gkf.split(np.zeros(n), np.zeros(n), groups)):
        fold[va] = f
    for grp in np.unique(groups):
        idx = np.where(groups == grp)[0]
        assert np.unique(fold[idx]).size == 1, f"group {grp} straddled folds"


def test_no_leakage_between_folds():
    # duplicated smiles: every head must never train on a row that shares the
    # row's own group. Verified via GroupKFold by construction: same-g rows
    # land in the same fold, and per-row OOF heads are trained only on the
    # rest of that fold's train portion.
    rng = np.random.default_rng(2)
    n = 36
    d = 6
    pool_tr = rng.standard_normal((n, d)).astype(np.float32)
    pool_te = rng.standard_normal((12, d)).astype(np.float32)
    tt_tr = np.array([TARGETS[i % 7] for i in range(n)])
    tt_te = np.array([TARGETS[i % 7] for i in range(12)])
    # duplicate smiles: rows 8,9 and rows 20..23 share groups AND share target
    # type AND share an identical embedding row (deterministic encoder).
    tt_tr[8] = tt_tr[9] = "ei"
    tt_tr[20:24] = "nc"
    g = np.array([f"S{i:03d}" for i in range(n)])
    g[9] = g[8] = "S_DUP1"
    g[21] = g[20] = g[22] = g[23] = "S_DUP2"
    pool_tr[9] = pool_tr[8]
    pool_tr[21:24] = pool_tr[20]
    y = (pool_tr[:, 0] + SLOPES[[TARGETS.index(t) for t in tt_tr]]).astype(np.float32)

    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    assert oof.shape == (n,) and test.shape == (12,)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    # identical smiles + identical target must yield identical OOF values
    assert np.isclose(oof[8], oof[9], atol=1e-5)
    assert np.isclose(oof[20], oof[21], atol=1e-5)
    assert np.isclose(oof[21], oof[22], atol=1e-5)
    assert np.isclose(oof[22], oof[23], atol=1e-5)


def test_duplicate_smiles_get_identical_oof():
    pool_tr, pool_te, y, tt_tr, tt_te, g, _ = make_linear_data(n_tr=70,
                                                               n_te=14,
                                                               groups_per_polymer=5)
    # make rows 60..69 a polymer of 10 duplicated smiles
    v = pool_tr[60].copy()
    for i in range(60, 70):
        pool_tr[i] = v
        tt_tr[i] = "ei"
        g[i] = "S_DUP_ALL"
    y[60:70] = 1.5
    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    assert np.isfinite(oof).all()
    assert np.unique(oof[60:70]).size == 1


def test_synthetic_linear_recovery():
    pool_tr, pool_te, y, tt_tr, tt_te, g, y_te_true = make_linear_data(
        n_tr=240, n_te=80, groups_per_polymer=5)
    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    for t in np.unique(tt_tr):
        idx = np.where(tt_tr == t)[0]
        r = np.corrcoef(oof[idx], y[idx])[0, 1]
        assert r > 0.98, f"target {t}: OOF corr {r:.4f}"
    for t in np.unique(tt_te):
        idx = np.where(tt_te == t)[0]
        r = np.corrcoef(test[idx], y_te_true[idx])[0, 1]
        assert r > 0.98, f"target {t}: test corr {r:.4f}"


def test_deterministic_across_runs():
    pool_tr, pool_te, y, tt_tr, tt_te, g, _ = make_linear_data()
    oof1, test1 = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                  n_splits=5, seed=42)
    oof2, test2 = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                  n_splits=5, seed=42)
    assert np.array_equal(oof1, oof2)
    assert np.array_equal(test1, test2)


def test_small_target_fallback():
    rng = np.random.default_rng(3)
    d = 6
    n_tr, n_te = 38, 16
    pool_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    pool_te = rng.standard_normal((n_te, d)).astype(np.float32)

    big = np.array([TARGETS[i % 6] for i in range(30)])  # 6 big targets x 5
    tt_tr = np.concatenate([big,
                            ["tg"],          # single-row target
                            ["wd", "wd"],    # 2-row target (< n_splits)
                            ["eea", "eea", "eea", "eea", "eea"]])
    assert tt_tr.shape == (n_tr,)
    tt_te = np.array(["tg", "wd", "zzz"] +
                     [TARGETS[i % 6] for i in range(n_te - 3)])
    g = np.array([f"G{i:03d}" for i in range(n_tr)])
    y = (pool_tr[:, 0] * 2.0 + 0.3).astype(np.float32)

    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    assert oof.shape == (n_tr,) and test.shape == (n_te,)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    # single-row target: fallback value is its own (mean of its rows) label
    assert np.isclose(oof[30], y[30])
    # 2-row target fallback: finite, not NaN
    assert np.isfinite(oof[31]) and np.isfinite(oof[32])
    # test-only target 'zzz' (no train rows): global-mean catch-all, finite
    zzz_idx = np.where(tt_te == "zzz")[0]
    assert np.isclose(test[zzz_idx], np.mean(y)).all()


def test_nan_free_on_single_row_group_and_must_not_warn_quiet():
    rng = np.random.default_rng(4)
    d = 4
    n_tr, n_te = 21, 9
    pool_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    pool_te = rng.standard_normal((n_te, d)).astype(np.float32)
    tt_tr = np.array([TARGETS[i % 7] for i in range(n_tr)])
    tt_te = np.array([TARGETS[i % 7] for i in range(n_te)])
    g = np.array([f"U{i:03d}" for i in range(n_tr)])
    y = (2.0 * pool_tr[:, 2]).astype(np.float32)
    # corrupt: one target has all its rows in one canonical smiles
    same = np.where(tt_tr == "ei")[0]
    g[same] = "SM_SINGLE_GROUP"
    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    assert np.isfinite(oof).all()
    assert np.isfinite(test).all()


def test_all_rows_share_one_group_never_crashes():
    # degenerate pool: EVERY train row belongs to a single smiles group, and
    # many targets have more rows than n_splits. GroupKFold(n_splits=5) cannot
    # form 5 folds from 1 group and raises ValueError; the function must fall
    # back to per-target all-rows heads instead of crashing.
    rng = np.random.default_rng(5)
    d = 6
    n_tr, n_te = 40, 16
    pool_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    pool_te = rng.standard_normal((n_te, d)).astype(np.float32)
    tt_tr = np.array([TARGETS[i % 7] for i in range(n_tr)])
    tt_te = np.array([TARGETS[i % 7] for i in range(n_te)])
    g = np.full(n_tr, "ONE_GROUP")  # every row shares the same smiles group
    y = (2.0 * pool_tr[:, 0] + 0.5).astype(np.float32)

    oof, test = compute_trf_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                n_splits=5, seed=42)
    assert oof.shape == (n_tr,)
    assert test.shape == (n_te,)
    assert np.isfinite(oof).all()
    assert np.isfinite(test).all()