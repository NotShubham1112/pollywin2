import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import GroupKFold

from v22_arm_cv import compute_bert_arm, compute_bert_only_r2

TARGETS = ("eea", "egb", "egc", "ei", "eps", "nc", "tg")
SLOPES = np.array([1.0, 1.6, 2.2, 0.7, 2.7, 1.2, 0.9])


def make_linear_data(n_tr=60, n_te=20, d=8, groups_per_polymer=5, seed=0):
    rng = np.random.default_rng(seed)
    pool_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    pool_te = rng.standard_normal((n_te, d)).astype(np.float32)
    tt_tr = np.array([TARGETS[(i * 3) % len(TARGETS)] for i in range(n_tr)])
    tt_te = np.array([TARGETS[(i * 3) % len(TARGETS)] for i in range(n_te)])
    g_tr = np.array([f"P{i // groups_per_polymer:03d}" for i in range(n_tr)])
    col = {t: j for j, t in enumerate(TARGETS)}
    j_tr = np.array([col[t] for t in tt_tr])
    y = (SLOPES[j_tr] * pool_tr[np.arange(n_tr), j_tr].astype(np.float64)).astype(np.float32)
    return pool_tr, pool_te, y, tt_tr, tt_te, g_tr


def test_interface_signature():
    sig = inspect.signature(compute_bert_arm)
    assert list(sig.parameters) == [
        "pool_tr", "pool_te", "y", "tt_tr", "tt_te", "g", "n_splits", "seed"]
    assert sig.parameters["n_splits"].default == 5
    assert sig.parameters["seed"].default == 42


def test_oof_alignment_and_shape():
    pool_tr, pool_te, y, tt_tr, tt_te, g = make_linear_data()
    oof, test = compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                 n_splits=5, seed=42)
    assert oof.shape == (60,)
    assert test.shape == (20,)
    assert np.isfinite(oof).all() and np.isfinite(test).all()


def test_groupkfold_never_splits_duplicated_smiles():
    n = 48
    groups = np.array([f"G{i // 4:03d}" for i in range(n)])
    gkf = GroupKFold(n_splits=5)
    fold = np.empty(n, dtype=int)
    for f, (_, va) in enumerate(gkf.split(np.zeros(n), np.zeros(n), groups)):
        fold[va] = f
    for grp in np.unique(groups):
        idx = np.where(groups == grp)[0]
        assert np.unique(fold[idx]).size == 1


def test_synthetic_linear_recovery():
    pool_tr, pool_te, y, tt_tr, tt_te, g = make_linear_data(n_tr=240, n_te=80)
    oof, test = compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                 n_splits=5, seed=42)
    for t in np.unique(tt_tr):
        idx = np.where(tt_tr == t)[0]
        assert np.corrcoef(oof[idx], y[idx])[0, 1] > 0.98


def test_deterministic_across_runs():
    pool_tr, pool_te, y, tt_tr, tt_te, g = make_linear_data()
    o1, t1 = compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g, n_splits=5, seed=42)
    o2, t2 = compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g, n_splits=5, seed=42)
    assert np.array_equal(o1, o2) and np.array_equal(t1, t2)


def test_bert_only_r2_keys_and_range():
    pool_tr, pool_te, y, tt_tr, tt_te, g = make_linear_data(n_tr=210, n_te=70)
    oof, _ = compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g, n_splits=5, seed=42)
    r2 = compute_bert_only_r2(oof, y, tt_tr)
    assert set(r2.keys()) == set(np.unique(tt_tr))
    assert all(np.isfinite(v) and -1e6 < v <= 1.0 for v in r2.values())


def test_all_rows_share_one_group_never_crashes():
    rng = np.random.default_rng(5)
    n_tr, n_te, d = 40, 16, 6
    pool_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    pool_te = rng.standard_normal((n_te, d)).astype(np.float32)
    tt_tr = np.array([TARGETS[i % 7] for i in range(n_tr)])
    tt_te = np.array([TARGETS[i % 7] for i in range(n_te)])
    g = np.full(n_tr, "ONE_GROUP")
    y = (2.0 * pool_tr[:, 0] + 0.5).astype(np.float32)
    oof, test = compute_bert_arm(pool_tr, pool_te, y, tt_tr, tt_te, g,
                                 n_splits=5, seed=42)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
