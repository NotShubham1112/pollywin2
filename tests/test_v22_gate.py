import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from v22_gate_report import (
    SOFT_DELTA, STRONG_DELTA, WORST_TOL, gate_1_leak_audit, gate_report,
    write_submission,
)


def _p14():
    return {"eea": 0.85, "egb": 0.86, "egc": 0.84, "ei": 0.87,
            "eps": 0.88, "nc": 0.89, "tg": 0.90}


def _v22(p14, delta):
    return {t: p14[t] + delta for t in p14}


def test_thresholds_pre_registered():
    assert SOFT_DELTA == 0.0015
    assert STRONG_DELTA == 0.003
    assert WORST_TOL == 0.003


def test_dict_keys_contract():
    r = gate_report(_p14(), _p14())
    assert {"gate0", "gate1", "gate2_soft", "gate2_strong", "gate3",
            "pass"} <= set(r.keys())


def test_soft_boundary_equality_passes():
    p14 = _p14()
    r = gate_report(p14, _v22(p14, 0.0015), leak_count=0)
    assert r["gate2_soft"] is True
    assert r["gate2_strong"] is False
    assert r["pass"] is True


def test_soft_boundary_below_fails():
    p14 = _p14()
    r = gate_report(p14, _v22(p14, 0.0014), leak_count=0)
    assert r["gate2_soft"] is False
    assert r["pass"] is False


def test_strong_boundary_equality_passes():
    p14 = _p14()
    r = gate_report(p14, _v22(p14, 0.003), leak_count=0)
    assert r["gate2_soft"] is True
    assert r["gate2_strong"] is True


def test_eps_nc_ei_and_overall_both_required():
    p14 = _p14()
    v22 = _v22(p14, 0.003)
    v22["eps"] = p14["eps"] - 0.01            # eps/nc/ei mean now ~0.0
    r = gate_report(p14, v22, leak_count=0)
    assert r["gate2_soft"] is False


def test_worst_boundary():
    p14 = _p14()
    v22 = _v22(p14, 0.003)
    v22["tg"] = p14["tg"] - 0.003
    r = gate_report(p14, v22, leak_count=0)
    assert r["gate3"] is True
    v22["tg"] = p14["tg"] - 0.0031
    r2 = gate_report(p14, v22, leak_count=0)
    assert r2["gate3"] is False
    assert r2["pass"] is False


def test_fails_on_any_leak():
    p14 = _p14()
    r = gate_report(p14, _v22(p14, 0.003), leak_count=1)
    assert r["gate1"] == 1
    assert r["pass"] is False


def test_gate0_diagnostic_passthrough():
    p14 = _p14()
    b = {"eps": 0.1, "nc": 0.2, "ei": 0.0}
    r = gate_report(p14, _v22(p14, 0.003), bert_only_r2=b, leak_count=0)
    assert r["gate0"] == b
    r2 = gate_report(p14, _p14())
    assert r2["gate0"] == {}


def test_write_submission_header_and_rows(tmp_path):
    ids = [f"row_{i:05d}" for i in range(4940)]
    df = pd.DataFrame({"id": ids, "target": np.linspace(0, 1, 4940)})
    path = write_submission(df, str(tmp_path / "sub.csv"))
    out = pd.read_csv(path)
    assert list(out.columns) == ["id", "target"]
    assert len(out) == 4940


def test_write_submission_rejects_wrong_shape(tmp_path):
    df = pd.DataFrame({"id": ["r%d" % i for i in range(100)],
                       "target": np.zeros(100)})
    with pytest.raises(ValueError):
        write_submission(df, str(tmp_path / "bad.csv"))
    bad_cols = pd.DataFrame({"id": ["r0"], "t": [0.0]})
    with pytest.raises(ValueError):
        write_submission(bad_cols, str(tmp_path / "bad2.csv"))


def _make_audit_data(seed=3):
    rng = np.random.default_rng(seed)
    targets = ("aa", "bb", "cc")
    n_tr = 6 * len(targets)
    T = np.array([targets[i % len(targets)] for i in range(n_tr)])
    G = np.array([f"P{i // 2:02d}" for i in range(n_tr)])
    Y = rng.integers(0, 20, size=n_tr).astype(np.float64)
    F = rng.standard_normal((n_tr, 2)).astype(np.float32)
    idx_of_target = {t: np.where(T == t)[0] for t in targets}
    trf = pd.DataFrame({"target": Y, "target_type": T, "canon": G})
    return F, trf, idx_of_target


def test_gate1_zero_on_clean_features():
    F, trf, idx = _make_audit_data()
    assert gate_1_leak_audit(F, trf, idx, folds=2) == 0


def test_gate1_positive_when_leak_injected():
    F, trf, idx = _make_audit_data()
    assert trf["target_type"][0] == "aa" and trf["target_type"][1] == "bb"
    assert trf["canon"][0] == trf["canon"][1]
    F[0, 0] = np.float32(trf["target"].values[1])
    assert gate_1_leak_audit(F, trf, idx, folds=2) > 0
