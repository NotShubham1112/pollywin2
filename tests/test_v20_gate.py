import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from v20_gate_report import compute_gate_report, write_submission


def test_dict_keys_match_contract():
    d = compute_gate_report(0.01, 0.0, [0.1, 0.2, 0.3])
    assert set(d.keys()) == {"pass", "mean_delta", "worst_delta", "alphas_ok"}


def test_pass_when_all_thresholds_met():
    d = compute_gate_report(0.01, 0.02, [0.1, 0.2, 0.3])
    assert d["pass"] is True
    assert d["alphas_ok"] is True


def test_fail_on_low_mean():
    d = compute_gate_report(0.001, 0.0, [0.1])
    assert d["pass"] is False
    assert d["mean_delta"] == 0.001


def test_fail_on_worst_target_below_threshold():
    d = compute_gate_report(0.01, -0.01, [0.1])
    assert d["pass"] is False


def test_fail_on_alpha_over_cap():
    d = compute_gate_report(0.01, 0.0, [0.1, 0.35])
    assert d["pass"] is False
    assert d["alphas_ok"] is False


def test_boundary_equality_passes():
    # exactly at thr_mean, exactly at -thr_worst, exactly at alpha_cap -> pass
    d = compute_gate_report(0.003, -0.003, [0.3])
    assert d["pass"] is True
    assert d["alphas_ok"] is True


def test_default_thresholds_are_pre_registered():
    import inspect
    params = inspect.signature(compute_gate_report).parameters
    assert params["thr_mean"].default == 0.003
    assert params["thr_worst"].default == 0.003
    assert params["alpha_cap"].default == 0.30


def test_write_submission_header_and_row_count(tmp_path):
    ids = [f"row_{i:05d}" for i in range(4940)]
    df = pd.DataFrame({"id": ids, "target": np.linspace(0, 1, 4940)})
    path = tmp_path / "sub.csv"
    write_submission(df, str(path))
    out = pd.read_csv(path)
    assert list(out.columns) == ["id", "target"]
    assert len(out) == 4940


def test_write_submission_preserves_id_order(tmp_path):
    rng = np.random.default_rng(0)
    ids = [f"X{i:04d}" for i in range(4940)]
    order = rng.permutation(4940)
    df = pd.DataFrame({"id": [ids[i] for i in order],
                       "target": rng.normal(size=4940)})
    path = tmp_path / "sub.csv"
    write_submission(df, str(path))
    out = pd.read_csv(path)
    assert np.array_equal(out["id"].values, df["id"].values)


def test_write_submission_rejects_wrong_row_count(tmp_path):
    df = pd.DataFrame({"id": [f"r{i}" for i in range(100)],
                       "target": np.zeros(100)})
    with pytest.raises(ValueError):
        write_submission(df, str(tmp_path / "bad.csv"))