import sys, pathlib, os, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from vault.compare_v16 import evaluate_gates, TARGETS

def _mk_fake_npz(arm_cover=0.5, arm_gain=1.0):
    """Synthetic blend_oof_test16-like dict.

    base arms (GBM, MT) are weak; the two decoder arms are strong copies of y
    present on the first `arm_cover` fraction of rows so evaluate_gates has a
    deterministic covered subset to measure the v16-vs-v14 gain on.
    """
    rng = np.random.RandomState(202)
    n, m = 240, 60
    g_all = np.array([f"p{i // 4:03d}" for i in range(n)], dtype=object)
    t_all = np.array([TARGETS[i % 7] for i in range(n)], dtype=object)
    y = rng.randn(n)
    oof_g = 0.55 * y + 0.9 * rng.randn(n)
    oof_m = 0.5 * y + 0.87 * rng.randn(n)
    cover = np.zeros(n, bool)
    cover[: int(n * arm_cover)] = True
    oof_ph = np.full(n, np.nan)
    oof_le = np.full(n, np.nan)
    oof_ph[cover] = arm_gain * y[cover]
    oof_le[cover] = (arm_gain * y[cover] + 0.1 * rng.randn(int(n * arm_cover)))
    return dict(
        oof_gbm=oof_g.astype(np.float32), oof_mt=oof_m.astype(np.float32),
        oof_phys=oof_ph, oof_learn=oof_le,
        test_gbm=np.zeros(m), test_mt=np.zeros(m),
        test_phys=np.zeros(m), test_learn=np.zeros(m),
        y_all=y, g_all=g_all, t_all=t_all,
        blends_tr=np.zeros(n), blends_te=np.zeros(m))

def test_gate_runs_and_returns_dict():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.npz")
        report = os.path.join(d, "v16_gates_report.csv")
        np.savez(p, **(_mk_fake_npz()))
        res = evaluate_gates(p, report_path=report)
        assert res is not None and "gate_pass" in res
        assert "per_target" in res and "small_five_mean_gain" in res and "message" in res
        assert os.path.exists(report)

def test_gate_passes_when_arms_help():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.npz")
        np.savez(p, **(_mk_fake_npz(arm_gain=3.0)))
        res = evaluate_gates(p)
        assert res["gate_pass"] is True

def test_gate_fails_when_arms_dead():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.npz")
        dd = _mk_fake_npz(arm_gain=3.0)
        dd["oof_phys"] = np.full(len(dd["y_all"]), np.nan)
        dd["oof_learn"] = np.full(len(dd["y_all"]), np.nan)
        np.savez(p, **dd)
        res = evaluate_gates(p)
        assert res["gate_pass"] is False

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
