import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

MOD = Path(__file__).resolve().parents[1]
REPO = MOD.parent


def test_harness_smoke_runs_and_writes_report(tmp_path):
    env = dict(os.environ)
    env["SMOKE"] = "1"
    # the harness writes to its default out_dir (relative to cwd=REPO)
    out = REPO / "vault" / "pipeline_out_v22"
    # run the harness in a subprocess so module state is clean
    r = subprocess.run(
        [sys.executable, "-X", "utf8", str(MOD / "run_v22_gate.py")],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(out / "v22_gate_report.csv")
    assert list(df.columns)[:3] == ["target", "r2_p14", "r2_v22"]
    assert "v22_gate_report.csv" in r.stdout


def test_harness_importable_functions():
    sys.path.insert(0, str(MOD))
    import run_v22_gate as H
    assert hasattr(H, "run_gate")
    assert H.P14_MEAN == 0.8641
    assert H.TARGETS == ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
