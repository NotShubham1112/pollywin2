"""v20 gate runner: fold-safe self-trained SMILES-encoder arm end-to-end.

Loads the frozen P14 arms (GBM + MT-GNN OOF/test predictions), builds a
leak-free self-supervised SMILES encoder arm (oof_trf/test_trf), blends the
three arms per target with the P14 fold-safe alpha sweep, then evaluates the
pre-registered gate against the P14 baseline (0.8641). If the gate passes a
v20 submission is written; if it fails P14 stays final.

Usage:
    python run_v20_gate.py          # FULL config
    $env:SMOKE='1'; python run_v20_gate.py   # SMOKE config (small, fast)

Env knobs: SMOKE, V20_SEED, V20_PI_COUNT, V20_D, V20_LAYERS, V20_EPOCHS.

Run with `python -X utf8` on Windows for the delta/Unicode gate output.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v20_codec import build_tokenizer, tokenize_batch
from v20_encoder import MaskEncoder, pool_embeddings, pretrain_encoder
from v20_arm_cv import compute_trf_arm
from v20_blend import blend_3d, ALPHAS
from v20_gate_report import compute_gate_report, write_submission

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
P14_MEAN = 0.8641
P14_TOL = 0.005

DATA_DIR = Path("competition/data/raw")
NPR_DIR = Path("vault") / "pipeline_out_pretrain"
OOF_PATH = NPR_DIR / "superblend_oof.npz"


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _config():
    smoke = os.environ.get("SMOKE", "").strip() == "1"
    cfg = {
        "smoke": smoke,
        "seed": int(os.environ.get("V20_SEED", "42")),
        "pi_count": int(os.environ.get("V20_PI_COUNT", "300" if smoke else "20000")),
        "d": int(os.environ.get("V20_D", "32" if smoke else "256")),
        "layers": int(os.environ.get("V20_LAYERS", "2" if smoke else "4")),
        "epochs": int(os.environ.get("V20_EPOCHS", "1" if smoke else "2")),
    }
    return cfg


def _load_data(cfg):
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    p1 = pd.read_csv(DATA_DIR / "PI1M.csv", nrows=cfg["pi_count"])
    z = np.load(OOF_PATH, allow_pickle=True)

    # Row-order alignment is critical: npz rows == train.csv/test.csv rows.
    tt_tr = z["target_type_train"]
    tt_te = z["target_type_test"]
    assert np.array_equal(tt_tr, tr["target_type"].values), (
        "npz target_type_train does not match train.csv target_type order")
    assert np.array_equal(tt_te, te["target_type"].values), (
        "npz target_type_test does not match test.csv target_type order")
    assert_float_close = np.abs(np.corrcoef(z["y_train"], tr["target"])[0, 1])
    assert assert_float_close > 0.999, "npz y_train not aligned with train.csv target"

    return {
        "train": tr, "test": te, "pi_smiles": p1["SMILES"].values,
        "oof_gbm": z["oof_gbm"].astype(float),
        "oof_mt": z["oof_mt"].astype(float),
        "test_gbm": z["test_gbm"].astype(float),
        "test_mt": z["test_mt"].astype(float),
        "y_train": z["y_train"].astype(float),
        "tt_tr": tt_tr, "tt_te": tt_te,
    }


def _p14_2arm_oof(M2, y, g, n_splits=5):
    """Fold-safe 2-arm OOF alpha scan (P14 baseline protocol).

    Same GroupKFold-on-g + per-alpha OOF r2 selection + refit-at-best as the
    3-arm blend, but on exactly the two P14 arms (gbm, mt).
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    M2 = np.asarray(M2, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    if n < 2:
        return y.copy()
    M = np.where(np.isfinite(M2), M2, np.nanmean(M2, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        return Ridge(alpha=ALPHAS[0]).fit(M, y).predict(M)
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, out = -np.inf, np.zeros(n)
    for a in ALPHAS:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, out = r, o.copy()
    return out


def run_gate(smoke=False, data_paths=None):
    """Run the full v20 gate pipeline.

    Returns a dict with per-target table rows, means, gate report, and the
    submission path if written (else None).
    """
    cfg = _config()
    if smoke:
        cfg["smoke"] = True
        # smoke=True forces the small config regardless of env (see brief).
        cfg["pi_count"] = 300
        cfg["d"] = 32
        cfg["layers"] = 2
        cfg["epochs"] = 1
    _seed_everything(cfg["seed"])

    print(
        f"[v20 gate] {'SMOKE' if cfg['smoke'] else 'FULL'} config: "
        f"PI1M={cfg['pi_count']} d={cfg['d']} layers={cfg['layers']} "
        f"epochs={cfg['epochs']} seed={cfg['seed']}")

    d = _load_data(cfg)

    # 2) Tokenizer on PI1M sample.
    print(f"[v20 gate] building tokenizer on {len(d['pi_smiles'])} PI1M rows ...")
    tok = build_tokenizer(d["pi_smiles"], max_vocab=1600, min_count=2)

    # 3) Tokenize train + test smiles.
    print("[v20 gate] tokenizing train/test smiles ...")
    ids_tr = tokenize_batch(tok, d["train"]["smiles"].values, max_len=128)
    ids_te = tokenize_batch(tok, d["test"]["smiles"].values, max_len=128)

    # 4) Pretrain the encoder (masked-region, labels-free).
    print("[v20 gate] pretraining MaskEncoder ...")
    model = MaskEncoder(
        vocab=len(tok["tok2id"]), d=cfg["d"], layers=cfg["layers"], max_len=128)
    pretrain_ids = np.concatenate([ids_tr, ids_te]).astype(np.int64)
    pretrain_encoder(
        model, pretrain_ids, epochs=cfg["epochs"],
        seed=cfg["seed"])

    # 5) Pooled embeddings.
    print("[v20 gate] pooling embeddings ...")
    pool_tr = pool_embeddings(model, ids_tr.astype(np.int64))
    pool_te = pool_embeddings(model, ids_te.astype(np.int64))

    # 6) TRF arm, fold-safe per-target Ridge on the frozen pool.
    print("[v20 gate] compute_trf_arm ...")
    oof_trf, test_trf = compute_trf_arm(
        pool_tr, pool_te, d["y_train"], d["tt_tr"], d["tt_te"],
        d["train"]["smiles"].values, n_splits=5, seed=cfg["seed"])

    n_tr = len(d["y_train"])
    oof_v20 = np.zeros(n_tr)
    alphas, r2_p14, r2_v20 = {}, {}, {}

    for t in TARGETS:
        idx = np.where(d["tt_tr"] == t)[0]
        g_t = d["train"]["smiles"].values[idx]
        M3 = np.column_stack([d["oof_gbm"][idx], d["oof_mt"][idx], oof_trf[idx]])
        oof_v20[idx], _coefs, alphas[t] = blend_3d(
            M3, d["y_train"][idx], g_t, alphas=ALPHAS, n_splits=5)
        r2_v20[t] = float(np.corrcoef(d["y_train"][idx], oof_v20[idx])[0, 1]) ** 2

        b2 = _p14_2arm_oof(
            np.column_stack([d["oof_gbm"][idx], d["oof_mt"][idx]]),
            d["y_train"][idx], g_t, n_splits=5)
        r2_p14[t] = float(np.corrcoef(d["y_train"][idx], b2)[0, 1]) ** 2

    mean_p14 = float(np.mean(list(r2_p14.values())))
    if abs(mean_p14 - P14_MEAN) > P14_TOL:
        raise AssertionError(
            f"recomputed P14 mean {mean_p14:.4f} deviates from reference "
            f"{P14_MEAN} by more than {P14_TOL}; the mean gate would be "
            f"misleading — P14 stays final")

    mean_v20 = float(np.mean(list(r2_v20.values())))
    # Mean gate compares against the recomputed honest P14 baseline so the
    # verdict is self-consistent (worst_delta already uses the recomputed
    # per-target baselines).
    mean_delta = mean_v20 - mean_p14
    worst_delta = float(min(r2_v20[t] - r2_p14[t] for t in TARGETS))
    report = compute_gate_report(
        mean_delta, worst_delta, list(alphas.values()),
        thr_mean=0.003, thr_worst=0.003, alpha_cap=0.30)

    print()
    print("==" * 34)
    print("target   r2_p14    r2_v20   delta    alpha")
    for t in TARGETS:
        print(
            f"{t:6s}   {r2_p14[t]:.4f}   {r2_v20[t]:.4f}   "
            f"{r2_v20[t]-r2_p14[t]:+.4f}   {alphas[t]:.2f}")
    print("-" * 34)
    print(f"mean_v20 {mean_v20:.4f}  mean_p14 {mean_p14:.4f}  "
          f"mean_delta {mean_delta:+.4f}")
    print(f"worst_delta {worst_delta:+.4f}  alphas_ok {report['alphas_ok']}")
    print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}")
    print("==" * 34)
    print(f"[v20 gate] GATE={'PASS' if report['pass'] else 'FAIL'} -> "
          f"{'writing submission' if report['pass'] else 'P14 stays final'}")

    rows = []
    for t in TARGETS:
        rows.append({
            "target": t, "r2_p14": r2_p14[t], "r2_v20": r2_v20[t],
            "delta": r2_v20[t] - r2_p14[t], "alpha": alphas[t]})
    rows.append({
        "target": "mean", "r2_p14": mean_p14, "r2_v20": mean_v20,
        "delta": mean_delta, "alpha": float(np.nan)})
    pd.DataFrame(rows).to_csv("vault/v20_gate_report.csv", index=False)

    submission_path = None
    if report["pass"]:
        test_pred = np.zeros(len(d["test"]))
        for t in TARGETS:
            idx = np.where(d["tt_tr"] == t)[0]
            idx_te = np.where(d["tt_te"] == t)[0]
            M_tr = np.column_stack([d["oof_gbm"][idx], d["oof_mt"][idx], oof_trf[idx]])
            M_te = np.column_stack(
                [d["test_gbm"][idx_te], d["test_mt"][idx_te], test_trf[idx_te]])
            from sklearn.linear_model import Ridge

            lr = Ridge(alpha=alphas[t], fit_intercept=True).fit(
                M_tr, d["y_train"][idx])
            test_pred[idx_te] = lr.predict(M_te)
        assert np.isfinite(test_pred).all(), "NaN in test predictions"
        submission_path = write_submission(
            pd.DataFrame({"id": d["test"]["id"].values, "target": test_pred}),
            "vault/submission_v20.csv")
        print(f"[v20 gate] wrote {submission_path}")

    return {
        "pass": report["pass"], "mean_v20": mean_v20, "mean_p14": mean_p14,
        "mean_delta": mean_delta, "worst_delta": worst_delta,
        "alphas": alphas, "r2_p14": r2_p14, "r2_v20": r2_v20,
        "submission_path": submission_path,
    }


if __name__ == "__main__":
    _smoke = os.environ.get("SMOKE", "").strip() == "1"
    run_gate(smoke=_smoke)