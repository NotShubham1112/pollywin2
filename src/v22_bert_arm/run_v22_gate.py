"""v22 gate runner: BPE + BERT MLM arm on P14, end-to-end (CPU, minutes).

Usage:
    python run_v22_gate.py          # FULL local config
    $env:SMOKE='1'; python run_v22_gate.py   # SMOKE config (small, fast)

Env knobs: SMOKE, V22_SEED, V22_PI_COUNT, V22_D, V22_LAYERS, V22_HEADS,
V22_EPOCHS, V22_BPE_SUBSET, V22_VOCAB.
Run with `python -X utf8` on Windows for unicode gate output.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v22_tokenizer import learn_bpe, tokenize_batch
from v22_encoder import BertEncoder, pool_embeddings, pretrain_mlm
from v22_arm_cv import compute_bert_arm, compute_bert_only_r2
from v22_blend import _p14_2arm_oof, blend_narm_oof
from v22_gate_report import (
    EPS_NC_EI, gate_1_leak_audit, gate_report, write_submission,
)

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
    if smoke:
        base = dict(pi_count=300, d=32, layers=2, heads=4, epochs=1,
                    bpe_subset=300, vocab=150, folds=2)
    else:
        base = dict(pi_count=5000, d=64, layers=2, heads=4, epochs=1,
                    bpe_subset=5000, vocab=400, folds=5)
    base.update({
        "smoke": smoke,
        "seed": int(os.environ.get("V22_SEED", "42")),
        "pi_count": int(os.environ.get("V22_PI_COUNT", str(base["pi_count"]))),
        "d": int(os.environ.get("V22_D", str(base["d"]))),
        "layers": int(os.environ.get("V22_LAYERS", str(base["layers"]))),
        "heads": int(os.environ.get("V22_HEADS", str(base["heads"]))),
        "epochs": int(os.environ.get("V22_EPOCHS", str(base["epochs"]))),
        "bpe_subset": int(os.environ.get("V22_BPE_SUBSET", str(base["bpe_subset"]))),
        "vocab": int(os.environ.get("V22_VOCAB", str(base["vocab"]))),
    })
    return base


def _load_data(cfg):
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    p1 = pd.read_csv(DATA_DIR / "PI1M.csv", nrows=cfg["pi_count"])
    z = np.load(OOF_PATH, allow_pickle=True)
    tt_tr = z["target_type_train"]
    tt_te = z["target_type_test"]
    assert np.array_equal(tt_tr, tr["target_type"].values), (
        "npz target_type_train does not match train.csv order")
    assert np.array_equal(tt_te, te["target_type"].values), (
        "npz target_type_test does not match test.csv order")
    c = np.abs(np.corrcoef(z["y_train"], tr["target"])[0, 1])
    assert c > 0.999, "npz y_train not aligned with train.csv target"
    return {
        "train": tr, "test": te, "pi_smiles": p1["SMILES"].values,
        "oof_gbm": z["oof_gbm"].astype(float),
        "oof_mt": z["oof_mt"].astype(float),
        "test_gbm": z["test_gbm"].astype(float),
        "test_mt": z["test_mt"].astype(float),
        "y_train": z["y_train"].astype(float),
        "tt_tr": tt_tr, "tt_te": tt_te,
    }


def run_gate(smoke=False, out_dir=None):
    cfg = _config()
    if smoke:
        cfg.update(smoke=True, pi_count=300, d=32, layers=2, heads=4,
                   epochs=1, bpe_subset=300, vocab=150, folds=2)
    _seed_everything(cfg["seed"])
    out_dir = Path(out_dir or "vault/pipeline_out_v22")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v22 gate] {'SMOKE' if cfg['smoke'] else 'FULL'} config: "
          f"PI1M={cfg['pi_count']} d={cfg['d']} layers={cfg['layers']} "
          f"heads={cfg['heads']} epochs={cfg['epochs']} seed={cfg['seed']}")

    d = _load_data(cfg)
    print(f"[v22 gate] learning BPE on {len(d['pi_smiles'])} PI1M rows ...")
    tok = learn_bpe(d["pi_smiles"], vocab_target=cfg["vocab"],
                    max_subset=cfg["bpe_subset"], seed=cfg["seed"])
    print("[v22 gate] tokenizing train/test ...")
    ids_tr = tokenize_batch(tok, d["train"]["smiles"].values, max_len=128)
    ids_te = tokenize_batch(tok, d["test"]["smiles"].values, max_len=128)

    print("[v22 gate] pretraining BertEncoder ...")
    protected_ids = tuple(tok["tok2id"].get(p, -1) for p in tok["protect"])
    protected_ids = tuple(p for p in protected_ids if p >= 0)
    model = BertEncoder(vocab=len(tok["tok2id"]), d=cfg["d"],
                        layers=cfg["layers"], heads=cfg["heads"])
    pretrain_ids = np.concatenate([ids_tr, ids_te]).astype(np.int64)
    n_val = max(1, int(0.05 * len(pretrain_ids)))
    pretrain_mlm(model, pretrain_ids[:-n_val], epochs=cfg["epochs"],
                 bs=256, lr=3e-4, seed=cfg["seed"], mask_p=0.15,
                 protected_ids=protected_ids, val_ids=pretrain_ids[-n_val:],
                 device="cpu")

    print("[v22 gate] pooling embeddings ...")
    pool_tr = pool_embeddings(model, ids_tr.astype(np.int64))
    pool_te = pool_embeddings(model, ids_te.astype(np.int64))

    print("[v22 gate] compute_bert_arm ...")
    oof_bert, test_bert = compute_bert_arm(
        pool_tr, pool_te, d["y_train"], d["tt_tr"], d["tt_te"],
        d["train"]["smiles"].values, n_splits=cfg["folds"], seed=cfg["seed"])

    n_tr = len(d["y_train"])
    oof_v22 = np.zeros(n_tr)
    alphas, w_bert, r2_p14, r2_v22 = {}, {}, {}, {}
    for t in TARGETS:
        idx = np.where(d["tt_tr"] == t)[0]
        g_t = d["train"]["smiles"].values[idx]
        M3 = np.column_stack([d["oof_gbm"][idx], d["oof_mt"][idx], oof_bert[idx]])
        oof_v22[idx], a, coefs = blend_narm_oof(M3, d["y_train"][idx], g_t,
                                                n_splits=cfg["folds"])
        alphas[t] = a
        w_bert[t] = float(coefs[2])
        r2_v22[t] = float(np.corrcoef(d["y_train"][idx], oof_v22[idx])[0, 1]) ** 2
        b2 = _p14_2arm_oof(
            np.column_stack([d["oof_gbm"][idx], d["oof_mt"][idx]]),
            d["y_train"][idx], g_t, n_splits=cfg["folds"])
        r2_p14[t] = float(np.corrcoef(d["y_train"][idx], b2)[0, 1]) ** 2

    mean_p14 = float(np.mean(list(r2_p14.values())))
    if abs(mean_p14 - P14_MEAN) > P14_TOL:
        raise AssertionError(
            f"recomputed P14 mean {mean_p14:.4f} deviates from reference "
            f"{P14_MEAN} by > {P14_TOL}; P14 stays final")
    mean_v22 = float(np.mean(list(r2_v22.values())))
    deltas = {t: r2_v22[t] - r2_p14[t] for t in TARGETS}
    eps_delta = float(np.mean([deltas[t] for t in EPS_NC_EI]))
    overall = float(np.mean(list(deltas.values())))
    worst = float(min(deltas.values()))

    trf = pd.DataFrame({"target": d["y_train"],
                        "target_type": d["tt_tr"],
                        "canon": d["train"]["smiles"].values})
    idx_of_target = {t: np.where(d["tt_tr"] == t)[0] for t in TARGETS}
    leak_count = gate_1_leak_audit(
        np.column_stack([d["oof_gbm"], d["oof_mt"], oof_bert]),
        trf, idx_of_target, cfg["folds"])
    bert_only_r2 = compute_bert_only_r2(oof_bert, d["y_train"], d["tt_tr"])
    report = gate_report(r2_p14, r2_v22, leak_count=leak_count,
                         bert_only_r2=bert_only_r2)

    print("\n==" * 34)
    print("target   r2_p14    r2_v22   delta    w_BERT   bert_only_r2")
    for t in TARGETS:
        print(f"{t:6s} {r2_p14[t]:.4f} {r2_v22[t]:.4f} "
              f"{deltas[t]:+.4f}  {w_bert[t]:+.3f}  {bert_only_r2.get(t, 0.0):+.4f}")
    print("-" * 34)
    print(f"mean_v22 {mean_v22:.4f}  mean_p14 {mean_p14:.4f}  mean_delta {overall:+.4f}")
    print(f"eps/nc/ei delta {eps_delta:+.4f}  worst_delta {worst:+.4f}")
    print(f"gate1 leak_count {leak_count}  gate2 soft {report['gate2_soft']}  "
          f"gate3 {report['gate3']}")
    print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}")
    print("==" * 34)

    rows = [{"target": t, "r2_p14": r2_p14[t], "r2_v22": r2_v22[t],
             "delta": deltas[t], "alpha": alphas[t],
             "w_bert": w_bert[t], "bert_only_r2": bert_only_r2.get(t, float("nan"))}
            for t in TARGETS]
    rows.append({"target": "mean", "r2_p14": mean_p14, "r2_v22": mean_v22,
                 "delta": overall, "alpha": float("nan"),
                 "w_bert": float("nan"), "bert_only_r2": float("nan")})
    report_path = out_dir / "v22_gate_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print("wrote", report_path, flush=True)

    submission_path = None
    if report["pass"]:
        test_pred = np.zeros(len(d["test"]))
        for t in TARGETS:
            idx = np.where(d["tt_tr"] == t)[0]
            idx_te = np.where(d["tt_te"] == t)[0]
            M_tr = np.column_stack([d["oof_gbm"][idx], d["oof_mt"][idx],
                                    oof_bert[idx]])
            M_te = np.column_stack([d["test_gbm"][idx_te],
                                    d["test_mt"][idx_te], test_bert[idx_te]])
            from sklearn.linear_model import Ridge
            lr = Ridge(alpha=alphas[t], fit_intercept=True).fit(
                M_tr, d["y_train"][idx])
            test_pred[idx_te] = lr.predict(M_te)
        assert np.isfinite(test_pred).all()
        submission_path = write_submission(
            pd.DataFrame({"id": d["test"]["id"].values, "target": test_pred}),
            str(out_dir / "submission_v22.csv"))
        print("wrote", submission_path, flush=True)

    return {
        "pass": report["pass"], "mean_p14": mean_p14, "mean_v22": mean_v22,
        "mean_delta": overall, "eps_delta": eps_delta, "worst_delta": worst,
        "report": report, "bert_only_r2": bert_only_r2,
        "report_path": str(report_path), "submission_path": submission_path,
    }


if __name__ == "__main__":
    _smoke = os.environ.get("SMOKE", "").strip() == "1"
    run_gate(smoke=_smoke)
