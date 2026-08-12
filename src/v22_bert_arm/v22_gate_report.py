"""Pure gate-evaluation + submission-writer for the v22 gate (gates 0-3)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

EPS_NC_EI = ("eps", "nc", "ei")
SOFT_DELTA = 0.0015
STRONG_DELTA = 0.003
WORST_TOL = 0.003

SUBMISSION_ROWS = 4940


def gate_report(p14, v22, leak_count=0, bert_only_r2=None):
    """Evaluate pre-registered gates 0-3.

    p14 / v22 : dict target -> r2 (P14 2-arm reference / v22 3-arm blend).
    Returns dict {gate0, gate1, gate2_soft, gate2_strong, gate3, pass}.
    """
    if not p14:
        return {"gate0": {}, "gate1": int(leak_count),
                "gate2_soft": False, "gate2_strong": False,
                "gate3": False, "pass": False}
    deltas = {t: round(float(v22[t] - p14[t]), 12) for t in p14}
    eps = [t for t in EPS_NC_EI if t in deltas]
    eps_mean = round(float(np.mean([deltas[t] for t in eps])) if eps else 0.0, 12)
    overall = round(float(np.mean(list(deltas.values()))), 12)
    worst = round(float(min(deltas.values())), 12)
    gate2_soft = eps_mean >= SOFT_DELTA and overall >= SOFT_DELTA
    gate2_strong = eps_mean >= STRONG_DELTA and overall >= STRONG_DELTA
    gate3 = worst >= -WORST_TOL
    return {
        "gate0": dict(bert_only_r2) if bert_only_r2 else {},
        "gate1": int(leak_count),
        "gate2_soft": bool(gate2_soft),
        "gate2_strong": bool(gate2_strong),
        "gate3": bool(gate3),
        "pass": bool(leak_count == 0 and gate2_soft and gate3),
    }


def write_submission(df, path):
    """Write the v22 submission in P14 format (id,target; 4940 rows)."""
    if list(df.columns) != ["id", "target"]:
        raise ValueError(
            f"submission frame must have columns ['id','target'], got "
            f"{list(df.columns)}")
    if len(df) != SUBMISSION_ROWS:
        raise ValueError(
            f"submission must have exactly {SUBMISSION_ROWS} rows, got {len(df)}")
    df.to_csv(path, index=False)
    return path


def gate_1_leak_audit(feat_matrix, trf, idx_of_target, folds):
    """v19-style leak audit over the blend input columns.

    Counts rows where any arm feature exactly equals a true other-target
    label of the same polymer (canonical smiles group). Must be 0.
    """
    Y = trf["target"].values
    T = trf["target_type"].values
    G = trf["canon"].values
    n = len(Y)
    F = np.asarray(feat_matrix, dtype=float)
    gkf = GroupKFold(n_splits=folds)
    row_fold = np.zeros(n, dtype=int)
    for f, (_, va) in enumerate(gkf.split(np.zeros((n, 1)), Y, G)):
        row_fold[va] = f
    polymer_rows = {}
    for i in range(n):
        polymer_rows.setdefault(G[i], []).append(i)
    matches = 0
    for f in range(folds):
        for i in np.where(row_fold == f)[0]:
            ti = T[i]
            others = [j for j in polymer_rows[G[i]] if T[j] != ti]
            if not others:
                continue
            labels = set(float(Y[j]) for j in others)
            if any(float(v) in labels for v in F[i]):
                matches += 1
    return int(matches)
