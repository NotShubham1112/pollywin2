#!/usr/bin/env python
"""Offline gate evaluator for v16 (cross-target decoder) vs v14 (P14).

Consumes the kernel's `blend_oof_test16.npz` (4 OOF arms + 4 test arms +
y_all/g_all/t_all, all row-aligned). Recomputes, per target, BOTH the v14
2-arm blend (Ridge[GBM, MT]) and the v16 4-arm blend (Ridge[GBM, MT, phys,
learn]) with the exact P14 protocol (per-target GroupKFold(5) on canonical
smiles, alpha grid [0.1..25], fillna arm -> target mean), then measures the
OOF gain on the arms-covered (multi-labeled) subset only.

Pre-registered gate (see design doc §3):
  gate_pass = small-five weighted-mean gain >= +0.003
              AND no target regresses > -0.003
  small-five = eps, nc, ei, eea, egb
"""
import numpy as np, pandas as pd, os
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
SMALL_FIVE = ["eps", "nc", "ei", "eea", "egb"]
ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
GATE_GAIN = 0.003
GATE_REGRESS = -0.003
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v16_gates_report.csv")


def _decode(a):
    if a.dtype.kind == "S":
        return np.char.decode(a)
    if a.dtype.kind == "O":
        return np.array([str(x) for x in a])
    return a.astype(str)


def _arm_matrix(idx, oof_g, oof_m, oof_ph, oof_le, tmean):
    fill = lambda v, fb: np.where(np.isfinite(v), v, fb)
    return np.column_stack([oof_g[idx], oof_m[idx],
                            fill(oof_ph[idx], tmean),
                            fill(oof_le[idx], tmean)])


def _oof_blend(M, y, groups, alphas=ALPHA_GRID, n_splits=5):
    """Per-target folded Ridge over an alpha grid; returns best-alpha OOF."""
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, groups))
    best, besta = -np.inf, alphas[0]
    for a in alphas:
        o = np.zeros(len(y))
        for trk, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[trk], y[trk]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, besta = r, a
    oof = np.zeros(len(y))
    for trk, vk in cv:
        oof[vk] = Ridge(alpha=besta).fit(M[trk], y[trk]).predict(M[vk])
    return oof


def evaluate_gates(npz_path, n_splits=5, report_path=None):
    report_path = report_path or REPORT
    d = np.load(npz_path, allow_pickle=True)
    oof_g = np.asarray(d["oof_gbm"], dtype=np.float64)
    oof_m = np.asarray(d["oof_mt"], dtype=np.float64)
    oof_ph = np.asarray(d["oof_phys"], dtype=np.float64)
    oof_le = np.asarray(d["oof_learn"], dtype=np.float64)
    y = np.asarray(d["y_all"], dtype=np.float64)
    g = _decode(d["g_all"])
    t = _decode(d["t_all"])
    tmean = {tg: float(np.nanmean(y[t == tg])) for tg in TARGETS}
    stored = np.asarray(d["blends_tr"], dtype=np.float64) if "blends_tr" in d.files else None

    rows = []
    covered_total = 0
    for tg in TARGETS:
        idx = np.where(t == tg)[0]
        if len(idx) < 5:
            continue
        cover = np.isfinite(oof_ph[idx]) | np.isfinite(oof_le[idx])
        ncov = int(cover.sum())
        covered_total += ncov
        if ncov < 5:
            rows.append(dict(target=tg, covered_n=ncov, v14_blend=np.nan,
                             v16_blend=np.nan, delta=np.nan, per_target_pass=False))
            continue
        Mt2 = np.column_stack([oof_g[idx], oof_m[idx]])
        Mt4 = _arm_matrix(idx, oof_g, oof_m, oof_ph, oof_le, tmean[tg])
        o14 = _oof_blend(Mt2, y[idx], g[idx], n_splits=n_splits)
        o16 = _oof_blend(Mt4, y[idx], g[idx], n_splits=n_splits)
        r14 = r2_score(y[idx][cover], o14[cover])
        r16 = r2_score(y[idx][cover], o16[cover])
        delta = r16 - r14
        rows.append(dict(target=tg, covered_n=ncov, v14_blend=r14,
                         v16_blend=r16, delta=delta,
                         per_target_pass=bool(delta >= GATE_REGRESS)))

    df = pd.DataFrame(rows).set_index("target")
    sf = df.loc[df.index.intersection(SMALL_FIVE)]
    sf = sf[sf["delta"].notna()]
    if len(sf) == 0 or sf["covered_n"].sum() == 0:
        small_five_gain = np.nan
        gate_pass = False
        message = "no small-five target has an arms-covered subset to evaluate"
    else:
        small_five_gain = float(np.average(sf["delta"], weights=sf["covered_n"]))
        worst = float(df["delta"].min())
        gate_pass = bool(small_five_gain >= GATE_GAIN and worst > GATE_REGRESS)
        message = (f"small-five weighted gain {small_five_gain:+.4f} (need >= {GATE_GAIN}); "
                   f"worst per-target delta {worst:+.4f} (need > {GATE_REGRESS})")

    df.to_csv(report_path)
    return dict(per_target=df, small_five_mean_gain=small_five_gain,
                gate_pass=gate_pass, message=message)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if not p or not os.path.exists(p):
        sys.exit("usage: python vault/compare_v16.py <path-to-blend_oof_test16.npz>")
    res = evaluate_gates(p)
    df = res["per_target"]
    print("=== v16 vs v14 (P14) on arms-covered rows ===")
    print(df.round(4).to_string())
    print(f"\ncovered rows total: {int(df['covered_n'].sum())}")
    print(f"small-five weighted gain: {res['small_five_mean_gain']:+.4f}")
    print("GATE:", "PASS -> submit v16" if res["gate_pass"] else "FAIL -> freeze P14 (0.883)")
    print("  " + res["message"])
    print(f"report written to {REPORT}")