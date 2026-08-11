"""v21 local gate harness: leak-safe Ridge sibling arm vs frozen P14.

Self-contained, CPU-only, no GNN / no pretrain retraining. It
  1. recomputes the twin source (`twin_scores` / `lgb_test_te`) from the
     feature pickles with a verbatim mirror of `mt_gnn_v2.leak_safe_oof_scores()`,
  2. builds the SIB arm (per-target Ridge over other-target twin OOF + miss
     flags),
  3. blends the three arms (gbm, mt, sib) per target with the P14 fold-safe
     alpha sweep,
  4. runs gates 0-3 against the frozen P14 baseline and prints + writes the
     report to vault/pipeline_out_v21/v21_gate_report.csv.

Usage:
    python vault/r2_sibling_validate.py      # FULL config (5 folds, 800 trees)
    SMOKE=1 python vault/r2_sibling_validate.py   # SMOKE (2 folds, 200 trees)

Run with `python -X utf8` on Windows for the delta/Unicode gate output.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

WORK = Path(__file__).resolve().parents[1]
TRAIN_PKL = WORK / "r2_train_feat.pkl"
TEST_PKL = WORK / "r2_test_feat.pkl"
OOF_PATH = WORK / "vault" / "pipeline_out_pretrain" / "superblend_oof.npz"
OUT_DIR = WORK / "vault" / "pipeline_out_v21"
REPORT_CSV = OUT_DIR / "v21_gate_report.csv"

SEED = 42
EARLY_HOLDOUT = 0.15
GLOBAL_FOLDS = 2 if os.environ.get("SMOKE", "0") == "1" else 5
N_ESTIMATORS = 200 if GLOBAL_FOLDS == 2 else 800

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]  # sorted, as in mt3
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
ALPHA_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]

# Pre-registered gate thresholds (do NOT soften).
SOFT_DELTA = 0.0015
STRONG_DELTA = 0.003
WORST_TOL = 0.003
EPS_NC_EI = ("eps", "nc", "ei")

# P14 honest-OOF reference (mean per-target corr^2), see run_v20_gate.py.
P14_MEAN = 0.8641
P14_TOL = 0.005

F32_MAX = np.finfo(np.float32).max


def corr2(y, o):
    """Per-target R^2 reported for the blend comparison (corr^2, as v20 gate)."""
    return float(np.corrcoef(y, o)[0, 1]) ** 2


# ---------------------------------------------------------------------------
# Feature loading (mirror mt_gnn_v2.py:40-108 verbatim).
# ---------------------------------------------------------------------------
def add_fingerprints(df):
    morgan = np.zeros((len(df), 2048), dtype=np.float32)
    maccs = np.zeros((len(df), 167), dtype=np.float32)
    ap = np.zeros((len(df), 1024), dtype=np.float32)
    tt = np.zeros((len(df), 1024), dtype=np.float32)
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    for i, s in enumerate(df['smiles']):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        morgan[i] = np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(
            m, 2, nBits=2048).ToBitString().encode(), 'u1') - ord('0')
        maccs[i] = np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(),
                                 'u1') - ord('0')
        ap[i] = np.frombuffer(ap_gen.GetFingerprint(m).ToBitString().encode(),
                              'u1') - ord('0')
        tt[i] = np.frombuffer(tt_gen.GetFingerprint(m).ToBitString().encode(),
                              'u1') - ord('0')
    return morgan, maccs, ap, tt


def _impute_median(D):
    D = np.clip(D, -F32_MAX, F32_MAX)
    for j in range(D.shape[1]):
        col = D[:, j]
        med = np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
    return D


def load_features(train_path, test_path):
    """Recompute descriptors + fingerprints + scaled matrices (mirror of
    mt_gnn_v2.py:40-108). Scaler is fit on TRAIN only; identical to the
    notebook's two `StandardScaler().fit(X)` calls because the fit is the
    same deterministic transform on the same X."""
    trf = pd.read_pickle(train_path)
    tef = pd.read_pickle(test_path)
    feat_cols = [c for c in trf.columns if c not in
                 ('smiles', 'target', 'target_type', 'canon', 'inchikey', 'id')]

    D_tr = _impute_median(trf[feat_cols].values)
    D_te = _impute_median(tef[feat_cols].values)
    mor_tr, mc_tr, ap_tr, tt_tr = add_fingerprints(trf)
    mor_te, mc_te, ap_te, tt_te = add_fingerprints(tef)

    Y = trf['target'].values.astype(np.float32)
    T = trf['target_type'].values
    G = np.asarray(trf['canon'].values)
    T_te = tef['target_type'].values

    X = np.hstack([D_tr, mor_tr, mc_tr, ap_tr, tt_tr]).astype(np.float32)
    Xte = np.hstack([D_te, mor_te, mc_te, ap_te, tt_te]).astype(np.float32)
    scaler = StandardScaler().fit(X)  # fit on TRAIN only
    Xs = scaler.transform(X).astype(np.float32)
    Xtes = scaler.transform(Xte).astype(np.float32)
    return trf, tef, Xs, Xtes, Y, T, G, T_te


# ---------------------------------------------------------------------------
# Twin source (verbatim mirror of mt_gnn_v2.leak_safe_oof_scores, 285-312).
# ---------------------------------------------------------------------------
def recompute_twin(Xs, Xtes, Y, G, idx_of_target, TARGET_IDX,
                   GLOBAL_FOLDS, EARLY_HOLDOUT, SEED, n_estimators):
    """Per-target LGBM OOF across all train rows (leak-safe by GroupKFold on
    canonical smiles) + fold-bagged test predictions."""
    n_tr, n_te = len(Xs), len(Xtes)
    scores = np.full((n_tr, len(TARGET_IDX)), np.nan, dtype=np.float32)
    lgb_test_te = np.zeros((n_te, len(TARGET_IDX)), dtype=np.float32)

    # canon -> group id per row, using one global fold assignment
    gkf = GroupKFold(n_splits=GLOBAL_FOLDS)
    row_fold = np.zeros(n_tr, dtype=int)
    for f, (_, va) in enumerate(gkf.split(Xs, Y, G)):
        row_fold[va] = f

    for u in TARGET_IDX:
        for f in range(GLOBAL_FOLDS):
            in_fold = np.where(row_fold == f)[0]
            out_fold = np.setdiff1d(np.arange(n_tr), in_fold)
            idx_u_out = np.intersect1d(out_fold, idx_of_target[u])
            if len(idx_u_out) == 0:
                continue
            fit_ids, ho_ids = train_test_split(idx_u_out,
                                               test_size=EARLY_HOLDOUT,
                                               random_state=SEED)
            m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05,
                                  num_leaves=15, min_child_samples=10,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=SEED, verbose=-1)
            # lightgbm's sklearn wrapper sets feature_names_in_ on numpy fit and
            # sklearn then warns on every predict; data are plain numpy arrays
            # (mirror of mt_gnn_v2), so this false positive is suppressed here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                m.fit(Xs[fit_ids], Y[fit_ids],
                      eval_set=[(Xs[ho_ids], Y[ho_ids])])
                scores[in_fold, TARGET_IDX[u]] = m.predict(Xs[in_fold])
                # test bag
                lgb_test_te[:, TARGET_IDX[u]] += m.predict(Xtes) / GLOBAL_FOLDS
    return scores, lgb_test_te


# ---------------------------------------------------------------------------
# SIB arm (pure, unit-tested).
# ---------------------------------------------------------------------------
def build_feats(twin, cols, targets, target_mean):
    """12-column feature block per target: for each sibling target u != t,
    [twin[:,u] (NaN -> TARGET_MEAN[u]), miss-flag].  No self column."""
    twin = np.asarray(twin)
    out = np.zeros((len(twin), 2 * len(cols)), dtype=np.float64)
    for j, u in enumerate(cols):
        v = twin[:, targets.index(u)]
        miss = np.isnan(v).astype(np.float64)
        v = np.where(miss, target_mean[u], v)
        out[:, 2 * j] = v
        out[:, 2 * j + 1] = miss
    return out


def ridge_oof(Xtr, Xte, yt, cv, alpha_grid):
    """Fold-safe per-target Ridge: alpha tuned by inner OOF r2 over the grid,
    refit on the full (train) OOF at the best alpha for the test predictions.

    Returns (oof, test_pred, best_alpha); oof is aligned to input row order.
    """
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    yt = np.asarray(yt, dtype=float)
    n = len(yt)
    if n < 2:
        lr = Ridge(alpha=alpha_grid[0]).fit(Xtr, yt)
        return lr.predict(Xtr), lr.predict(Xte), float(alpha_grid[0])
    best, besta, oof = -np.inf, alpha_grid[0], np.zeros(n)
    for a in alpha_grid:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(Xtr[tr], yt[tr]).predict(Xtr[vk])
        r = r2_score(yt, o)
        if r > best:
            best, besta, oof = r, a, o.copy()
    lr = Ridge(alpha=besta).fit(Xtr, yt)
    return oof, lr.predict(Xte), float(besta)


def build_sib_arm(twin_scores, lgb_test_te, targets, target_mean, Y, T, G, T_te,
                  idx_of_target, GLOBAL_FOLDS, alpha_grid):
    """SIB arm: per-target Ridge over the other-target twin features.

    Returns (sib_oof, sib_test, sib_only_r2): sib_oof indexed at the original
    train row order, sib_test indexed at the original test row order, and the
    per-target OOF r2 of the sibling Ridge alone (gate 0 diagnostic).
    """
    sib_oof = np.full(len(Y), np.nan)
    sib_test = np.zeros(len(lgb_test_te))
    sib_only_r2 = {}
    for t in targets:
        idx = idx_of_target[t]
        idx_te = np.where(T_te == t)[0]
        cols = [u for u in targets if u != t]
        Xtr = build_feats(twin_scores[idx], cols, targets, target_mean)
        Xte = build_feats(lgb_test_te, cols, targets, target_mean)
        yt = Y[idx].astype(np.float64)
        if len(np.unique(G[idx])) < GLOBAL_FOLDS:
            # too few smiles groups for GroupKFold -> plain Ridge on all rows
            lr = Ridge(alpha=alpha_grid[0]).fit(Xtr, yt)
            sib_oof[idx] = lr.predict(Xtr)
            sib_test[idx_te] = lr.predict(Xte)[idx_te]
            sib_only_r2[t] = float(r2_score(yt, sib_oof[idx]))
            continue
        cv = list(GroupKFold(n_splits=GLOBAL_FOLDS).split(Xtr, yt, G[idx]))
        oof_t, te_t, a_best = ridge_oof(Xtr, Xte, yt, cv, alpha_grid)
        sib_oof[idx] = oof_t
        sib_test[idx_te] = te_t[idx_te]
        sib_only_r2[t] = float(r2_score(yt, oof_t))
    return sib_oof, sib_test, sib_only_r2


# ---------------------------------------------------------------------------
# P14 2-arm reference (verbatim copy of run_v20_gate._p14_2arm_oof, 96-125).
# ---------------------------------------------------------------------------
def _p14_2arm_oof(M2, y, g, n_splits=5):
    """Fold-safe 2-arm OOF alpha scan (P14 baseline protocol).

    Same GroupKFold-on-g + per-alpha OOF r2 selection + refit-at-best as the
    3-arm blend, but on exactly the two P14 arms (gbm, mt).
    """
    M2 = np.asarray(M2, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    if n < 2:
        return y.copy()
    M = np.where(np.isfinite(M2), M2, np.nanmean(M2, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        return Ridge(alpha=ALPHA_GRID[0]).fit(M, y).predict(M)
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, out = -np.inf, np.zeros(n)
    for a in ALPHA_GRID:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, out = r, o.copy()
    return out


def blend_3arm_oof(M3, y, g, n_splits=5):
    """3-arm blend (gbm, mt, sib) with the same fold-safe alpha scan.

    Returns (oof, best_alpha, coefs_mean); w_SIB = coefs_mean[2].
    """
    M3 = np.asarray(M3, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g)
    n = len(y)
    if n < 2:
        return y.copy(), float(ALPHA_GRID[0]), np.zeros(3)
    M = np.where(np.isfinite(M3), M3, np.nanmean(M3, axis=0))
    M = np.where(np.isfinite(M), M, 0.0)
    if len(np.unique(g)) < n_splits:
        lr = Ridge(alpha=ALPHA_GRID[0]).fit(M, y)
        return lr.predict(M), float(ALPHA_GRID[0]), lr.coef_
    cv = list(GroupKFold(n_splits=n_splits).split(M, y, g))
    best, besta = -np.inf, ALPHA_GRID[0]
    for a in ALPHA_GRID:
        o = np.zeros(n)
        for tr, vk in cv:
            o[vk] = Ridge(alpha=a).fit(M[tr], y[tr]).predict(M[vk])
        r = r2_score(y, o)
        if r > best:
            best, besta = r, a
    oof = np.zeros(n)
    coefs = []
    for tr, vk in cv:
        lr = Ridge(alpha=besta).fit(M[tr], y[tr])
        oof[vk] = lr.predict(M[vk])
        coefs.append(lr.coef_)
    return oof, float(besta), np.mean(coefs, axis=0)


def blend_driver(oof_gbm, oof_mt, sib_oof, y, tt_tr, g_smiles, targets,
                 n_splits=5):
    """Per-target P14 2-arm reference + v21 3-arm blend.

    Returns (r2_p14, r2_v21, alphas, w_sib) dicts keyed by target (corr^2).
    """
    r2_p14, r2_v21, alphas, w_sib = {}, {}, {}, {}
    for t in targets:
        idx = np.where(tt_tr == t)[0]
        gt = g_smiles[idx]
        b2 = _p14_2arm_oof(np.column_stack([oof_gbm[idx], oof_mt[idx]]),
                           y[idx], gt, n_splits=n_splits)
        r2_p14[t] = corr2(y[idx], b2)
        M3 = np.column_stack([oof_gbm[idx], oof_mt[idx], sib_oof[idx]])
        oof3, a3, coefs = blend_3arm_oof(M3, y[idx], gt, n_splits=n_splits)
        r2_v21[t] = corr2(y[idx], oof3)
        alphas[t] = a3
        w_sib[t] = coefs[2]
    return r2_p14, r2_v21, alphas, w_sib


# ---------------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------------
def gate_1_leak_audit(twin_scores, trf, idx_of_target, folds):
    """v19-style leak audit: count rows where any sibling feature exactly equals
    a true other-target label of that polymer (same canonical smiles group).

    Returns the exact-match count across all val folds; must be 0.
    """
    Y = trf["target"].values
    T = trf["target_type"].values
    G = trf["canon"].values
    targets = list(idx_of_target.keys())
    n = len(Y)

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
            labels = [Y[j] for j in others]
            for iu, u in enumerate(targets):
                if u == ti:
                    continue
                if any(twin_scores[i, iu] == lab for lab in labels):
                    matches += 1
                    break
    return int(matches)


def gate_report(p14_r2, v21_r2, sib_only_r2=None, leak_count=0,
                eps_nc_ei=EPS_NC_EI, soft_delta=SOFT_DELTA,
                strong_delta=STRONG_DELTA, worst_tol=WORST_TOL):
    """Evaluate gates 0-3 against the pre-registered table.

    gate0 (diagnostic): per-target sib_only_r2, report only.
    gate1 (leak audit): leak_count must be 0.
    gate2 (OOF gain): blend mean over {eps,nc,ei} AND overall mean both
        >= p14 + delta; soft delta +0.0015, strong +0.003.
    gate3 (worst-target): every per-target delta >= -0.003.

    Returns dict with keys gate0, gate1, gate2_soft, gate2_strong, gate3,
    pass (plus informative extras).
    """
    deltas = {t: float(v21_r2[t] - p14_r2[t]) for t in p14_r2}
    eps = [t for t in eps_nc_ei if t in deltas]
    eps_mean = float(np.mean([deltas[t] for t in eps])) if eps else 0.0
    overall = float(np.mean(list(deltas.values())))
    worst = float(min(deltas.values()))

    gate2_soft = eps_mean >= soft_delta and overall >= soft_delta
    gate2_strong = eps_mean >= strong_delta and overall >= strong_delta
    gate3 = worst >= -worst_tol
    gate1_ok = leak_count == 0
    passed = bool(gate1_ok and gate2_soft and gate3)
    return {
        "gate0": dict(sib_only_r2) if sib_only_r2 else {},
        "gate1": int(leak_count),
        "gate2_soft": bool(gate2_soft),
        "gate2_strong": bool(gate2_strong),
        "gate3": bool(gate3),
        "pass": passed,
        "eps_nc_ei_delta": eps_mean,
        "overall_delta": overall,
        "worst_delta": worst,
    }


# ---------------------------------------------------------------------------
# main().
# ---------------------------------------------------------------------------
def main():
    smoke = GLOBAL_FOLDS == 2
    print(f"[v21 harness] {'SMOKE' if smoke else 'FULL'} | "
          f"folds={GLOBAL_FOLDS} n_estimators={N_ESTIMATORS}")
    t0 = time.time()

    # 1) features + twin source
    trf, tef, Xs, Xtes, Y, T, G, T_te = load_features(TRAIN_PKL, TEST_PKL)
    idx_of_target = {t: np.where(T == t)[0] for t in TARGETS}
    target_mean = {t: float(Y[idx_of_target[t]].mean()) for t in TARGETS}
    print(f"[v21] features ready: train {Xs.shape} test {Xtes.shape} "
          f"({time.time()-t0:.0f}s)")

    twin_scores, lgb_test_te = recompute_twin(
        Xs, Xtes, Y, G, idx_of_target, TARGET_IDX, GLOBAL_FOLDS,
        EARLY_HOLDOUT, SEED, N_ESTIMATORS)
    # Alignment guard (mirror run_v20_gate.py:75-83): npz rows == pkl rows.
    assert twin_scores.shape[0] == len(trf), \
        "twin_scores rows != train feature rows"
    assert lgb_test_te.shape[0] == len(tef), \
        "lgb_test_te rows != test feature rows"
    print(f"[v21] twin source: {twin_scores.shape} / {lgb_test_te.shape} "
          f"({time.time()-t0:.0f}s)")

    # 2) cached P14 OOF + row alignment
    z = np.load(OOF_PATH, allow_pickle=True)
    tt_tr, tt_te = z["target_type_train"], z["target_type_test"]
    assert np.array_equal(tt_tr, T), \
        "npz target_type_train does not match train feature order"
    assert np.array_equal(tt_te, T_te), \
        "npz target_type_test does not match test feature order"
    assert np.abs(np.corrcoef(z["y_train"], Y)[0, 1]) > 0.999, \
        "npz y_train not aligned with train feature target"
    oof_gbm = z["oof_gbm"].astype(float)
    oof_mt = z["oof_mt"].astype(float)
    y = z["y_train"].astype(float)
    g_blend = np.asarray(trf["smiles"].values)  # P14 blend groups by SMILES

    # 3) SIB arm
    sib_oof, sib_test, sib_only_r2 = build_sib_arm(
        twin_scores, lgb_test_te, TARGETS, target_mean, Y, T, G, T_te,
        idx_of_target, GLOBAL_FOLDS, ALPHA_GRID)
    print(f"[v21] SIB arm: oof {sib_oof.shape} test {sib_test.shape} "
          f"({time.time()-t0:.0f}s)")

    # 4) blend: P14 2-arm reference + v21 3-arm
    r2_p14, r2_v21, alphas, w_sib = blend_driver(
        oof_gbm, oof_mt, sib_oof, y, tt_tr, g_blend, TARGETS, n_splits=5)

    mean_p14 = float(np.mean(list(r2_p14.values())))
    mean_v21 = float(np.mean(list(r2_v21.values())))
    if abs(mean_p14 - P14_MEAN) > P14_TOL:
        raise AssertionError(
            f"recomputed P14 mean {mean_p14:.4f} deviates from reference "
            f"{P14_MEAN} by more than {P14_TOL}; the mean gate would be "
            f"misleading - P14 stays final")

    # 5) gates 0-3
    leak_count = gate_1_leak_audit(twin_scores, trf, idx_of_target, GLOBAL_FOLDS)
    report = gate_report(r2_p14, r2_v21, sib_only_r2, leak_count)

    print()
    print("==" * 36)
    print("target   r2_p14   r2_v21   delta    alpha   w_sib  sib_only_r2")
    for t in TARGETS:
        print(f"{t:6s}   {r2_p14[t]:.4f}   {r2_v21[t]:.4f}   "
              f"{r2_v21[t]-r2_p14[t]:+.4f}   {alphas[t]:.2f}   "
              f"{w_sib[t]:+.3f}   {sib_only_r2[t]:+.4f}")
    print("-" * 36)
    print(f"mean_v21 {mean_v21:.4f}  mean_p14 {mean_p14:.4f}  "
          f"mean_delta {report['overall_delta']:+.4f}")
    print(f"eps/nc/ei mean delta {report['eps_nc_ei_delta']:+.4f}  "
          f"worst_delta {report['worst_delta']:+.4f}")
    print()
    print("gate0 sib_only_r2:", {t: round(v, 4) for t, v in sib_only_r2.items()})
    print(f"gate1 leak audit: {report['gate1']} (must be 0)")
    print(f"gate2 soft  ({EPS_NC_EI} + {SOFT_DELTA} / overall): "
          f"{report['gate2_soft']}")
    print(f"gate2 strong ({EPS_NC_EI} + {STRONG_DELTA} / overall): "
          f"{report['gate2_strong']}")
    print(f"gate3 worst-target >= {-WORST_TOL:+.3f} (delta): "
          f"{report['gate3']}")
    print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}"
          f" -> {'v21 proceeds' if report['pass'] else 'P14 stays final'}")
    print("==" * 36)

    # 6) write report CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in TARGETS:
        rows.append({
            "target": t, "r2_p14": r2_p14[t], "r2_v21": r2_v21[t],
            "delta": r2_v21[t] - r2_p14[t], "alpha": alphas[t], "w_sib": w_sib[t],
            "sib_only_r2": sib_only_r2[t], "gate1": np.nan,
            "gate2_soft": np.nan, "gate2_strong": np.nan, "gate3": np.nan,
            "pass": np.nan})
    rows.append({
        "target": "mean", "r2_p14": mean_p14, "r2_v21": mean_v21,
        "delta": report["overall_delta"], "alpha": np.nan, "w_sib": np.nan,
        "sib_only_r2": np.nan, "gate1": np.nan, "gate2_soft": np.nan,
        "gate2_strong": np.nan, "gate3": np.nan, "pass": np.nan})
    rows.append({
        "target": "VERDICT", "r2_p14": np.nan, "r2_v21": np.nan,
        "delta": report["overall_delta"], "alpha": np.nan, "w_sib": np.nan,
        "sib_only_r2": np.nan, "gate1": report["gate1"],
        "gate2_soft": report["gate2_soft"], "gate2_strong": report["gate2_strong"],
        "gate3": report["gate3"], "pass": report["pass"]})
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)
    print(f"[v21] report written to {REPORT_CSV}")
    print(f"[v21] total wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
