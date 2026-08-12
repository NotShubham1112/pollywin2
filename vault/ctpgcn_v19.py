"""CT-PGCN (Cross-Target Property Graph Completion Network).

Offline, honest, fold-safe engine on top of the cached P14 arms
('superblend_oof.npz', produced by the P14/v14 kernel). Implements the
pre-registered protocol from codex.md:

  * GroupKFold(5) on SMILES groups (identical to P14); all blend/arm fits
    happen fold-safely: OOF arms are fitted only on the fold's train part.
  * Sibling lattice: train SMILES pivot -> per-row other-target facts (this
    mirrors the v17 kernel which matches raw SMILES; no test labels used).
  * Physics arms (from decoder_v16):
        eps = A*nc^2 + B       (A,B fit on train nc/eps pairs)
        egb = C*egc + D        (C,D fit on train egc/egb pairs)
        egc = ei - eea         (identity, used only when both siblings present)
  * Property-completion arm: per-target Ridge over the OTHER 6 sibling values,
    fold-local NaN imputation, alpha merged into a per-target meta-blend.
  * Per-target meta-blend: p14 + a_sib*sib + a_phys*phys, alphas selected by
    nested on each fold's train part and capped at 0.30 (gate 3).
  * Rows with no sibling stay EXACTLY on P14 (binary hard coverage gate).

Pre-registered gates:
    1. small-five (eps,nc,ei,eea,egb) equal-weight mean R2 delta >= +0.003
    2. no target regresses > -0.003
    3. every alpha <= 0.30

If all gates pass -> writes 'vault/ctpgcn_submission_v19.csv' (id,target)
and prints SHIP. Else prints "freeze P14 anchor (LB 0.883)" and writes nothing.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

WORK = r"D:\Parth\ploywin r2"
TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]
GI = {t: i for i, t in enumerate(TARGETS)}
SMALL_FIVE = ["eps", "nc", "ei", "eea", "egb"]

NPZ = os.path.join(WORK, "vault", "pipeline_out_pretrain", "superblend_oof.npz")
TRP = os.path.join(WORK, "official_dataset", "train.csv")
TEP = os.path.join(WORK, "official_dataset", "test.csv")
OUT = os.path.join(WORK, "vault", "ctpgcn_submission_v19.csv")

FOLDS = 5
ALPHA_SIB = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_PHYS = [0.0, 0.10, 0.20, 0.30]
CAP = 0.30
GATE_GAIN = 0.003
GATE_REGRESS = -0.003


def _decode(a):
    if a.dtype.kind == "S":
        return np.char.decode(a)
    if a.dtype.kind == "O":
        return np.array([str(x) for x in a])
    return a.astype(str)


def build_pivot(train_df):
    return train_df.dropna(subset=["target"]).pivot_table(
        index="smiles", columns="target_type", values="target", aggfunc="first")


def sib_matrix(pivot, smiles):
    out = np.full((len(smiles), 7), np.nan, dtype=np.float64)
    if pivot is None or len(pivot) == 0:
        return out
    for i, sm in enumerate(smiles):
        if sm in pivot.index:
            row = pivot.loc[sm]
            for j, t in enumerate(TARGETS):
                if t in row.index and pd.notna(row[t]):
                    out[i, j] = row[t]
    return out


def oof_sib_ridge(X, y, groups):
    o = np.zeros(len(y))
    if len(y) < 2:
        return o
    n_g = len(np.unique(groups))
    n_s = min(FOLDS, max(2, n_g))
    cv = list(GroupKFold(n_splits=n_s).split(X, y, groups))
    for tri, vai in cv:
        Xf = X[tri].copy()
        cm = np.nanmean(Xf, axis=0)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        Xf = np.where(np.isfinite(Xf), Xf, cm)
        Xv = X[vai].copy()
        o[vai] = Ridge(alpha=1.0).fit(Xf, y[tri]).predict(
            np.where(np.isfinite(Xv), Xv, cm))
    return o


def pick_alpha(y, p14v, armv, alpha_grid):
    best_a, best_r2 = 0.0, -np.inf
    for a in alpha_grid:
        est = p14v.copy()
        ok = np.isfinite(armv)
        est[ok] = (1 - a) * est[ok] + a * armv[ok]
        r = r2_score(y, est)
        if r > best_r2:
            best_r2, best_a = r, a
    return float(best_a)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("loading cached P14 arms ...", flush=True)
    npz = np.load(NPZ, allow_pickle=True)
    oof_gbm = np.asarray(npz["oof_gbm"], dtype=float)
    oof_mt = np.asarray(npz["oof_mt"], dtype=float)
    test_gbm = np.asarray(npz["test_gbm"], dtype=float)
    test_mt = np.asarray(npz["test_mt"], dtype=float)
    y_tr = np.asarray(npz["y_train"], dtype=float)
    tt_tr = _decode(npz["target_type_train"])
    tt_te = _decode(npz["target_type_test"])

    tr = pd.read_csv(TRP)
    te = pd.read_csv(TEP)
    assert len(tr) == len(y_tr), (len(tr), len(y_tr))
    assert np.array_equal(tr["target_type"].values, tt_tr)

    pivot = build_pivot(tr)
    sib_tr = sib_matrix(pivot, tr["smiles"].values)
    sib_te = sib_matrix(pivot, te["smiles"].values)
    groups_tr = tr["smiles"].values

    # ---------- P14 baseline: per-target Ridge over [gbm, mt] ----------
    print("\n== P14 per-target Ridge (alpha grid) ==")
    p14_oof = np.zeros(len(tr))
    p14_test = np.zeros(len(te))
    p14_alphas = {}
    P14_ALPHA = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
    for tt in TARGETS:
        idx = np.where(tt_tr == tt)[0]
        M = np.column_stack([oof_gbm[idx], oof_mt[idx]])
        yv = y_tr[idx]
        gv = groups_tr[idx]
        n_g = len(np.unique(gv))
        n_s = min(FOLDS, max(2, n_g))
        cv = list(GroupKFold(n_splits=n_s).split(M, yv, gv))
        best, besta = -np.inf, 1.0
        for a in P14_ALPHA:
            o = np.zeros(len(idx))
            for tri, vai in cv:
                o[vai] = Ridge(alpha=a).fit(M[tri], yv[tri]).predict(M[vai])
            r = r2_score(yv, o)
            if r > best:
                best, besta = r, a
        p14_alphas[tt] = besta
        oof = np.zeros(len(idx))
        for tri, vai in cv:
            oof[vai] = Ridge(alpha=besta).fit(M[tri], yv[tri]).predict(M[vai])
        p14_oof[idx] = oof
        _idx_te = np.where(tt_te == tt)[0]
        if len(_idx_te):
            Mte = np.column_stack([test_gbm[_idx_te], test_mt[_idx_te]])
            p14_test[_idx_te] = Ridge(alpha=besta).fit(M, yv).predict(Mte)
        print(f"  {tt}: alpha={besta:5.2f}  P14 R2={r2_score(yv, oof):.4f}  (n={len(idx)})")

    base_mean = float(np.mean([r2_score(y_tr[tt_tr == tt], p14_oof[tt_tr == tt])
                               for tt in TARGETS]))
    print(f"\nP14 mean R2 (equal-weight): {base_mean:.4f}")

    # ---------- Physics laws (fit on train pairs) ----------
    print("\n== physics ==", flush=True)
    C, D = 1.0, 0.0
    ok = np.isfinite(sib_tr[:, GI["egc"]]) & np.isfinite(sib_tr[:, GI["egb"]])
    if int(ok.sum()) >= 3:
        C, D = np.polyfit(sib_tr[ok, GI["egc"]], sib_tr[ok, GI["egb"]], 1)
    A2, B2 = 1.0, 0.0
    ok = np.isfinite(sib_tr[:, GI["nc"]]) & np.isfinite(sib_tr[:, GI["eps"]])
    if int(ok.sum()) >= 3:
        A2, B2 = np.polyfit(sib_tr[ok, GI["nc"]] ** 2, sib_tr[ok, GI["eps"]], 1)
    n_pair_egb = int(ok.sum())
    print(f"  egb = {C:.3f}*egc + {D:.3f}   (n_pairs={n_pair_egb})")
    print(f"  eps = {A2:.4f}*nc^2 + {B2:.4f}   (n_pairs={int(ok.sum())})")

    def phys_col(sc, kind):
        v = np.full(len(sc), np.nan, dtype=np.float64)
        if kind == "egb":
            src = sc[:, GI["egc"]]
            m = np.isfinite(src)
            v[m] = C * src[m] + D
        elif kind == "eps":
            src = sc[:, GI["nc"]]
            m = np.isfinite(src)
            v[m] = A2 * src[m] ** 2 + B2
        elif kind == "egc":
            ei = sc[:, GI["ei"]]
            ea = sc[:, GI["eea"]]
            m = np.isfinite(ei) & np.isfinite(ea)
            v[m] = ei[m] - ea[m]
        return v

    phys_tr = {k: phys_col(sib_tr, k) for k in ("egb", "eps", "egc")}
    phys_te = {k: phys_col(sib_te, k) for k in ("egb", "eps", "egc")}

    # ---------- Final gated per-target meta blend ----------
    print("\n== gated meta blend (alphas capped <= 0.30) ==")
    final_oof = p14_oof.copy()
    final_test = p14_test.copy()
    report_rows = []
    for tt in TARGETS:
        idx = np.where(tt_tr == tt)[0]
        j = GI[tt]
        keep = [k for k in range(7) if k != j]
        X = sib_tr[idx][:, keep]
        yv = y_tr[idx]
        gv = groups_tr[idx]
        n_g = len(np.unique(gv))
        n_s = min(FOLDS, max(2, n_g))
        cv = list(GroupKFold(n_splits=n_s).split(X, yv, gv))
        oar = oof_sib_ridge(X, yv, gv)
        par = np.full(len(idx), np.nan, dtype=np.float64)
        if tt in phys_tr:
            par = phys_tr[tt][idx]

        res_oof = np.zeros(len(idx))
        a_sibs, a_physs = [], []
        sib_cov = 0
        for tri, vai in cv:
            a_s = pick_alpha(yv[tri], p14_oof[idx][tri], oar[tri], ALPHA_SIB)
            a_p = pick_alpha(yv[tri], p14_oof[idx][tri], par[tri], ALPHA_PHYS)
            a_sibs.append(a_s)
            a_physs.append(a_p)
            v = p14_oof[idx][vai].copy()
            ok_s = np.isfinite(oar[vai])
            sib_cov += int(ok_s.sum())
            v[ok_s] = (1 - a_s) * v[ok_s] + a_s * oar[vai][ok_s]
            ok_p = np.isfinite(par[vai])
            v[ok_p] = (1 - a_p) * v[ok_p] + a_p * par[vai][ok_p]
            res_oof[vai] = v
        final_oof[idx] = res_oof

        a_s = float(np.mean(a_sibs))
        a_p = float(np.mean(a_physs))
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        Xf = np.where(np.isfinite(X), X, cm)
        mdl = Ridge(alpha=1.0).fit(Xf, yv)
        idx_te = np.where(tt_te == tt)[0]
        f = p14_test[idx_te].copy()
        if len(idx_te):
            Xte = sib_te[idx_te][:, keep]
            okte = np.isfinite(Xte).sum(1) >= 1
            if okte.sum():
                sib_te_pred = mdl.predict(np.where(np.isfinite(Xte[okte]), Xte[okte], cm))
                f[okte] = (1 - a_s) * f[okte] + a_s * sib_te_pred
            if tt in phys_te:
                pte = phys_te[tt][idx_te]
                m2 = np.isfinite(pte)
                f[m2] = (1 - a_p) * f[m2] + a_p * pte[m2]
        final_test[idx_te] = f

        r_old = r2_score(yv, p14_oof[idx])
        r_new = r2_score(yv, final_oof[idx])
        report_rows.append((tt, r_old, r_new, r_new - r_old, a_s, a_p, sib_cov))
        print(f"  {tt}: a_sib={a_s:.3f} a_phys={a_p:.2f}  "
              f"P14={r_old:.4f}  CT={r_new:.4f}  d={r_new - r_old:+.4f}")

    df = pd.DataFrame(report_rows,
                      columns=["target", "p14", "ct", "d", "a_sib", "a_phys", "sib_cov"])
    new_mean = float(np.mean(df["ct"]))
    deltas = df["ct"] - df["p14"]
    sf = df[df["target"].isin(SMALL_FIVE)]
    sf_w = float(np.mean(sf["ct"] - sf["p14"]))
    worst = float(deltas.min())
    alpha_max = float(max(df["a_sib"].max(), df["a_phys"].max()))

    print("\n=== GATE REPORT ===")
    print(df.round(4).to_string(index=False))
    print(f"\nP14 mean          = {base_mean:.4f}")
    print(f"CT mean           = {df['ct'].mean():.4f}")
    print(f"mean delta        = {df['ct'].mean() - base_mean:+.4f}")
    print(f"small-five mean d = {sf_w:+.4f} (need >= {GATE_GAIN})")
    print(f"worst delta       = {worst:+.4f} (need > {GATE_REGRESS})")
    print(f"max alpha         = {alpha_max:.2f} (need <= {CAP})")

    gate_ok = (sf_w >= GATE_GAIN) and (worst > GATE_REGRESS) and (alpha_max <= CAP)
    if gate_ok:
        sub = pd.DataFrame({"id": te["id"].values, "target": final_test})
        sub.to_csv(OUT, index=False)
        print(f"\nGATE PASS -> wrote {OUT} (n={len(sub)}) -> ship to Kaggle")
    else:
        print("\nGATE FAIL -> freeze P14 anchor (LB 0.883); no submission emitted")


if __name__ == "__main__":
    main()