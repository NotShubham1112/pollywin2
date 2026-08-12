"""
MT-GNN v2: PI1M-pretrained shared GINE trunk -> multi-task per-target heads,
1/freq loss weighting + explicit cross-target twin features, leak-safe OOF,
per-target fallback vs the GBM trio stack.

Answers: does {shared multi-task trunk + PI1M init + twin features} beat the
GBM stack on each target (esp. the 220-row heads)?

Leak-safety:
  * GroupKFold on canonical SMILES (a canon's rows all land in one fold).
  * Twin feature for row i / target u = target-u LGBM's OOF prediction on
    row i's features (a model trained WITHOUT row i's fold) for train, and the
    fold-bagged target-u test prediction for test. No label of row i leaks.
  * Per-target z-score on the fit fold only; 1/freq[target] sample weighting.

Usage:
  python mt_gnn_v2.py                     # full run
  SMOKE=1 python mt_gnn_v2.py             # 2 folds, 4 epochs, 500 pretrain cap
"""
import os, sys, time, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, rdFingerprintGenerator
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score

SMOKE = os.environ.get("SMOKE", "0") == "1"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_FOLDS = 2 if SMOKE else 5
MAX_EPOCHS = 4 if SMOKE else 120
PATIENCE = 5 if SMOKE else 20
EARLY_HOLDOUT = 0.15
BS = 256 if not SMOKE else 64
LR = 1e-3

WORK = r"D:\Parth\ploywin r2"
PRETRAINED = os.path.join(WORK, "vault", "kernel-v10-output", "pretrained_encoder.pt")
OUT = os.path.join(WORK, "vault", "pipeline_out_pretrain_smoke" if SMOKE else "pipeline_out_pretrain")
os.makedirs(OUT, exist_ok=True)

TARGETS = ["eea", "egb", "egc", "ei", "eps", "nc", "tg"]  # sorted, as in mt3
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
print("device:", DEVICE, "| SMOKE:", SMOKE, "| folds:", GLOBAL_FOLDS,
      "| max_epochs:", MAX_EPOCHS, "| out:", OUT, flush=True)

trf = pd.read_pickle(os.path.join(WORK, "r2_train_feat.pkl"))
tef = pd.read_pickle(os.path.join(WORK, "r2_test_feat.pkl"))
FEAT_COLS = [c for c in trf.columns if c not in
             ('smiles', 'target', 'target_type', 'canon', 'inchikey', 'id')]
F32_MAX = np.finfo(np.float32).max


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


D_tr = np.clip(trf[FEAT_COLS].values, -F32_MAX, F32_MAX)
for j in range(D_tr.shape[1]):
    col = D_tr[:, j]
    med = np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
    col[~np.isfinite(col)] = med
mor_tr, mc_tr, ap_tr, tt_tr = add_fingerprints(trf)

D_te = np.clip(tef[FEAT_COLS].values, -F32_MAX, F32_MAX)
for j in range(D_te.shape[1]):
    col = D_te[:, j]
    med = np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
    col[~np.isfinite(col)] = med
mor_te, mc_te, ap_te, tt_te = add_fingerprints(tef)

Y = trf['target'].values.astype(np.float32)
T = trf['target_type'].values
G = trf['canon'].values
idx_of_target = {t: np.where(T == t)[0] for t in TARGETS}

X = np.hstack([D_tr, mor_tr, mc_tr, ap_tr, tt_tr]).astype(np.float32)
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit(X).transform(X).astype(np.float32)

Xte = np.hstack([D_te, mor_te, mc_te, ap_te, tt_te]).astype(np.float32)
Xtes = StandardScaler().fit(X).transform(Xte).astype(np.float32)
print("train:", X.shape, "test:", Xte.shape, flush=True)


# =====================================================================
# Graph featurization (MUST match the v10 pretrain kernel so the saved
# pretrained_encoder.pt loads into the same GINEEncoder).
# =====================================================================
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "Si", "P", "OTHER"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def one_hot(value, choices):
    vec = [0.0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec


def atom_features(atom):
    return (one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
            + one_hot(atom.GetHybridization().name, HYBRIDIZATIONS)
            + [atom.GetIsAromatic() * 1.0, atom.IsInRing() * 1.0,
               atom.GetDegree() / 4.0, atom.GetTotalNumHs() / 4.0,
               atom.GetFormalCharge() / 2.0])


N_ATOM_FEATS = len(ATOM_SYMBOLS) + len(HYBRIDIZATIONS) + 5
N_BOND_FEATS = len(BOND_TYPES) + 2


def bond_features(bond):
    return one_hot(bond.GetBondType().name, BOND_TYPES) + [
        bond.GetIsConjugated() * 1.0, bond.IsInRing() * 1.0]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    if len(edge_index) == 0:
        edge_index = [[0, 0]]; edge_attr = [[0.0] * N_BOND_FEATS]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_graphs(df, has_target=True):
    out = {}
    freq = df["target_type"].value_counts(normalize=True)
    for row_id, row in zip(df.index, df.itertuples()):
        g = smiles_to_graph(row.smiles)
        if g is None:
            continue
        g.row_id = row_id
        g.smiles = row.smiles
        if has_target:
            g.target_idx = torch.tensor([TARGET_IDX[row.target_type]], dtype=torch.long)
            g.y = torch.tensor([float(row.target)], dtype=torch.float)
            g.w = torch.tensor([1.0 / freq[row.target_type]], dtype=torch.float)
        out[row_id] = g
    return out


def to_pyg(graphs):
    if isinstance(graphs, dict):
        graphs = list(graphs.values())
    return Batch.from_data_list(graphs)


t0 = time.time()
train_graphs = build_graphs(trf, has_target=True)
test_graphs = build_graphs(tef, has_target=False)
print(f"graphs: {len(train_graphs)} train, {len(test_graphs)} test "
      f"({time.time()-t0:.0f}s)", flush=True)


# =====================================================================
# Shared encoder + multi-task trunk (same GINEEncoder as v10 kernel).
# =====================================================================
class GINEEncoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, hidden=128, n_layers=4, dropout=0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(n_atom_feats, hidden)
        self.bond_encoder = nn.ModuleList(
            [nn.Linear(n_bond_feats, hidden) for _ in range(n_layers)])
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_encoder(x)
        for conv, bn, bond_enc in zip(self.convs, self.bns, self.bond_encoder):
            e = bond_enc(edge_attr)
            h = conv(h, edge_index, e)
            h = bn(h); h = F.relu(h); h = F.dropout(h, p=self.dropout,
                                                    training=self.training)
        return h


class MTGNN(nn.Module):
    """Shared trunk + per-target heads. Optional cross-target twin features
    are concatenated to the pooled embedding before the shared trunk."""

    def __init__(self, n_atom_feats, n_bond_feats, n_twin=0, hidden=128,
                 n_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = GINEEncoder(n_atom_feats, n_bond_feats, hidden,
                                   n_layers, dropout)
        pool_in = hidden * 2 + n_twin
        self.trunk = nn.Sequential(
            nn.Linear(pool_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(64, 1))
            for _ in TARGETS])

    def forward(self, data, twin=None):
        h = self.encoder(data.x, data.edge_index, data.edge_attr)
        pooled = torch.cat([global_mean_pool(h, data.batch),
                            global_add_pool(h, data.batch)], dim=1)
        if twin is not None and twin.size(1) > 0:
            pooled = torch.cat([pooled, twin.to(pooled.device)], dim=1)
        ht = self.trunk(pooled)
        out = torch.empty(data.batch.max() + 1, len(TARGETS), device=h.device)
        for i, head in enumerate(self.heads):
            out[:, i] = head(ht)[:, 0]
        return out

    def load_encoder(self, state_dict):
        enc = {k[len("encoder."):]: v for k, v in state_dict.items()
               if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        print(f"  encoder init: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)


# =====================================================================
# Twin source: per-target LGBM OOF (leak-safe) + fold-bagged test preds.
# twin_u(row i) = target-u LGBM's prediction on row i's features.
# =====================================================================
print("\n=== Twin source: per-target LGBM OOF (leak-safe) ===", flush=True)
lgb_test_te = np.zeros((len(Xte), len(TARGETS)), dtype=np.float32)
TARGET_MEAN = {t: float(Y[idx_of_target[t]].mean()) for t in TARGETS}


# For train, twin_u(row i) uses lgb_oof_all from the target-u LGBM. But
# lgb_oof_all is stored by row index only for target-u rows. For a row of
# target t, its target-u twin value is the OOF prediction of the model_u on
# THAT row's features - we approximate with the per-target model_u evaluated
# on every train row (OOF where available, fold-safe holdout elsewhere).
# Simplest leak-safe approach: evaluate each target-u LGBM on ALL train rows
# via a dedicated OOF-style pass below.
print("\n=== Building leak-safe twin feature matrices ===", flush=True)
twin_train = np.zeros((len(X), (len(TARGETS) - 1) * 2), dtype=np.float32)
twin_test = np.zeros((len(Xte), (len(TARGETS) - 1) * 2), dtype=np.float32)
col_map = {}
for u in TARGETS:
    col = 0
    for t2 in TARGETS:
        if t2 == u:
            continue
        col_map[(u, t2)] = (col, col + 1)
        col += 2
def leak_safe_oof_scores():
    """For each target u, score every train row with a model trained on a
    canon-group that excludes that row (grouped OOF across all targets)."""
    scores = np.full((len(X), len(TARGETS)), np.nan, dtype=np.float32)
    # canon -> group id per row, using one global fold assignment
    gkf = GroupKFold(n_splits=GLOBAL_FOLDS)
    row_fold = np.zeros(len(X), dtype=int)
    for f, (_, va) in enumerate(gkf.split(Xs, Y, G)):
        row_fold[va] = f
    for u in TARGETS:
        for f in range(GLOBAL_FOLDS):
            in_fold = np.where(row_fold == f)[0]
            out_fold = np.setdiff1d(np.arange(len(X)), in_fold)
            idx_u_out = np.intersect1d(out_fold, idx_of_target[u])
            if len(idx_u_out) == 0:
                continue
            fit_ids, ho_ids = train_test_split(idx_u_out,
                                               test_size=EARLY_HOLDOUT,
                                               random_state=SEED)
            m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05,
                                  num_leaves=15, min_child_samples=10,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=SEED, verbose=-1)
            m.fit(Xs[fit_ids], Y[fit_ids], eval_set=[(Xs[ho_ids], Y[ho_ids])])
            scores[in_fold, TARGET_IDX[u]] = m.predict(Xs[in_fold])
            # test bag
            lgb_test_te[:, TARGET_IDX[u]] += m.predict(Xtes) / GLOBAL_FOLDS
    return scores, lgb_test_te


twin_scores, lgb_test_te = leak_safe_oof_scores()
for t in TARGETS:
    for u in TARGETS:
        if u == t:
            continue
        iu = TARGET_IDX[u]
        c0, c1 = col_map[(t, u)]
        impute = TARGET_MEAN[u]
        v = twin_scores[:, iu]
        miss = np.isnan(v).astype(np.float32)
        v = np.where(miss, impute, v)
        twin_train[:, c0] = v; twin_train[:, c1] = miss
        # test: fold-bagged model_u prediction, always available
        tv = lgb_test_te[:, iu]
        tmiss = np.isnan(tv).astype(np.float32)
        tv = np.where(tmiss, impute, tv)
        twin_test[:, c0] = tv; twin_test[:, c1] = tmiss
print("twin matrices:", twin_train.shape, twin_test.shape, flush=True)


# =====================================================================
# MT-GNN fold-safe OOF + test bag
# =====================================================================
def early_split(fit_ids):
    uniq_g = np.unique(G[fit_ids])
    uniq_f, uniq_h = train_test_split(uniq_g, test_size=EARLY_HOLDOUT,
                                      random_state=SEED)
    return (fit_ids[np.isin(G[fit_ids], uniq_f)],
            fit_ids[np.isin(G[fit_ids], uniq_h)])


row_to_graph = {g.row_id: g for g in train_graphs.values()}
print("\n=== MT-GNN v2 (pretrained-init trunk + twins) ===", flush=True)
pretrained_state = torch.load(PRETRAINED, map_location="cpu") if os.path.exists(
    PRETRAINED) else None
if pretrained_state is not None:
    print("loaded pretrained_encoder.pt", flush=True)

GNN_SEEDS = [int(s) for s in os.environ.get("GNN_SEEDS", "42").split(",") if s.strip()]


def run_gnn_seed(seed):
    """One seed's MT-GNN: fold-safe GroupKFold OOF + fold-bagged test preds.
    Returns (mt_oof_all, mt_test) in raw scale. Identical math to the v13 run
    except torch/np/random seeding are reset per seed."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n_twin = twin_train.shape[1]
    mt_oof_all = np.full(len(X), np.nan, dtype=np.float32)
    mt_test_folds = np.zeros((len(Xte), GLOBAL_FOLDS), dtype=np.float32)
    for f, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=GLOBAL_FOLDS).split(
            Xs, Y, G)):
        t0f = time.time()
        stats = {}
        y_norm = np.empty(len(tr_idx), dtype=np.float32)
        for t in TARGETS:
            mask = (T[tr_idx] == t)
            if mask.sum() > 0:
                mu, sd = Y[tr_idx][mask].mean(), Y[tr_idx][mask].std() + 1e-6
                stats[t] = (mu, sd)
                y_norm[mask] = (Y[tr_idx][mask] - mu) / sd
        fit_ids, ho_ids = early_split(tr_idx)
        model = MTGNN(N_ATOM_FEATS, N_BOND_FEATS, n_twin=n_twin).to(DEVICE)
        if pretrained_state is not None:
            model.load_encoder(pretrained_state)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        pos_of = {int(o): p for p, o in enumerate(tr_idx)}
        pos_of_all = {int(o): p for p, o in enumerate(tr_idx)}

        def predict_ids(ids, m=model):
            m.eval()
            out = np.empty(len(ids), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(ids), 256):
                    bi = ids[i:i + 256]
                    graphs = [row_to_graph[int(b)] for b in bi]
                    batch = to_pyg(graphs).to(DEVICE)
                    twin = torch.tensor(twin_train[bi], dtype=torch.float)
                    p = m(batch, twin=twin).cpu().numpy()
                    for j, b in enumerate(bi):
                        ti = TARGET_IDX[T[b]]
                        mu, sd = stats[T[b]]
                        out[i + j] = p[j, ti] * sd + mu
            return out

        best, best_r2, pat = None, -np.inf, 0
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = np.random.permutation(len(fit_ids))
            for i in range(0, len(perm), BS):
                bi = fit_ids[perm[i:i + BS]]
                idxs = [pos_of_all[int(b)] for b in bi]
                yb = torch.tensor(y_norm[idxs]).unsqueeze(1).to(DEVICE)
                wb = torch.tensor([row_to_graph[int(b)].w.item() for b in bi],
                                  dtype=torch.float).unsqueeze(1).to(DEVICE)
                graphs = [row_to_graph[int(b)] for b in bi]
                batch = to_pyg(graphs).to(DEVICE)
                twin = torch.tensor(twin_train[bi], dtype=torch.float)
                opt.zero_grad()
                pred = model(batch, twin=twin)
                ti = torch.tensor([TARGET_IDX[T[b]] for b in bi], device=DEVICE)
                pred_sel = pred.gather(1, ti.unsqueeze(1))
                loss = (F.mse_loss(pred_sel, yb, reduction="none") * wb).mean()
                loss.backward(); opt.step()
            hp = predict_ids(ho_ids)
            hr = r2_score(Y[ho_ids], hp)
            if hr > best_r2:
                best_r2 = hr
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break
        model.load_state_dict(best)
        mt_oof_all[va_idx] = predict_ids(va_idx)
        # test prediction via graphs
        model.eval()
        with torch.no_grad():
            te_pred = np.zeros(len(Xte), dtype=np.float32)
            for i in range(0, len(Xte), 256):
                bi = np.arange(i, min(i + 256, len(Xte)))
                graphs = [test_graphs[int(b)] for b in bi]
                batch = to_pyg(graphs).to(DEVICE)
                twin = torch.tensor(twin_test[bi], dtype=torch.float)
                p = model(batch, twin=twin).cpu().numpy()
                for j, b in enumerate(bi):
                    ttt = tef["target_type"].iloc[int(b)]
                    ti = TARGET_IDX[ttt]
                    mu, sd = stats[ttt]
                    te_pred[i + j] = p[j, ti] * sd + mu
        mt_test_folds[:, f] = te_pred
        print(f"seed {seed}  fold {f}: holdout R2={best_r2:.4f} ({time.time()-t0f:.0f}s)", flush=True)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert not np.isnan(mt_oof_all).any()
    return mt_oof_all, mt_test_folds.mean(axis=1)


print("GNN_SEEDS =", GNN_SEEDS, flush=True)
mt_oof_sum = np.zeros(len(X), dtype=np.float32)
mt_test_sum = np.zeros(len(Xte), dtype=np.float32)
for _gs in GNN_SEEDS:
    _oo, _mt = run_gnn_seed(_gs)
    mt_oof_sum += _oo
    mt_test_sum += _mt
mt_oof_all = mt_oof_sum / len(GNN_SEEDS)
mt_test = mt_test_sum / len(GNN_SEEDS)
assert not np.isnan(mt_oof_all).any()

mt_oof = {t: mt_oof_all[idx_of_target[t]] for t in TARGETS}


# =====================================================================
# Per-target fallback vs the GBM trio stack (Ridge on lgb+xgb+cb).
# =====================================================================
print("\n=== GBM trio stack OOF (fallback floor) ===", flush=True)
gbm_oof = {t: {m: np.zeros(len(idx_of_target[t])) for m in ('lgb', 'xgb', 'cb')}
           for t in TARGETS}
gbm_test = {t: {m: np.zeros(len(Xte)) for m in ('lgb', 'xgb', 'cb')} for t in TARGETS}
import xgboost as xgb
import catboost as cb

for t in TARGETS:
    idx = idx_of_target[t]
    Xt, yt, gt = Xs[idx], Y[idx], G[idx]
    for tr_idx, va_idx in GroupKFold(n_splits=GLOBAL_FOLDS).split(Xt, yt, gt):
        fit_ids, ho_ids = early_split(tr_idx)
        l = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03,
                              num_leaves=15, min_child_samples=10, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, verbose=-1)
        x = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             random_state=SEED, verbosity=0)
        c = cb.CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6,
                                 random_seed=SEED, task_type='CPU', verbose=False,
                                 allow_writing_files=False)
        for m, est in ((l, l), (x, x), (c, c)):
            est.fit(Xt[fit_ids], yt[fit_ids], eval_set=[(Xt[ho_ids], yt[ho_ids])])
        gbm_oof[t]['lgb'][va_idx] = l.predict(Xt[va_idx])
        gbm_oof[t]['xgb'][va_idx] = x.predict(Xt[va_idx])
        gbm_oof[t]['cb'][va_idx] = c.predict(Xt[va_idx])
        gbm_test[t]['lgb'] += l.predict(Xtes) / GLOBAL_FOLDS
        gbm_test[t]['xgb'] += x.predict(Xtes) / GLOBAL_FOLDS
        gbm_test[t]['cb'] += c.predict(Xtes) / GLOBAL_FOLDS
    print(f"  {t} done", flush=True)

from sklearn.linear_model import Ridge

stack_oof = {}
stack_test = {}
for t in TARGETS:
    idx = idx_of_target[t]
    yt = Y[idx]; gt = G[idx]
    M = np.column_stack([gbm_oof[t][m] for m in ('lgb', 'xgb', 'cb')])
    Mte = np.column_stack([gbm_test[t][m] for m in ('lgb', 'xgb', 'cb')])
    oof = np.zeros(len(idx)); te_pred = np.zeros(len(Xte))
    for tr_idx, va_idx in GroupKFold(n_splits=GLOBAL_FOLDS).split(M, yt, gt):
        r = Ridge(alpha=1.0).fit(M[tr_idx], yt[tr_idx])
        oof[va_idx] = r.predict(M[va_idx])
        te_pred += r.predict(Mte) / GLOBAL_FOLDS
    stack_oof[t] = oof; stack_test[t] = te_pred


# =====================================================================
# Persist fold OOF predictions for the level-2 super-blend (Exp 3/4).
# Writes global-index-aligned arrays (train rows / test rows) so the blend
# script can stack GBM + MT-GNN + retrieval without re-running the GNN.
# =====================================================================
OOF_NPZ = os.path.join(OUT, "superblend_oof.npz")
oof_gbm_global = np.full(len(X), np.nan, dtype=np.float32)
oof_mt_global = np.full(len(X), np.nan, dtype=np.float32)
for t in TARGETS:
    idx = idx_of_target[t]
    oof_gbm_global[idx] = stack_oof[t]
    oof_mt_global[idx] = mt_oof[t]
test_gbm_global = np.zeros(len(Xte), dtype=np.float32)
test_mt_global = np.zeros(len(Xte), dtype=np.float32)
for t in TARGETS:
    m_te = (tef["target_type"] == t).values
    test_gbm_global[m_te] = stack_test[t][m_te]
    test_mt_global[m_te] = mt_test[m_te]
np.savez(OOF_NPZ,
         oof_gbm=oof_gbm_global, oof_mt=oof_mt_global,
         test_gbm=test_gbm_global, test_mt=test_mt_global,
         target_type_train=T.astype("U4"), target_type_test=tef["target_type"].values.astype("U4"),
         y_train=Y)
print("saved superblend OOF caches to", OOF_NPZ, flush=True)


# =====================================================================
# Verdict + per-target fallback selection
# =====================================================================
print("\n=== Per-target: GBM stack vs MT-GNN v2 (fallback = best) ===", flush=True)
rows = []
for t in TARGETS:
    idx = idx_of_target[t]
    r_s = r2_score(Y[idx], stack_oof[t])
    r_m = r2_score(Y[idx], mt_oof[t])
    rows.append((t, r_s, r_m))
    print(f"  {t:<5} stack={r_s:.4f}  mtgnn={r_m:.4f}  "
          f"{'MT' if r_m >= r_s else 'GBM'}", flush=True)
n_s = np.mean([r[1] for r in rows]); n_m = np.mean([r[2] for r in rows])
print(f"\nmean: GBM stack {n_s:.4f} | MT-GNN v2 {n_m:.4f} | delta {n_m-n_s:+.4f}",
      flush=True)

pd.DataFrame(rows, columns=["target", "gbm_stack_oof", "mtgnn_oof"]).round(4)\
    .to_csv(os.path.join(OUT, "mtgnn_v2_compare.csv"), index=False)

# Save OOF/test caches for blending (gnn_oof.csv / gnn_test.csv format)
sub = pd.DataFrame({"id": tef["id"].values, "target_type": tef["target_type"].values})
final_te = np.zeros(len(Xte))
for t in TARGETS:
    m_tr = (T == t); m_te = (tef["target_type"] == t).values
    use_mt = r2_score(Y[idx_of_target[t]], mt_oof[t]) >= r2_score(Y[idx_of_target[t]],
                                                                  stack_oof[t])
    final_te[m_te] = mt_test[m_te] if use_mt else stack_test[t][m_te]
    print(f"  {t}: using {'MT' if use_mt else 'GBM'} for test", flush=True)
sub["target"] = final_te
sub[["id", "target"]].to_csv(os.path.join(OUT, "submission.csv"), index=False)
print("\nwrote submission.csv + mtgnn_v2_compare.csv to", OUT, flush=True)
print("DONE")
