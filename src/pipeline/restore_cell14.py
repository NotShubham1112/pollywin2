import json

CELL14 = '''print("\\n" + "="*70)
print("SECTION 7 \\u2014 MODEL TRAINING")
print("="*70)

def make_models():
    lgb_model = lgb.LGBMRegressor(
        n_estimators=MAX_ESTIMATORS, learning_rate=0.03, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1
    )
    xgb_model = xgb.XGBRegressor(
        n_estimators=MAX_ESTIMATORS, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, tree_method=XGB_TREE_METHOD,
        random_state=SEED, verbosity=0, early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    cb_model = cb.CatBoostRegressor(
        iterations=MAX_ESTIMATORS, learning_rate=0.03, depth=6,
        random_seed=SEED, task_type=CB_TASK_TYPE, verbose=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    return lgb_model, xgb_model, cb_model

results = {}
oof_store = {}
test_pred_store = {}
fold_importance_store = {}

for tt in sorted(train['target_type'].unique()):
    print(f"\\n----- Target: {tt} -----")
    tr = train[train['target_type'] == tt].reset_index(drop=True)
    te = test[test['target_type'] == tt].reset_index(drop=True)

    X = tr[feature_cols].values
    y = tr['target'].values
    groups = tr['canonical_smiles'].values
    X_test = te[feature_cols].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    n = len(y)
    oof_lgb, oof_xgb, oof_cb = np.zeros(n), np.zeros(n), np.zeros(n)
    test_pred_lgb = np.zeros((N_FOLDS, len(te)))
    test_pred_xgb = np.zeros((N_FOLDS, len(te)))
    test_pred_cb = np.zeros((N_FOLDS, len(te)))
    fold_scores = {"lgb": [], "xgb": [], "cb": []}
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr_full, X_va = X[tr_idx], X[va_idx]
        y_tr_full, y_va = y[tr_idx], y[va_idx]

        # Nested split: carve an early-stopping holdout OUT OF THE TRAINING FOLD ONLY.
        # The validation fold (X_va/y_va) is never used for early stopping -> no leakage into OOF.
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_tr_full, y_tr_full, test_size=EARLY_STOP_HOLDOUT_FRAC, random_state=SEED
        )

        lgb_model, xgb_model, cb_model = make_models()

        lgb_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        cb_model.fit(X_tr, y_tr, eval_set=(X_es, y_es))

        p_lgb = lgb_model.predict(X_va)
        p_xgb = xgb_model.predict(X_va)
        p_cb = cb_model.predict(X_va)

        oof_lgb[va_idx] = p_lgb
        oof_xgb[va_idx] = p_xgb
        oof_cb[va_idx] = p_cb

        test_pred_lgb[fold] = lgb_model.predict(X_test)
        test_pred_xgb[fold] = xgb_model.predict(X_test)
        test_pred_cb[fold] = cb_model.predict(X_test)

        r2_lgb_fold = r2_score(y_va, p_lgb)
        r2_xgb_fold = r2_score(y_va, p_xgb)
        r2_cb_fold = r2_score(y_va, p_cb)
        fold_scores["lgb"].append(r2_lgb_fold)
        fold_scores["xgb"].append(r2_xgb_fold)
        fold_scores["cb"].append(r2_cb_fold)
        importances.append(lgb_model.feature_importances_)

        print(f"  Fold {fold}: LGB R2={r2_lgb_fold:.4f}  XGB R2={r2_xgb_fold:.4f}  CB R2={r2_cb_fold:.4f}")

    for name, arr in fold_scores.items():
        print(f"  {name.upper()} \\u2014 mean R2={np.mean(arr):.4f}  std={np.std(arr):.4f}")

    results[tt] = {
        "lgb": {"oof_r2": r2_score(y, oof_lgb), "fold_mean": float(np.mean(fold_scores["lgb"])), "fold_std": float(np.std(fold_scores["lgb"]))},
        "xgb": {"oof_r2": r2_score(y, oof_xgb), "fold_mean": float(np.mean(fold_scores["xgb"])), "fold_std": float(np.std(fold_scores["xgb"]))},
        "cb": {"oof_r2": r2_score(y, oof_cb), "fold_mean": float(np.mean(fold_scores["cb"])), "fold_std": float(np.std(fold_scores["cb"]))},
    }
    oof_store[tt] = {"y": y, "lgb": oof_lgb, "xgb": oof_xgb, "cb": oof_cb, "tr_index": tr.index.values}
    test_pred_store[tt] = {
        "lgb": test_pred_lgb.mean(axis=0),
        "xgb": test_pred_xgb.mean(axis=0),
        "cb": test_pred_cb.mean(axis=0),
        "ids": te["id"].values if "id" in te.columns else None,
    }
    fold_importance_store[tt] = np.mean(importances, axis=0)
    print(f"\\n  OOF R2 (full, out-of-fold) \\u2014 LGB={results[tt]['lgb']['oof_r2']:.4f}  "
          f"XGB={results[tt]['xgb']['oof_r2']:.4f}  CB={results[tt]['cb']['oof_r2']:.4f}")
'''

p = r'd:\Parth\ploywin r2\polymer_prediction_notebook.ipynb'
nb = json.load(open(p, encoding='utf-8'))
print('cell14 was:', repr(''.join(nb['cells'][14]['source'])[:60]))
nb['cells'][14]['source'] = [CELL14]
json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1)
print('cell14 restored, len:', len(CELL14))
