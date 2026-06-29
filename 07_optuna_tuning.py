"""Optuna hyperparameter tuning for LightGBM — all 3 evaluation levels."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pathlib import Path

from config import (RANDOM_STATE, N_FOLDS, LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2,
                    CLASS_NAMES_4, set_seed)

optuna.logging.set_verbosity(optuna.logging.WARNING)
FEATURES_DIR = Path(__file__).parent / 'outputs' / 'features'
set_seed()

# Load data
X_orig = np.load(FEATURES_DIR / 'X.npy')
X_c    = np.load(FEATURES_DIR / 'X_option_c.npy')
X_survey = np.load(FEATURES_DIR / 'X_survey.npy')
y      = np.load(FEATURES_DIR / 'y.npy')
lengths = np.load(FEATURES_DIR / 'lengths.npy')
X_wear = np.concatenate([X_orig, X_c], axis=2)
N, T, F = X_wear.shape

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    stats = []
    for f in range(F):
        col = X_wear[:, :, f]
        col_mean = np.nanmean(col, axis=1)
        col_std  = np.nanstd(col, axis=1)
        col_min  = np.nanmin(col, axis=1)
        col_max  = np.nanmax(col, axis=1)
        slope = np.zeros(N)
        diff = np.zeros(N)
        for i in range(N):
            valid = col[i, :int(lengths[i])]
            valid = valid[~np.isnan(valid)]
            if len(valid) >= 2:
                slope[i] = np.polyfit(np.arange(len(valid)), valid, 1)[0]
                diff[i] = valid[-1] - valid[0]
        stats.extend([col_mean, col_std, col_min, col_max, col_max - col_min, slope, diff])
    stats.append(lengths.astype(float))
    X_flat = np.column_stack(stats)

X_summary = np.concatenate([X_flat, X_survey], axis=1)
print(f'Summary features: {X_summary.shape[1]}')

y3 = np.array([LABEL_MAP_4_TO_3[i] for i in y])
y2 = np.array([LABEL_MAP_4_TO_2[i] for i in y])

N_TRIALS = 150

def make_params(trial, objective, num_class=None):
    p = {
        'boosting_type': 'gbdt', 'verbose': -1, 'is_unbalance': True,
        'random_state': RANDOM_STATE, 'n_estimators': 2000,
        'objective': objective,
        'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 5.0),
    }
    if objective == 'multiclass':
        p['num_class'] = num_class
        p['metric'] = 'multi_logloss'
    else:
        p['metric'] = 'binary_logloss'
    return p

# ===================== 4-CLASS =====================
print(f'\n=== Optuna 4-class ({N_TRIALS} trials) ===')
def obj4(trial):
    params = make_params(trial, 'multiclass', 4)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((N, 4))
    for tr, va in skf.split(X_summary, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_summary[tr], y[tr], eval_set=[(X_summary[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X_summary[va])
    return roc_auc_score(y, oof, multi_class='ovr', average='macro')

study4 = optuna.create_study(direction='maximize')
study4.optimize(obj4, n_trials=N_TRIALS)
print(f'Best 4-AUC: {study4.best_value:.4f}')
print(f'Params: {study4.best_params}')

# Retrain best
bp4 = make_params(study4.best_trial, 'multiclass', 4)
set_seed()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_opt4 = np.zeros((N, 4))
for fold, (tr, va) in enumerate(skf.split(X_summary, y)):
    m = lgb.LGBMClassifier(**bp4)
    m.fit(X_summary[tr], y[tr], eval_set=[(X_summary[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_opt4[va] = m.predict_proba(X_summary[va])
np.save(FEATURES_DIR / 'oof_optuna_4class.npy', oof_opt4)

# ===================== 3-CLASS =====================
print(f'\n=== Optuna 3-class ({N_TRIALS} trials) ===')
def obj3(trial):
    params = make_params(trial, 'multiclass', 3)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((N, 3))
    for tr, va in skf.split(X_summary, y3):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_summary[tr], y3[tr], eval_set=[(X_summary[va], y3[va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X_summary[va])
    return roc_auc_score(y3, oof, multi_class='ovr', average='macro')

study3 = optuna.create_study(direction='maximize')
study3.optimize(obj3, n_trials=N_TRIALS)
print(f'Best 3-AUC: {study3.best_value:.4f}')
print(f'Params: {study3.best_params}')

bp3 = make_params(study3.best_trial, 'multiclass', 3)
set_seed()
oof_opt3 = np.zeros((N, 3))
for fold, (tr, va) in enumerate(skf.split(X_summary, y3)):
    m = lgb.LGBMClassifier(**bp3)
    m.fit(X_summary[tr], y3[tr], eval_set=[(X_summary[va], y3[va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_opt3[va] = m.predict_proba(X_summary[va])
np.save(FEATURES_DIR / 'oof_optuna_3class.npy', oof_opt3)
auc3_opt = roc_auc_score(y3, oof_opt3, multi_class='ovr', average='macro')
print(f'Retrained 3-AUC: {auc3_opt:.4f}')

# ===================== 2-CLASS =====================
print(f'\n=== Optuna 2-class ({N_TRIALS} trials) ===')
def obj2(trial):
    params = make_params(trial, 'binary')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(N)
    for tr, va in skf.split(X_summary, y2):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_summary[tr], y2[tr], eval_set=[(X_summary[va], y2[va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X_summary[va])[:, 1]
    return roc_auc_score(y2, oof)

study2 = optuna.create_study(direction='maximize')
study2.optimize(obj2, n_trials=N_TRIALS)
print(f'Best 2-AUC: {study2.best_value:.4f}')
print(f'Params: {study2.best_params}')

bp2 = make_params(study2.best_trial, 'binary')
set_seed()
oof_opt2 = np.zeros(N)
for fold, (tr, va) in enumerate(skf.split(X_summary, y2)):
    m = lgb.LGBMClassifier(**bp2)
    m.fit(X_summary[tr], y2[tr], eval_set=[(X_summary[va], y2[va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_opt2[va] = m.predict_proba(X_summary[va])[:, 1]
np.save(FEATURES_DIR / 'oof_optuna_2class.npy', oof_opt2)
auc2_opt = roc_auc_score(y2, oof_opt2)
print(f'Retrained 2-AUC: {auc2_opt:.4f}')

# ===================== METRICS =====================
def compute_all(y4, proba4, label):
    pred = np.argmax(proba4, axis=1)
    acc = accuracy_score(y4, pred)
    auc4 = roc_auc_score(y4, proba4, multi_class='ovr', average='macro')
    y3_ = np.array([LABEL_MAP_4_TO_3[i] for i in y4])
    p3 = np.zeros((len(y4), 3))
    p3[:, 0] = proba4[:, 0]; p3[:, 1] = proba4[:, 1]
    p3[:, 2] = proba4[:, 2] + proba4[:, 3]
    auc3 = roc_auc_score(y3_, p3, multi_class='ovr', average='macro')
    y2_ = np.array([LABEL_MAP_4_TO_2[i] for i in y4])
    auc2 = roc_auc_score(y2_, proba4[:, 1] + proba4[:, 2] + proba4[:, 3])
    print(f'  {label}: acc={acc*100:.1f}%  4-AUC={auc4:.4f}  3-AUC={auc3:.4f}  2-AUC={auc2:.4f}')
    return auc4, auc3, auc2

print('\n' + '=' * 70)
print('RESULTS')
print('=' * 70)
auc4_o, auc3_o, auc2_o = compute_all(y, oof_opt4, 'Optuna LGB 4-class')
print(f'  Optuna LGB 3-class dedicated: 3-AUC={auc3_opt:.4f}')
print(f'  Optuna LGB 2-class dedicated: 2-AUC={auc2_opt:.4f}')

# Ensemble with hybrid LSTM
hybrid_oof = np.load(FEATURES_DIR / 'oof_hybrid_T2_a0.5.npy')
print('\nEnsembles (Optuna LGB + Hybrid LSTM):')
best_a4, best_w = 0, 0
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ens = w * oof_opt4 + (1 - w) * hybrid_oof
    a4, a3, a2 = compute_all(y, ens, f'OptLGB×{w:.1f}+Hybrid×{1-w:.1f}')
    if a4 > best_a4:
        best_a4, best_w, best_a3, best_a2 = a4, w, a3, a2

print(f'\n{"="*70}')
print(f'SUMMARY')
print(f'{"="*70}')
print(f'{"Model":<50} {"4-AUC":>7} {"3-AUC":>7} {"2-AUC":>7}')
print(f'{"BASELINE (LSTM distilled)":<50} {"0.6879":>7} {"0.6858":>7} {"0.6846":>7}')
print(f'{"Prev best ensemble":<50} {"0.7338":>7} {"0.7407":>7} {"0.7852":>7}')
print(f'{"Optuna LGB 4-class":<50} {auc4_o:>7.4f} {auc3_o:>7.4f} {auc2_o:>7.4f}')
print(f'{"Optuna LGB 3-class dedicated":<50} {"—":>7} {auc3_opt:>7.4f} {"—":>7}')
print(f'{"Optuna LGB 2-class dedicated":<50} {"—":>7} {"—":>7} {auc2_opt:>7.4f}')
print(f'{"Best ensemble (OptLGB+Hybrid)":<50} {best_a4:>7.4f} {best_a3:>7.4f} {best_a2:>7.4f}')
print(f'\nImprovement from baseline:')
print(f'  4-AUC: {best_a4 - 0.6879:+.4f}')
print(f'  3-AUC: {best_a3 - 0.6858:+.4f}')
print(f'  2-AUC: {best_a2 - 0.6846:+.4f}')
