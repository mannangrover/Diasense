"""Tune RF and LR hyperparameters."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import RANDOM_STATE, N_FOLDS

X = np.load('outputs/features/X.npy')
y = np.load('outputs/features/y.npy')
lengths = np.load('outputs/features/lengths.npy')

# Flatten sequences to (n, 22) by averaging valid days
n = X.shape[0]
X_flat = np.zeros((n, X.shape[2]), dtype=np.float32)
for i in range(n):
    X_flat[i] = np.nanmean(X[i, :lengths[i], :], axis=0)

# Also create features with std across days (temporal variability)
X_std = np.zeros((n, X.shape[2]), dtype=np.float32)
for i in range(n):
    X_std[i] = np.nanstd(X[i, :lengths[i], :], axis=0)

# Combine mean + std = 44 features
X_extended = np.hstack([X_flat, X_std])

# Impute
imputer = SimpleImputer(strategy='median')
X_flat_imp = imputer.fit_transform(X_flat)
imputer2 = SimpleImputer(strategy='median')
X_ext_imp = imputer2.fit_transform(X_extended)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def evaluate_model(model, X_data, name):
    oof_proba = np.zeros((len(y), 4))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_data[train_idx])
        X_va = scaler.transform(X_data[val_idx])
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_tr, y[train_idx])
        proba = m.predict_proba(X_va)
        for j, cls in enumerate(m.classes_):
            oof_proba[val_idx, cls] = proba[:, j]
    auc = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')
    return auc, oof_proba

print("=" * 70)
print("RANDOM FOREST TUNING")
print("=" * 70)

rf_configs = [
    {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 10, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 15, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 20, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": None, "class_weight": "balanced"},
    {"n_estimators": 1000, "max_depth": 15, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 15, "class_weight": "balanced", "min_samples_leaf": 5},
    {"n_estimators": 500, "max_depth": 15, "class_weight": "balanced", "min_samples_leaf": 10},
    {"n_estimators": 500, "max_depth": 15, "class_weight": "balanced_subsample"},
    {"n_estimators": 500, "max_depth": 15, "class_weight": "balanced", "max_features": "log2"},
]

best_rf_auc = 0
best_rf_name = ""
for i, cfg in enumerate(rf_configs):
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
    # Try both 22-feature and 44-feature
    auc22, _ = evaluate_model(model, X_flat_imp, f"RF-{i}")
    auc44, _ = evaluate_model(model, X_ext_imp, f"RF-{i}-ext")
    marker22 = " ***BEST***" if auc22 > best_rf_auc else ""
    marker44 = " ***BEST***" if auc44 > best_rf_auc else ""
    if auc22 > best_rf_auc:
        best_rf_auc = auc22
        best_rf_name = f"RF-{i} (22feat)"
    if auc44 > best_rf_auc:
        best_rf_auc = auc44
        best_rf_name = f"RF-{i} (44feat)"
    print(f"  RF cfg {i}: 22feat={auc22:.4f}{marker22}  44feat={auc44:.4f}{marker44}  {cfg}", flush=True)

print(f"\nBest RF: {best_rf_name} = {best_rf_auc:.4f}")

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION TUNING")
print("=" * 70)

lr_configs = [
    {"C": 0.01, "class_weight": "balanced", "max_iter": 2000},
    {"C": 0.1, "class_weight": "balanced", "max_iter": 2000},
    {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
    {"C": 10.0, "class_weight": "balanced", "max_iter": 2000},
    {"C": 0.1, "class_weight": "balanced", "max_iter": 2000, "penalty": "l1", "solver": "saga"},
    {"C": 1.0, "class_weight": "balanced", "max_iter": 2000, "penalty": "l1", "solver": "saga"},
    {"C": 0.1, "class_weight": "balanced", "max_iter": 2000, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
]

best_lr_auc = 0
best_lr_name = ""
for i, cfg in enumerate(lr_configs):
    model = LogisticRegression(random_state=RANDOM_STATE, **cfg)
    auc22, _ = evaluate_model(model, X_flat_imp, f"LR-{i}")
    auc44, _ = evaluate_model(model, X_ext_imp, f"LR-{i}-ext")
    marker22 = " ***BEST***" if auc22 > best_lr_auc else ""
    marker44 = " ***BEST***" if auc44 > best_lr_auc else ""
    if auc22 > best_lr_auc:
        best_lr_auc = auc22
        best_lr_name = f"LR-{i} (22feat)"
    if auc44 > best_lr_auc:
        best_lr_auc = auc44
        best_lr_name = f"LR-{i} (44feat)"
    print(f"  LR cfg {i}: 22feat={auc22:.4f}{marker22}  44feat={auc44:.4f}{marker44}  {cfg}", flush=True)

print(f"\nBest LR: {best_lr_name} = {best_lr_auc:.4f}")

print("\n" + "=" * 70)
print("GRADIENT BOOSTING (bonus)")
print("=" * 70)

gb_configs = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.01},
]

best_gb_auc = 0
for i, cfg in enumerate(gb_configs):
    model = GradientBoostingClassifier(random_state=RANDOM_STATE, **cfg)
    auc22, _ = evaluate_model(model, X_flat_imp, f"GB-{i}")
    auc44, _ = evaluate_model(model, X_ext_imp, f"GB-{i}-ext")
    marker22 = " ***BEST***" if auc22 > best_gb_auc else ""
    marker44 = " ***BEST***" if auc44 > best_gb_auc else ""
    if auc22 > best_gb_auc:
        best_gb_auc = auc22
    if auc44 > best_gb_auc:
        best_gb_auc = auc44
    print(f"  GB cfg {i}: 22feat={auc22:.4f}{marker22}  44feat={auc44:.4f}{marker44}  {cfg}", flush=True)

print(f"\nBest GB: {best_gb_auc:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Best RF:  {best_rf_auc:.4f}")
print(f"  Best LR:  {best_lr_auc:.4f}")
print(f"  Best GB:  {best_gb_auc:.4f}")
