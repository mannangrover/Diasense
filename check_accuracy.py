"""Compute accuracy for best RF, LR, and LSTM configs."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from config import RANDOM_STATE, N_FOLDS, CLASS_NAMES_4

X = np.load('outputs/features/X.npy')
y = np.load('outputs/features/y.npy')
lengths = np.load('outputs/features/lengths.npy')

n = X.shape[0]
X_flat = np.zeros((n, X.shape[2]), dtype=np.float32)
X_std = np.zeros((n, X.shape[2]), dtype=np.float32)
for i in range(n):
    X_flat[i] = np.nanmean(X[i, :lengths[i], :], axis=0)
    X_std[i] = np.nanstd(X[i, :lengths[i], :], axis=0)

X_ext = np.hstack([X_flat, X_std])
imp = SimpleImputer(strategy='median')
X_ext = imp.fit_transform(X_ext)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ---- RF (best config) ----
print("=" * 60)
print("RANDOM FOREST (best: 500 trees, depth=15, min_leaf=5, 44feat)")
print("=" * 60)
oof_pred_rf = np.zeros(len(y), dtype=int)
oof_proba_rf = np.zeros((len(y), 4))
for fold, (tr, va) in enumerate(skf.split(X_ext, y)):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_ext[tr])
    Xva = sc.transform(X_ext[va])
    m = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                               class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(Xtr, y[tr])
    oof_pred_rf[va] = m.predict(Xva)
    p = m.predict_proba(Xva)
    for j, c in enumerate(m.classes_):
        oof_proba_rf[va, c] = p[:, j]

acc_rf = accuracy_score(y, oof_pred_rf)
auc_rf = roc_auc_score(y, oof_proba_rf, multi_class='ovr', average='macro')
print(f"Accuracy: {acc_rf:.4f} ({acc_rf*100:.1f}%)")
print(f"ROC-AUC:  {auc_rf:.4f}")
print("\nPer-class report:")
print(classification_report(y, oof_pred_rf, target_names=CLASS_NAMES_4, digits=3))

# ---- LR (best config) ----
print("=" * 60)
print("LOGISTIC REGRESSION (best: C=0.01, 44feat)")
print("=" * 60)
oof_pred_lr = np.zeros(len(y), dtype=int)
oof_proba_lr = np.zeros((len(y), 4))
for fold, (tr, va) in enumerate(skf.split(X_ext, y)):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_ext[tr])
    Xva = sc.transform(X_ext[va])
    m = LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
    m.fit(Xtr, y[tr])
    oof_pred_lr[va] = m.predict(Xva)
    p = m.predict_proba(Xva)
    for j, c in enumerate(m.classes_):
        oof_proba_lr[va, c] = p[:, j]

acc_lr = accuracy_score(y, oof_pred_lr)
auc_lr = roc_auc_score(y, oof_proba_lr, multi_class='ovr', average='macro')
print(f"Accuracy: {acc_lr:.4f} ({acc_lr*100:.1f}%)")
print(f"ROC-AUC:  {auc_lr:.4f}")
print("\nPer-class report:")
print(classification_report(y, oof_pred_lr, target_names=CLASS_NAMES_4, digits=3))

# ---- LSTM (from saved OOF proba) ----
print("=" * 60)
print("LSTM (best: combo1 — attn, dropout=0.4, lr=0.0005)")
print("=" * 60)
oof_proba_lstm = np.load('outputs/features/best_lstm_oof_proba.npy')
oof_pred_lstm = np.argmax(oof_proba_lstm, axis=1)
acc_lstm = accuracy_score(y, oof_pred_lstm)
auc_lstm = roc_auc_score(y, oof_proba_lstm, multi_class='ovr', average='macro')
print(f"Accuracy: {acc_lstm:.4f} ({acc_lstm*100:.1f}%)")
print(f"ROC-AUC:  {auc_lstm:.4f}")
print("\nPer-class report:")
print(classification_report(y, oof_pred_lstm, target_names=CLASS_NAMES_4, digits=3))

# ---- Summary ----
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Model':<25s}  {'Accuracy':>10s}  {'ROC-AUC':>10s}")
print("-" * 50)
print(f"{'Random Forest':<25s}  {acc_rf*100:>9.1f}%  {auc_rf:>10.4f}")
print(f"{'Logistic Regression':<25s}  {acc_lr*100:>9.1f}%  {auc_lr:>10.4f}")
print(f"{'LSTM (combo1)':<25s}  {acc_lstm*100:>9.1f}%  {auc_lstm:>10.4f}")

# Class distribution for context
print(f"\nClass distribution:")
for i, name in enumerate(CLASS_NAMES_4):
    cnt = (y == i).sum()
    print(f"  {name}: {cnt} ({cnt/len(y)*100:.1f}%)")
print(f"  Majority-class baseline accuracy: {np.bincount(y).max()/len(y)*100:.1f}%")
