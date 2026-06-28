"""
Baseline Models
===============
Quick Random Forest / Logistic Regression baseline on flattened features.

Purpose: sanity check that our 22 daily features carry diabetes signal.
If RF can't beat random chance, something is wrong with the features.

Usage:
    from src.models.baseline import run_baseline
    results = run_baseline(X, y, lengths)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import N_FOLDS, RANDOM_STATE, CLASS_NAMES_4
from src.evaluation.metrics import evaluate_hierarchical


def _flatten_sequences(X, lengths):
    """
    Convert (n, 14, 22) sequences to (n, 22) by averaging across valid days.

    For each participant, average only the non-padded timesteps
    (determined by `lengths`).

    Also handles NaN: uses nanmean so missing modality days are skipped.
    """

    n = X.shape[0]
    n_features = X.shape[2]
    X_flat = np.zeros((n, n_features), dtype=np.float32)

    for i in range(n):
        valid_days = X[i, :lengths[i], :]
        # Replace zeros in padded region (shouldn't happen, but safety)
        X_flat[i] = np.nanmean(valid_days, axis=0)

    return X_flat


def run_baseline(X, y, lengths, person_ids=None):
    """
    Run RF and LR baselines with stratified K-Fold cross-validation.

    Steps:
        1. Flatten sequences to (n, 22) by averaging valid days
        2. Impute remaining NaN with median
        3. Scale features
        4. Train RF and LR with StratifiedKFold
        5. Evaluate at 4/3/2 class levels

    Args:
        X: np.array (n, 14, 22)
        y: np.array (n,)
        lengths: np.array (n,)
        person_ids: optional, not used for baseline (no leakage risk with flat features)

    Returns:
        dict with results for RF and LR
    """

    # Step 1: Flatten
    X_flat = _flatten_sequences(X, lengths)
    print(f"Flattened: {X_flat.shape}")

    # Step 2: Impute NaN
    imputer = SimpleImputer(strategy="median")
    X_flat = imputer.fit_transform(X_flat)

    nan_count = np.isnan(X_flat).sum()
    print(f"NaN after imputation: {nan_count}")

    # Step 3: K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}

    for name, model_template in models.items():
        print(f"\n{'#'*60}")
        print(f"  {name}")
        print(f"{'#'*60}")

        # Collect OOF predictions
        oof_proba = np.zeros((len(y), 4))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_flat, y)):
            X_train, X_val = X_flat[train_idx], X_flat[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # Clone and fit
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train, y_train)

            # Predict probabilities
            proba = model.predict_proba(X_val)

            # Handle missing classes in fold
            for j, cls in enumerate(model.classes_):
                oof_proba[val_idx, cls] = proba[:, j]

            print(f"  Fold {fold+1}: val size={len(val_idx)}")

        # Evaluate
        print(f"\n--- {name} OOF Results ---")
        results[name] = evaluate_hierarchical(y, oof_proba, verbose=True)

    return results
