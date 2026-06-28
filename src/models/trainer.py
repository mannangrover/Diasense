"""
Training Loop
=============
K-Fold cross-validation training for the LSTM with early stopping,
class weighting, and hierarchical evaluation.

Usage:
    from src.models.trainer import train_kfold
    results = train_kfold(X, y, lengths)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    N_FOLDS,
    RANDOM_STATE,
    LSTM_LEARNING_RATE,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    SEQUENCE_LENGTH,
    NUM_DAILY_FEATURES,
    set_seed,
)
from src.models.lstm import DiasenseLSTM
from src.evaluation.metrics import evaluate_hierarchical


def _preprocess(X, lengths):
    """
    Impute NaN and normalize features across the dataset.

    NaN values come from missing modalities (e.g., no SpO2 for a participant).
    We impute with the global median for each feature, then standardize.

    Returns:
        X_processed: np.array same shape as X, NaN-free and normalized
        scaler: fitted StandardScaler (for later use)
    """
    n, seq_len, n_feat = X.shape

    # Reshape to 2D for imputation/scaling: (n * seq_len, n_feat)
    X_2d = X.reshape(-1, n_feat).copy()

    # Impute NaN with median
    imputer = SimpleImputer(strategy="median")
    X_2d = imputer.fit_transform(X_2d)

    # Standardize each feature to mean=0, std=1
    scaler = StandardScaler()
    X_2d = scaler.fit_transform(X_2d)

    return X_2d.reshape(n, seq_len, n_feat).astype(np.float32), scaler


def train_kfold(X, y, lengths, verbose=True):
    """
    Train LSTM with StratifiedKFold cross-validation.

    For each fold:
        1. Split train/val
        2. Impute NaN and normalize (fit on train, transform val)
        3. Compute class weights from train set
        4. Train LSTM with early stopping
        5. Collect OOF predictions

    Args:
        X: np.array (n, 14, 22)
        y: np.array (n,)
        lengths: np.array (n,)

    Returns:
        dict with OOF evaluation results
    """

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_proba = np.zeros((len(y), 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"  FOLD {fold+1}/{N_FOLDS}")
            print(f"{'='*50}")

        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        len_train, len_val = lengths[train_idx], lengths[val_idx]

        # --- Impute and normalize ---
        n_tr, seq_len, n_feat = X_train_raw.shape
        n_va = X_val_raw.shape[0]

        # Fit imputer + scaler on train, transform both
        X_tr_2d = X_train_raw.reshape(-1, n_feat).copy()
        X_va_2d = X_val_raw.reshape(-1, n_feat).copy()

        imputer = SimpleImputer(strategy="median")
        X_tr_2d = imputer.fit_transform(X_tr_2d)
        X_va_2d = imputer.transform(X_va_2d)

        scaler = StandardScaler()
        X_tr_2d = scaler.fit_transform(X_tr_2d)
        X_va_2d = scaler.transform(X_va_2d)

        X_train_np = X_tr_2d.reshape(n_tr, seq_len, n_feat).astype(np.float32)
        X_val_np = X_va_2d.reshape(n_va, seq_len, n_feat).astype(np.float32)

        # --- Tensors ---
        X_train_t = torch.tensor(X_train_np)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        len_train_t = torch.tensor(len_train, dtype=torch.long)

        X_val_t = torch.tensor(X_val_np).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
        len_val_t = torch.tensor(len_val, dtype=torch.long)

        train_ds = TensorDataset(X_train_t, y_train_t, len_train_t)
        train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        # --- Class weights ---
        class_counts = np.bincount(y_train, minlength=4).astype(float)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # --- Model ---
        set_seed()
        model = DiasenseLSTM().to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # --- Training loop with early stopping ---
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(LSTM_EPOCHS):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch, len_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch, len_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)

            train_loss /= len(y_train)

            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t, len_val_t)
                val_loss = criterion(val_logits, y_val_t).item()

            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:>3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and get predictions
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t, len_val_t)
            val_proba = torch.softmax(val_logits, dim=1).cpu().numpy()

        oof_proba[val_idx] = val_proba

        if verbose:
            val_pred = np.argmax(val_proba, axis=1)
            acc = (val_pred == y_val).mean()
            print(f"  Fold {fold+1} val acc: {acc:.4f}")

    # --- Final evaluation ---
    if verbose:
        print(f"\n{'#'*60}")
        print(f"  LSTM OOF RESULTS")
        print(f"{'#'*60}")

    results = evaluate_hierarchical(y, oof_proba, verbose=verbose)

    # Save best model from last fold
    torch.save(best_state, "outputs/models/lstm_best.pt")
    if verbose:
        print("\nSaved best model to outputs/models/lstm_best.pt")

    return results
