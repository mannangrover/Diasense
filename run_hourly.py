"""
Hourly Resolution LSTM — Extraction + Evaluation
=================================================
Shape: (1586, 336, 10)  [14 days × 24 hours × 10 features]
Model: Attention-LSTM, hidden=128, 2 layers — runs on CUDA if available.

Saves X_hourly.npy before evaluation so extraction is never repeated.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pathlib import Path

from config import (RANDOM_STATE, N_FOLDS, LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2,
                    CLASS_NAMES_4, set_seed)
from src.data.cohort import build_cohort
from src.features.hourly import extract_hourly_features, SEQ_HOURS, NUM_HOURLY_FEATURES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

FEATURES_DIR = Path("outputs/features")
HOURLY_PATH = FEATURES_DIR / "X_hourly.npy"
LENGTHS_PATH = FEATURES_DIR / "lengths_hourly.npy"

# =============================================================================
# STEP 1: EXTRACTION (skip if already saved)
# =============================================================================

y = np.load(FEATURES_DIR / "y.npy")
person_ids = np.load(FEATURES_DIR / "person_ids.npy")

if HOURLY_PATH.exists() and LENGTHS_PATH.exists():
    print(f"\nLoading cached X_hourly from {HOURLY_PATH}")
    X_hourly = np.load(HOURLY_PATH)
    lengths_hourly = np.load(LENGTHS_PATH)
    print(f"  Shape: {X_hourly.shape}, lengths range [{lengths_hourly.min()}–{lengths_hourly.max()}]")
else:
    print(f"\nExtracting hourly features for {len(person_ids)} participants...")
    cohort = build_cohort(exclude_dead_sensors=True)
    pid_to_row = {int(row["person_id"]): row for _, row in cohort.iterrows()}

    # We need the existing dates per participant (from the saved X.npy)
    # Reconstruct from the original lengths and the original feature sequence
    X_orig = np.load(FEATURES_DIR / "X.npy")
    lengths_orig = np.load(FEATURES_DIR / "lengths.npy")

    n = len(person_ids)
    X_hourly = np.zeros((n, SEQ_HOURS, NUM_HOURLY_FEATURES), dtype=np.float32)
    lengths_hourly = np.zeros(n, dtype=np.int64)

    start = time.time()
    errors = 0

    for i, pid in enumerate(person_ids):
        try:
            row = pid_to_row.get(int(pid))
            if row is None:
                errors += 1
                continue

            # Reconstruct the dates from the original cohort data
            # We extract dates from the manifest the same way daily_summary does
            from src.features.continuous import extract_continuous_features
            cont = extract_continuous_features(row)
            existing_dates = sorted(cont.keys()) if cont else []

            if not existing_dates:
                errors += 1
                continue

            X_seq, actual_len = extract_hourly_features(row, existing_dates)
            X_hourly[i] = X_seq
            lengths_hourly[i] = actual_len

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR PID {pid}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            print(f"  {i+1}/{n} | {elapsed:.0f}s elapsed | ETA {eta:.1f}min")

    elapsed = time.time() - start
    print(f"\nExtraction done: {n - errors}/{n} ok in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Shape: {X_hourly.shape}")
    print(f"  Lengths range: [{lengths_hourly.min()}–{lengths_hourly.max()}]")

    # Check NaN coverage
    valid_mask = lengths_hourly > 0
    nan_pct = np.isnan(X_hourly[valid_mask]).mean() * 100
    print(f"  NaN rate in valid rows: {nan_pct:.1f}%")

    np.save(HOURLY_PATH, X_hourly)
    np.save(LENGTHS_PATH, lengths_hourly)
    print(f"  Saved to {HOURLY_PATH}")

# =============================================================================
# STEP 2: LSTM MODEL
# =============================================================================

class HourlyLSTM(nn.Module):
    """Attention LSTM for 336-step hourly sequences."""
    def __init__(self, input_size=NUM_HOURLY_FEATURES, hidden=128, layers=2, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            dropout=drop if layers > 1 else 0,
                            batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Dropout(drop),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(drop / 2),
            nn.Linear(64, 4),
        )

    def forward(self, x, lens=None):
        if lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        # Attention
        scores = self.attn(out).squeeze(-1)
        if lens is not None:
            mask = (torch.arange(out.size(1), device=out.device).unsqueeze(0)
                    >= lens.unsqueeze(1).to(out.device))
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)
        return self.head(context)


# =============================================================================
# STEP 3: EVALUATION
# =============================================================================

def eval_hourly_lstm(X_seq, y, lens, label="Hourly"):
    set_seed()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 4))
    nf = X_seq.shape[2]
    sl = X_seq.shape[1]

    print(f"\n  Running {N_FOLDS}-fold CV for {label}...")

    for fold, (tr, va) in enumerate(skf.split(X_seq, y)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", flush=True)
        ntr, nva = len(tr), len(va)

        # Impute NaNs per fold (fit on train, apply to val)
        Xtr_flat = X_seq[tr].reshape(-1, nf).copy()
        Xva_flat = X_seq[va].reshape(-1, nf).copy()

        imp = SimpleImputer(strategy='median')
        Xtr_flat = imp.fit_transform(Xtr_flat)
        Xva_flat = imp.transform(Xva_flat)

        sc = StandardScaler()
        Xtr_flat = sc.fit_transform(Xtr_flat)
        Xva_flat = sc.transform(Xva_flat)

        Xt = torch.tensor(Xtr_flat.reshape(ntr, sl, nf).astype(np.float32))
        yt = torch.tensor(y[tr], dtype=torch.long)
        lt = torch.tensor(lens[tr], dtype=torch.long)

        Xv = torch.tensor(Xva_flat.reshape(nva, sl, nf).astype(np.float32)).to(device)
        yv = torch.tensor(y[va], dtype=torch.long).to(device)
        lv = torch.tensor(lens[va], dtype=torch.long)

        loader = DataLoader(TensorDataset(Xt, yt, lt), batch_size=128, shuffle=True)

        # Class weights
        cc = np.bincount(y[tr], minlength=4).astype(float)
        cw = 1.0 / (cc + 1e-6)
        cw = cw / cw.sum() * 4
        wt = torch.tensor(cw, dtype=torch.float32).to(device)

        set_seed()
        model = HourlyLSTM(input_size=nf).to(device)
        crit = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

        best_val_loss, patience_count = float('inf'), 0
        best_state = None

        for ep in range(100):
            model.train()
            for Xb, yb, lb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(Xb, lb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = crit(model(Xv, lv), yv).item()
            sched.step(vl)

            if vl < best_val_loss:
                best_val_loss = vl
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= 20:
                    print(f"      Early stop at epoch {ep+1}", flush=True)
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_proba[va] = torch.softmax(model(Xv, lv), dim=1).cpu().numpy()

    pred = np.argmax(oof_proba, axis=1)
    acc = accuracy_score(y, pred)
    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')

    y3 = np.array([LABEL_MAP_4_TO_3[i] for i in y])
    p3 = np.zeros((len(y), 3))
    p3[:, 0] = oof_proba[:, 0]
    p3[:, 1] = oof_proba[:, 1]
    p3[:, 2] = oof_proba[:, 2] + oof_proba[:, 3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')

    y2 = np.array([LABEL_MAP_4_TO_2[i] for i in y])
    auc2 = roc_auc_score(y2, oof_proba[:, 1] + oof_proba[:, 2] + oof_proba[:, 3])

    print(f"\n{'='*70}")
    print(f"  {label} LSTM Results:")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  4-class AUC: {auc4:.4f}")
    print(f"  3-class AUC: {auc3:.4f}")
    print(f"  2-class AUC: {auc2:.4f}")
    print(f"{'='*70}")
    print(f"\n  Per-class report:")
    print(classification_report(y, pred, target_names=CLASS_NAMES_4, digits=3))

    np.save(FEATURES_DIR / "hourly_oof_proba.npy", oof_proba)
    print("  Saved OOF probabilities to outputs/features/hourly_oof_proba.npy")

    return acc, auc4, auc3, auc2


print("\n" + "=" * 70)
print("HOURLY LSTM (336 timesteps × 10 features)")
print("=" * 70)

eval_hourly_lstm(X_hourly, y, lengths_hourly, "Hourly-336")

print("\nDone.")
