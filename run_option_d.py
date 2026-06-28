"""
Option D Evaluation: Original(22) + Option C(19) + Option D(10) = 51 features/day
Extracts Option D (hourly-derived daily) features and evaluates LSTM on CUDA.
Option C .npy already saved — only Option D extraction is new.
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
from src.features.continuous import extract_continuous_features
from src.features.enhanced import (
    extract_option_d_features, NUM_OPTION_D_FEATURES, OPTION_D_FEATURE_NAMES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

FEATURES_DIR = Path("outputs/features")
OPT_D_PATH = FEATURES_DIR / "X_option_d.npy"

# Load existing saved arrays
X_orig = np.load(FEATURES_DIR / "X.npy")
X_c = np.load(FEATURES_DIR / "X_option_c.npy")
y = np.load(FEATURES_DIR / "y.npy")
lengths = np.load(FEATURES_DIR / "lengths.npy")
person_ids = np.load(FEATURES_DIR / "person_ids.npy")

print(f"\nLoaded: X_orig {X_orig.shape}, X_c {X_c.shape}, y {y.shape}", flush=True)

# =============================================================================
# STEP 1: EXTRACT OPTION D (skip if cached)
# =============================================================================

if OPT_D_PATH.exists():
    print(f"\nLoading cached X_option_d from {OPT_D_PATH}", flush=True)
    X_d = np.load(OPT_D_PATH)
else:
    print(f"\nExtracting Option D features for {len(person_ids)} participants...", flush=True)
    cohort = build_cohort(exclude_dead_sensors=True)
    pid_to_row = {int(row["person_id"]): row for _, row in cohort.iterrows()}

    n, seq_len, _ = X_orig.shape
    X_d = np.full((n, seq_len, NUM_OPTION_D_FEATURES), np.nan, dtype=np.float32)

    start = time.time()
    errors = 0

    for i, pid in enumerate(person_ids):
        try:
            row = pid_to_row.get(int(pid))
            if row is None:
                errors += 1
                continue

            cont = extract_continuous_features(row)
            existing_dates = sorted(cont.keys())[:seq_len]
            if not existing_dates:
                errors += 1
                continue

            feats_by_date = extract_option_d_features(row, existing_dates)

            for day_idx, date in enumerate(existing_dates):
                if date in feats_by_date:
                    X_d[i, day_idx] = feats_by_date[date]

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR PID {pid}: {e}", flush=True)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            print(f"  {i+1}/{n} | {elapsed:.0f}s | ETA {eta:.1f}min", flush=True)

    elapsed = time.time() - start
    print(f"\nExtraction done: {n - errors}/{n} ok in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    nan_pct = np.isnan(X_d).mean() * 100
    print(f"  NaN rate: {nan_pct:.1f}%", flush=True)

    np.save(OPT_D_PATH, X_d)
    print(f"  Saved to {OPT_D_PATH}", flush=True)

print(f"\nOption D shape: {X_d.shape}", flush=True)
nan_pct = np.isnan(X_d).mean() * 100
print(f"NaN rate: {nan_pct:.1f}%", flush=True)

# Per-feature NaN coverage
for i, name in enumerate(OPTION_D_FEATURE_NAMES):
    cov = (~np.isnan(X_d[:, :, i])).mean() * 100
    print(f"  {name}: {cov:.1f}% non-nan", flush=True)

# =============================================================================
# STEP 2: LSTM MODEL (same as Option C run)
# =============================================================================

class LSTMAttn(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, drop=0.4):
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
        scores = self.attn(out).squeeze(-1)
        if lens is not None:
            mask = (torch.arange(out.size(1), device=out.device).unsqueeze(0)
                    >= lens.unsqueeze(1).to(out.device))
            scores = scores.masked_fill(mask, float('-inf'))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return self.head((out * w).sum(dim=1))


def eval_lstm(X_seq, y, lens, label):
    set_seed()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 4))
    nf = X_seq.shape[2]
    sl = X_seq.shape[1]

    print(f"\n  {label}: {N_FOLDS}-fold CV | input={sl}×{nf}...", flush=True)

    for fold, (tr, va) in enumerate(skf.split(X_seq, y)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", flush=True)
        ntr, nva = len(tr), len(va)

        Xtr_f = X_seq[tr].reshape(-1, nf).copy()
        Xva_f = X_seq[va].reshape(-1, nf).copy()
        imp = SimpleImputer(strategy='median')
        Xtr_f = imp.fit_transform(Xtr_f)
        Xva_f = imp.transform(Xva_f)
        sc = StandardScaler()
        Xtr_f = sc.fit_transform(Xtr_f)
        Xva_f = sc.transform(Xva_f)

        Xt = torch.tensor(Xtr_f.reshape(ntr, sl, nf).astype(np.float32))
        yt = torch.tensor(y[tr], dtype=torch.long)
        lt = torch.tensor(lens[tr], dtype=torch.long)
        Xv = torch.tensor(Xva_f.reshape(nva, sl, nf).astype(np.float32)).to(device)
        yv = torch.tensor(y[va], dtype=torch.long).to(device)
        lv = torch.tensor(lens[va], dtype=torch.long)

        loader = DataLoader(TensorDataset(Xt, yt, lt), batch_size=64, shuffle=True)
        cc = np.bincount(y[tr], minlength=4).astype(float)
        cw = 1.0 / (cc + 1e-6); cw = cw / cw.sum() * 4
        wt = torch.tensor(cw, dtype=torch.float32).to(device)

        set_seed()
        model = LSTMAttn(nf).to(device)
        crit = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

        best_vl, pat, best_state = float('inf'), 0, None
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
            if vl < best_vl:
                best_vl = vl; pat = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= 20:
                    print(f"      Early stop ep {ep+1}", flush=True)
                    break

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_proba[va] = torch.softmax(model(Xv, lv), dim=1).cpu().numpy()

    pred = np.argmax(oof_proba, axis=1)
    acc = accuracy_score(y, pred)
    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')
    y3 = np.array([LABEL_MAP_4_TO_3[i] for i in y])
    p3 = np.zeros((len(y), 3))
    p3[:, 0] = oof_proba[:, 0]; p3[:, 1] = oof_proba[:, 1]
    p3[:, 2] = oof_proba[:, 2] + oof_proba[:, 3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')
    y2 = np.array([LABEL_MAP_4_TO_2[i] for i in y])
    auc2 = roc_auc_score(y2, oof_proba[:, 1] + oof_proba[:, 2] + oof_proba[:, 3])

    print(f"\n  {label}: acc={acc*100:.1f}% | 4-AUC={auc4:.4f} | 3-AUC={auc3:.4f} | 2-AUC={auc2:.4f}")
    print(f"\n{classification_report(y, pred, target_names=CLASS_NAMES_4, digits=3)}", flush=True)
    return acc, auc4, auc3, auc2


# =============================================================================
# STEP 3: EVALUATE — Option C only (reference) vs Option C + D (combined)
# =============================================================================

print("\n" + "=" * 70)
print("EVALUATING: Original(22) + Option C(19) + Option D(10) = 51 feat/day")
print("=" * 70, flush=True)

X_cd = np.concatenate([X_orig, X_c, X_d], axis=2)
print(f"Combined shape: {X_cd.shape}", flush=True)
np.save(FEATURES_DIR / "X_option_cd.npy", X_cd)

eval_lstm(X_cd, y, lengths, "Orig+C+D (51feat)")

print("\nDone.")
