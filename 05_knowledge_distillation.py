"""
Knowledge Distillation: Teacher (wearable+CGM) → Student (wearable only)
========================================================================
Step 1: Train Teacher with wearable(41) + CGM(8) = 49 features/day
        → save 5-fold OOF soft predictions
Step 2: Train Student with wearable(41) only
        → distillation loss: α × CE(hard) + (1-α) × KL(student/T, teacher/T)
Step 3: Sweep α ∈ {0.3, 0.5, 0.7}, T ∈ {2, 3, 4}
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pathlib import Path

from config import (RANDOM_STATE, N_FOLDS, LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2,
                    CLASS_NAMES_4, set_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

FEATURES_DIR = Path("outputs/features")

# Load data
X_orig = np.load(FEATURES_DIR / "X.npy")
X_c    = np.load(FEATURES_DIR / "X_option_c.npy")
X_cgm  = np.load(FEATURES_DIR / "X_cgm.npy")
cgm_mask = np.load(FEATURES_DIR / "cgm_mask.npy").astype(bool)
y       = np.load(FEATURES_DIR / "y.npy")
lengths = np.load(FEATURES_DIR / "lengths.npy")

X_wear = np.concatenate([X_orig, X_c], axis=2)  # (1586, 14, 41)

# For teacher: concat wearable + CGM = 49 features/day
# Replace CGM NaNs with 0 for teacher input (masked days get zero signal)
X_cgm_filled = np.nan_to_num(X_cgm, nan=0.0)
X_teacher = np.concatenate([X_wear, X_cgm_filled], axis=2)  # (1586, 14, 49)

print(f"X_wear   : {X_wear.shape}")
print(f"X_teacher: {X_teacher.shape}")
print(f"X_cgm    : {X_cgm.shape}  valid days: {cgm_mask.mean()*100:.1f}%")
print(f"y        : {np.bincount(y)}", flush=True)

NF_WEAR = X_wear.shape[2]     # 41
NF_TEACH = X_teacher.shape[2]  # 49
SEQ_LEN = X_wear.shape[1]      # 14


# =========================================================================
# MODEL: Same proven architecture for both teacher and student
# =========================================================================

class LSTMAttn(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            dropout=drop if layers > 1 else 0,
                            batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden), nn.Dropout(drop),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(drop / 2), nn.Linear(64, 4),
        )

    def forward(self, x, lens=None):
        if lens is not None:
            pk = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(pk)
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


def compute_metrics(y, oof_proba, label=""):
    pred = np.argmax(oof_proba, axis=1)
    acc  = accuracy_score(y, pred)
    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')
    y3   = np.array([LABEL_MAP_4_TO_3[i] for i in y])
    p3   = np.zeros((len(y), 3))
    p3[:,0] = oof_proba[:,0]; p3[:,1] = oof_proba[:,1]
    p3[:,2] = oof_proba[:,2] + oof_proba[:,3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')
    y2   = np.array([LABEL_MAP_4_TO_2[i] for i in y])
    auc2 = roc_auc_score(y2, oof_proba[:,1]+oof_proba[:,2]+oof_proba[:,3])
    print(f"\n  {label}: acc={acc*100:.1f}%  4-AUC={auc4:.4f}  3-AUC={auc3:.4f}  2-AUC={auc2:.4f}")
    print(classification_report(y, pred, target_names=CLASS_NAMES_4, digits=3), flush=True)
    return acc, auc4, auc3, auc2


def _prep_fold(X_seq, y, lengths, tr, va):
    """Impute + scale a fold, return tensors."""
    nf = X_seq.shape[2]
    ntr, nva = len(tr), len(va)
    Xtr_f = X_seq[tr].reshape(-1, nf).copy()
    Xva_f = X_seq[va].reshape(-1, nf).copy()
    imp = SimpleImputer(strategy='median')
    Xtr_f = imp.fit_transform(Xtr_f)
    Xva_f = imp.transform(Xva_f)
    sc = StandardScaler()
    Xtr_f = sc.fit_transform(Xtr_f)
    Xva_f = sc.transform(Xva_f)
    Xt = torch.tensor(Xtr_f.reshape(ntr, SEQ_LEN, nf).astype(np.float32))
    Xv = torch.tensor(Xva_f.reshape(nva, SEQ_LEN, nf).astype(np.float32))
    yt = torch.tensor(y[tr], dtype=torch.long)
    yv = torch.tensor(y[va], dtype=torch.long)
    lt = torch.tensor(lengths[tr], dtype=torch.long)
    lv = torch.tensor(lengths[va], dtype=torch.long)
    return Xt, yt, lt, Xv, yv, lv


def _class_weights(y_train):
    cc = np.bincount(y_train, minlength=4).astype(float)
    cw = 1.0 / (cc + 1e-6); cw = cw / cw.sum() * 4
    return torch.tensor(cw, dtype=torch.float32).to(device)


# =========================================================================
# STEP 1: TRAIN TEACHER (wearable + CGM = 49 features)
# =========================================================================

print("\n" + "=" * 70)
print("STEP 1: TRAINING TEACHER (wearable + CGM = 49 features/day)")
print("=" * 70, flush=True)

set_seed()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
teacher_oof = np.zeros((len(y), 4))

for fold, (tr, va) in enumerate(skf.split(X_teacher, y)):
    print(f"  Teacher fold {fold+1}/{N_FOLDS}...", flush=True)
    Xt, yt, lt, Xv, yv, lv = _prep_fold(X_teacher, y, lengths, tr, va)
    Xv, yv = Xv.to(device), yv.to(device)

    loader = DataLoader(TensorDataset(Xt, yt, lt), batch_size=64, shuffle=True)
    wt = _class_weights(y[tr])

    set_seed()
    model = LSTMAttn(input_size=NF_TEACH).to(device)
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
                print(f"    Early stop ep {ep+1}", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        teacher_oof[va] = torch.softmax(model(Xv, lv), dim=1).cpu().numpy()

np.save(FEATURES_DIR / "teacher_oof_proba.npy", teacher_oof)
t_acc, t_auc4, t_auc3, t_auc2 = compute_metrics(y, teacher_oof, "TEACHER")


# =========================================================================
# STEP 2: TRAIN STUDENT WITH DISTILLATION
# =========================================================================

print("\n" + "=" * 70)
print("STEP 2: KNOWLEDGE DISTILLATION — STUDENT (wearable only = 41 feat)")
print("=" * 70, flush=True)


def distillation_loss(student_logits, hard_labels, teacher_probs, T, alpha, cls_crit):
    """
    Combined distillation loss:
      α × CE(student, hard_labels) + (1-α) × KL(student/T, teacher/T) × T²
    """
    # Hard label loss
    l_hard = cls_crit(student_logits, hard_labels)

    # Soft label loss (KL divergence with temperature)
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft     = torch.tensor(teacher_probs, dtype=torch.float32, device=student_logits.device)
    # Apply temperature to teacher probs: re-soften by converting to logits then back
    # Since we only have probs (not logits), use log(probs)/T then softmax
    teacher_log = torch.log(teacher_soft + 1e-8)
    teacher_tempered = F.softmax(teacher_log / T, dim=1)
    l_soft = F.kl_div(student_log_soft, teacher_tempered, reduction='batchmean') * (T * T)

    return alpha * l_hard + (1 - alpha) * l_soft


def run_student_cv(X_w, y, lengths, teacher_oof, T=3.0, alpha=0.5, label="Student"):
    set_seed()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 4))

    print(f"\n  [{label}]  T={T}  alpha={alpha}  |  {N_FOLDS}-fold CV", flush=True)

    for fold, (tr, va) in enumerate(skf.split(X_w, y)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", flush=True)
        Xt, yt, lt, Xv, yv, lv = _prep_fold(X_w, y, lengths, tr, va)
        Xv, yv = Xv.to(device), yv.to(device)

        # Teacher soft labels for this training set
        teacher_tr = teacher_oof[tr]  # (ntr, 4) — already probability distributions

        loader_data = TensorDataset(
            Xt, yt, lt,
            torch.tensor(teacher_tr, dtype=torch.float32),
        )
        loader = DataLoader(loader_data, batch_size=64, shuffle=True)
        wt = _class_weights(y[tr])

        set_seed()
        model = LSTMAttn(input_size=NF_WEAR).to(device)
        cls_crit = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

        best_vl, pat, best_state = float('inf'), 0, None
        for ep in range(100):
            model.train()
            for Xb, yb, lb, tb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(Xb, lb)
                loss = distillation_loss(logits, yb, tb.numpy(), T, alpha, cls_crit)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = cls_crit(model(Xv, lv), yv).item()
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

    acc, auc4, auc3, auc2 = compute_metrics(y, oof_proba, label)
    np.save(FEATURES_DIR / f"oof_{label.replace(' ','_')}.npy", oof_proba)
    return acc, auc4, auc3, auc2


# =========================================================================
# STEP 3: SWEEP α and T
# =========================================================================

print("\n" + "=" * 70)
print("STEP 3: HYPERPARAMETER SWEEP")
print("=" * 70, flush=True)

results = {}

for T in [2.0, 3.0, 4.0]:
    for alpha in [0.3, 0.5, 0.7]:
        label = f"T{T:.0f}_a{alpha}"
        acc, auc4, auc3, auc2 = run_student_cv(
            X_wear, y, lengths, teacher_oof,
            T=T, alpha=alpha, label=label,
        )
        results[(T, alpha)] = dict(acc=acc, auc4=auc4, auc3=auc3, auc2=auc2)
        print("-" * 60, flush=True)


# =========================================================================
# FINAL COMPARISON
# =========================================================================

print("\n" + "=" * 70)
print("FULL RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Model':<32} {'Acc':>7} {'4-AUC':>7} {'3-AUC':>7} {'2-AUC':>7}")
print("-" * 65)
print(f"{'Option C LSTM (prev best)':<32} {'33.9%':>7} {'0.6725':>7} {'0.6706':>7} {'0.6632':>7}")
print(f"{'Teacher (wear+CGM=49)':<32} {t_acc*100:>6.1f}% {t_auc4:>7.4f} {t_auc3:>7.4f} {t_auc2:>7.4f}")
print()
print("Student (wearable only, distilled):")
print("-" * 65)

best_key = max(results, key=lambda k: results[k]['auc4'])
for (T, alpha), r in sorted(results.items()):
    tag = "  << BEST" if (T, alpha) == best_key else ""
    name = f"  T={T:.0f} alpha={alpha}"
    print(f"{name:<32} {r['acc']*100:>6.1f}% {r['auc4']:>7.4f} {r['auc3']:>7.4f} {r['auc2']:>7.4f}{tag}")

best_r = results[best_key]
print(f"\nBest student: T={best_key[0]:.0f}, alpha={best_key[1]}")
print(f"  vs Option C (0.6725):  {best_r['auc4']-0.6725:+.4f}")
print(f"  vs Teacher:            {best_r['auc4']-t_auc4:+.4f}")

print("\nDone.", flush=True)
