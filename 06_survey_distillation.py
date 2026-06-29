"""
Survey-Augmented Knowledge Distillation
========================================
Adds ~25 non-invasive survey features (demographics, lifestyle, comorbidities,
mental health, diet, family history) as static features alongside the 41 daily
wearable features. Uses a hybrid LSTM: temporal branch for wearable sequences +
static branch for survey features, fused before the classification head.

Then retrains with knowledge distillation from the CGM teacher.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
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
                    CLASS_NAMES_4, set_seed, DATASET_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

FEATURES_DIR = Path("outputs/features")
SURVEY_DIR = DATASET_PATH / "survey"


# =========================================================================
# STEP 1: EXTRACT SURVEY FEATURES
# =========================================================================

print("\n" + "=" * 70)
print("STEP 1: EXTRACTING SURVEY FEATURES")
print("=" * 70, flush=True)

person_ids = np.load(FEATURES_DIR / "person_ids.npy")
y = np.load(FEATURES_DIR / "y.npy")
N = len(person_ids)
print(f"Cohort: {N} participants")

obs = pd.read_csv(SURVEY_DIR / "observation.csv")
participants = pd.read_csv(DATASET_PATH / "participants.tsv", sep='\t')

# Build a lookup: person_id -> observation_source_value -> value_as_number
obs_pivot = obs.pivot_table(
    index='person_id',
    columns='observation_source_value',
    values='value_as_number',
    aggfunc='first'
)

SENTINEL_VALUES = {777.0, 555.0, 888.0, 999.0, 99.0}

def get_feat(pid, col_prefix):
    """Get numeric value for a person, returning NaN for sentinel/missing."""
    matching = [c for c in obs_pivot.columns if c.startswith(col_prefix + ',') or c == col_prefix]
    if not matching:
        return np.nan
    if pid not in obs_pivot.index:
        return np.nan
    val = obs_pivot.loc[pid, matching[0]]
    if pd.isna(val) or val in SENTINEL_VALUES:
        return np.nan
    return float(val)


# Define features to extract
# CAREFUL: exclude anything that leaks diabetes diagnosis labels
SURVEY_FEATURES = [
    # Demographics
    ('age',              'cage'),
    ('education_years',  'years_of_education'),
    # Lifestyle — smoking
    ('smoking_ever',     'susmkncf'),        # binary
    ('smoking_now',      'susmkcdur'),        # binary (subset)
    # Lifestyle — alcohol
    ('alcohol_ever',     'sualckncf'),        # binary
    # Mental health
    ('cesd_score',       'cestl'),            # CES-D depression 0-30
    ('sleep_restless',   'ces7'),             # 0-3
    ('paid_score',       'paidscore'),        # diabetes distress 0-100
    # Medications (non-diabetes)
    ('sleeping_pills',   'cm_slp'),           # 0-4
    # Diet
    ('diet_score',       'dietscore'),        # composite 0-17
    ('diet_fast_food',   'diet1'),            # 0-2 frequency
    ('diet_beans',       'diet5'),            # 0-2
    ('diet_regular_food','diet6'),            # 0-2
    ('diet_desserts',    'diet7'),            # 0-2
    ('diet_fats',        'diet8'),            # 0-2
    # Family history (risk factors, not label leakage)
    ('fh_diabetes_parent', 'fh_dm2pt'),       # binary
    ('fh_diabetes_sibling','fh_dm2sb'),        # binary
    # Self-reported comorbidities (non-diabetes)
    ('has_hypertension', 'mhoccur_hbp'),      # binary
    ('has_obesity',      'mhoccur_obs'),       # binary
    ('has_high_cholesterol', 'mhoccur_clsh'),  # binary
    ('has_heart_attack', 'mhoccur_mi'),        # binary
    ('has_stroke',       'mhoccur_strk'),      # binary
    ('has_kidney_problems', 'mhoccur_rnl'),    # binary
    ('has_circulation',  'mhoccur_circ'),      # binary
    # Vision difficulty (non-invasive survey)
    ('vision_difficulty','via1'),              # 1-5
    # Food insecurity
    ('food_insecurity_1','pxfi1'),            # 0-2
    ('food_insecurity_2','pxfi2'),            # 0-2
]

SURVEY_FEATURE_NAMES = [name for name, _ in SURVEY_FEATURES]
N_SURVEY = len(SURVEY_FEATURE_NAMES)

print(f"Extracting {N_SURVEY} survey features per participant...")

X_survey = np.full((N, N_SURVEY), np.nan, dtype=np.float32)

for i, pid in enumerate(person_ids):
    for j, (name, col_prefix) in enumerate(SURVEY_FEATURES):
        X_survey[i, j] = get_feat(int(pid), col_prefix)

# Report coverage
print(f"\nSurvey feature coverage (out of {N}):")
for j, name in enumerate(SURVEY_FEATURE_NAMES):
    valid = np.sum(~np.isnan(X_survey[:, j]))
    print(f"  {name:25s}: {valid:5d}/{N}  ({valid/N*100:.1f}%)")

coverage = np.mean(~np.isnan(X_survey))
print(f"\nOverall coverage: {coverage*100:.1f}%")

np.save(FEATURES_DIR / "X_survey.npy", X_survey)
print(f"Saved X_survey.npy: {X_survey.shape}", flush=True)


# =========================================================================
# STEP 2: MODEL WITH STATIC BRANCH
# =========================================================================

print("\n" + "=" * 70)
print("STEP 2: HYBRID LSTM — TEMPORAL + STATIC BRANCHES")
print("=" * 70, flush=True)

X_orig = np.load(FEATURES_DIR / "X.npy")
X_c = np.load(FEATURES_DIR / "X_option_c.npy")
X_cgm = np.load(FEATURES_DIR / "X_cgm.npy")
cgm_mask = np.load(FEATURES_DIR / "cgm_mask.npy").astype(bool)
lengths = np.load(FEATURES_DIR / "lengths.npy")
teacher_oof = np.load(FEATURES_DIR / "teacher_oof_proba.npy")

X_wear = np.concatenate([X_orig, X_c], axis=2)  # (1586, 14, 41)
X_cgm_filled = np.nan_to_num(X_cgm, nan=0.0)
X_teacher_input = np.concatenate([X_wear, X_cgm_filled], axis=2)  # (1586, 14, 49)

NF_WEAR = X_wear.shape[2]      # 41
NF_TEACH = X_teacher_input.shape[2]  # 49
SEQ_LEN = X_wear.shape[1]      # 14

print(f"X_wear:   {X_wear.shape}")
print(f"X_survey: {X_survey.shape}")
print(f"Teacher OOF: {teacher_oof.shape}")
print(f"Classes: {np.bincount(y)}", flush=True)


class HybridLSTMAttn(nn.Module):
    """LSTM with attention for temporal features + static branch for survey data."""
    def __init__(self, seq_input_size, static_size, hidden=64, layers=2, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(seq_input_size, hidden, layers,
                            dropout=drop if layers > 1 else 0,
                            batch_first=True)
        self.attn = nn.Linear(hidden, 1)

        self.static_net = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.ReLU(),
            nn.Dropout(drop / 2),
        )

        fused_size = hidden + 32
        self.head = nn.Sequential(
            nn.BatchNorm1d(fused_size), nn.Dropout(drop),
            nn.Linear(fused_size, 64), nn.ReLU(),
            nn.Dropout(drop / 2), nn.Linear(64, 4),
        )

    def forward(self, x_seq, x_static, lens=None):
        if lens is not None:
            pk = nn.utils.rnn.pack_padded_sequence(
                x_seq, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(pk)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x_seq)
        scores = self.attn(out).squeeze(-1)
        if lens is not None:
            mask = (torch.arange(out.size(1), device=out.device).unsqueeze(0)
                    >= lens.unsqueeze(1).to(out.device))
            scores = scores.masked_fill(mask, float('-inf'))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        temporal = (out * w).sum(dim=1)

        static = self.static_net(x_static)
        fused = torch.cat([temporal, static], dim=1)
        return self.head(fused)


class LSTMAttn(nn.Module):
    """Original LSTM for teacher (no static branch needed)."""
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
    acc = accuracy_score(y, pred)
    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')
    y3 = np.array([LABEL_MAP_4_TO_3[i] for i in y])
    p3 = np.zeros((len(y), 3))
    p3[:, 0] = oof_proba[:, 0]; p3[:, 1] = oof_proba[:, 1]
    p3[:, 2] = oof_proba[:, 2] + oof_proba[:, 3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')
    y2 = np.array([LABEL_MAP_4_TO_2[i] for i in y])
    auc2 = roc_auc_score(y2, oof_proba[:, 1] + oof_proba[:, 2] + oof_proba[:, 3])
    print(f"\n  {label}: acc={acc*100:.1f}%  4-AUC={auc4:.4f}  3-AUC={auc3:.4f}  2-AUC={auc2:.4f}")
    print(classification_report(y, pred, target_names=CLASS_NAMES_4, digits=3), flush=True)
    return acc, auc4, auc3, auc2


def _class_weights(y_train):
    cc = np.bincount(y_train, minlength=4).astype(float)
    cw = 1.0 / (cc + 1e-6); cw = cw / cw.sum() * 4
    return torch.tensor(cw, dtype=torch.float32).to(device)


def _prep_fold_hybrid(X_seq, X_static, y, lengths, tr, va):
    """Impute + scale both temporal and static features per fold."""
    nf_seq = X_seq.shape[2]
    nf_stat = X_static.shape[1]
    ntr, nva = len(tr), len(va)

    # Temporal
    Xtr_s = X_seq[tr].reshape(-1, nf_seq).copy()
    Xva_s = X_seq[va].reshape(-1, nf_seq).copy()
    imp_s = SimpleImputer(strategy='median')
    Xtr_s = imp_s.fit_transform(Xtr_s)
    Xva_s = imp_s.transform(Xva_s)
    sc_s = StandardScaler()
    Xtr_s = sc_s.fit_transform(Xtr_s)
    Xva_s = sc_s.transform(Xva_s)

    # Static
    Xtr_st = X_static[tr].copy()
    Xva_st = X_static[va].copy()
    imp_st = SimpleImputer(strategy='median')
    Xtr_st = imp_st.fit_transform(Xtr_st)
    Xva_st = imp_st.transform(Xva_st)
    sc_st = StandardScaler()
    Xtr_st = sc_st.fit_transform(Xtr_st)
    Xva_st = sc_st.transform(Xva_st)

    Xt_seq = torch.tensor(Xtr_s.reshape(ntr, SEQ_LEN, nf_seq).astype(np.float32))
    Xv_seq = torch.tensor(Xva_s.reshape(nva, SEQ_LEN, nf_seq).astype(np.float32))
    Xt_stat = torch.tensor(Xtr_st.astype(np.float32))
    Xv_stat = torch.tensor(Xva_st.astype(np.float32))
    yt = torch.tensor(y[tr], dtype=torch.long)
    yv = torch.tensor(y[va], dtype=torch.long)
    lt = torch.tensor(lengths[tr], dtype=torch.long)
    lv = torch.tensor(lengths[va], dtype=torch.long)
    return Xt_seq, Xt_stat, yt, lt, Xv_seq, Xv_stat, yv, lv


def _prep_fold(X_seq, y, lengths, tr, va):
    """Impute + scale a fold (temporal only)."""
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


def distillation_loss(student_logits, hard_labels, teacher_probs, T, alpha, cls_crit):
    l_hard = cls_crit(student_logits, hard_labels)
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = torch.tensor(teacher_probs, dtype=torch.float32, device=student_logits.device)
    teacher_log = torch.log(teacher_soft + 1e-8)
    teacher_tempered = F.softmax(teacher_log / T, dim=1)
    l_soft = F.kl_div(student_log_soft, teacher_tempered, reduction='batchmean') * (T * T)
    return alpha * l_hard + (1 - alpha) * l_soft


# =========================================================================
# EXPERIMENT A: Re-train Teacher with survey features (hybrid)
# =========================================================================

print("\n" + "=" * 70)
print("EXPERIMENT A: TEACHER + SURVEY (hybrid, wearable+CGM+survey)")
print("=" * 70, flush=True)

set_seed()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
teacher_survey_oof = np.zeros((len(y), 4))

for fold, (tr, va) in enumerate(skf.split(X_teacher_input, y)):
    print(f"  Teacher+Survey fold {fold+1}/{N_FOLDS}...", flush=True)
    (Xt_seq, Xt_stat, yt, lt,
     Xv_seq, Xv_stat, yv, lv) = _prep_fold_hybrid(
        X_teacher_input, X_survey, y, lengths, tr, va)
    Xv_seq, Xv_stat, yv = Xv_seq.to(device), Xv_stat.to(device), yv.to(device)

    loader = DataLoader(
        TensorDataset(Xt_seq, Xt_stat, yt, lt), batch_size=64, shuffle=True)
    wt = _class_weights(y[tr])

    set_seed()
    model = HybridLSTMAttn(
        seq_input_size=NF_TEACH, static_size=N_SURVEY).to(device)
    crit = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

    best_vl, pat, best_state = float('inf'), 0, None
    for ep in range(100):
        model.train()
        for Xb_seq, Xb_stat, yb, lb in loader:
            Xb_seq, Xb_stat, yb = Xb_seq.to(device), Xb_stat.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb_seq, Xb_stat, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv_seq, Xv_stat, lv), yv).item()
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
        teacher_survey_oof[va] = torch.softmax(
            model(Xv_seq, Xv_stat, lv), dim=1).cpu().numpy()

np.save(FEATURES_DIR / "teacher_survey_oof_proba.npy", teacher_survey_oof)
t_acc, t_auc4, t_auc3, t_auc2 = compute_metrics(
    y, teacher_survey_oof, "TEACHER+SURVEY")


# =========================================================================
# EXPERIMENT B: Student (wearable+survey) with distillation from ORIGINAL teacher
# =========================================================================

print("\n" + "=" * 70)
print("EXPERIMENT B: STUDENT (wearable+survey) — distilled from ORIGINAL teacher")
print("=" * 70, flush=True)


def run_hybrid_student_cv(X_seq, X_static, y, lengths, teacher_oof_probs,
                          T=2.0, alpha=0.3, label="HybridStudent"):
    set_seed()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 4))

    nf_seq = X_seq.shape[2]
    nf_stat = X_static.shape[1]

    print(f"\n  [{label}]  T={T}  alpha={alpha}  seq={nf_seq}  static={nf_stat}", flush=True)

    for fold, (tr, va) in enumerate(skf.split(X_seq, y)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", flush=True)
        (Xt_seq, Xt_stat, yt, lt,
         Xv_seq, Xv_stat, yv, lv) = _prep_fold_hybrid(
            X_seq, X_static, y, lengths, tr, va)
        Xv_seq, Xv_stat, yv = Xv_seq.to(device), Xv_stat.to(device), yv.to(device)

        teacher_tr = teacher_oof_probs[tr]
        loader = DataLoader(
            TensorDataset(Xt_seq, Xt_stat, yt, lt,
                          torch.tensor(teacher_tr, dtype=torch.float32)),
            batch_size=64, shuffle=True)
        wt = _class_weights(y[tr])

        set_seed()
        model = HybridLSTMAttn(
            seq_input_size=nf_seq, static_size=nf_stat).to(device)
        cls_crit = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

        best_vl, pat, best_state = float('inf'), 0, None
        for ep in range(100):
            model.train()
            for Xb_seq, Xb_stat, yb, lb, tb in loader:
                Xb_seq, Xb_stat, yb = Xb_seq.to(device), Xb_stat.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(Xb_seq, Xb_stat, lb)
                loss = distillation_loss(logits, yb, tb.numpy(), T, alpha, cls_crit)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = cls_crit(model(Xv_seq, Xv_stat, lv), yv).item()
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
            oof_proba[va] = torch.softmax(
                model(Xv_seq, Xv_stat, lv), dim=1).cpu().numpy()

    acc, auc4, auc3, auc2 = compute_metrics(y, oof_proba, label)
    np.save(FEATURES_DIR / f"oof_{label.replace(' ', '_')}.npy", oof_proba)
    return acc, auc4, auc3, auc2


# B1: Best config from previous distillation (T=2, alpha=0.3) with original teacher
b1 = run_hybrid_student_cv(
    X_wear, X_survey, y, lengths, teacher_oof,
    T=2.0, alpha=0.3, label="B1_wear+survey_origTeach")

# B2: Same but with updated teacher+survey OOF
b2 = run_hybrid_student_cv(
    X_wear, X_survey, y, lengths, teacher_survey_oof,
    T=2.0, alpha=0.3, label="B2_wear+survey_survTeach")


# =========================================================================
# EXPERIMENT C: Sweep T and alpha for best hybrid student
# =========================================================================

print("\n" + "=" * 70)
print("EXPERIMENT C: HYPERPARAMETER SWEEP (hybrid student)")
print("=" * 70, flush=True)

# Use whichever teacher was better
best_teacher_oof = teacher_survey_oof if t_auc4 > 0.7472 else teacher_oof
teacher_label = "survey_teacher" if t_auc4 > 0.7472 else "orig_teacher"
print(f"Using {teacher_label} (4-AUC={max(t_auc4, 0.7472):.4f})")

results = {}
for T in [2.0, 3.0, 4.0]:
    for alpha in [0.3, 0.5, 0.7]:
        label = f"C_T{T:.0f}_a{alpha}"
        acc, auc4, auc3, auc2 = run_hybrid_student_cv(
            X_wear, X_survey, y, lengths, best_teacher_oof,
            T=T, alpha=alpha, label=label)
        results[(T, alpha)] = dict(acc=acc, auc4=auc4, auc3=auc3, auc2=auc2)
        print("-" * 60, flush=True)


# =========================================================================
# EXPERIMENT D: ENSEMBLE — average previous best + hybrid best
# =========================================================================

print("\n" + "=" * 70)
print("EXPERIMENT D: ENSEMBLE")
print("=" * 70, flush=True)

prev_best_oof = np.load(FEATURES_DIR / "oof_T2_a0.3.npy")  # previous best student

best_key = max(results, key=lambda k: results[k]['auc4'])
best_hybrid_label = f"C_T{best_key[0]:.0f}_a{best_key[1]}"
best_hybrid_oof = np.load(FEATURES_DIR / f"oof_{best_hybrid_label}.npy")

for w in [0.3, 0.5, 0.7]:
    ens = w * best_hybrid_oof + (1 - w) * prev_best_oof
    compute_metrics(y, ens, f"Ensemble w_hybrid={w:.1f}")


# =========================================================================
# FINAL COMPARISON
# =========================================================================

print("\n" + "=" * 70)
print("FULL RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Model':<42} {'Acc':>7} {'4-AUC':>7} {'3-AUC':>7} {'2-AUC':>7}")
print("-" * 75)
print(f"{'Option C LSTM (prev best no-distill)':<42} {'33.9%':>7} {'0.6725':>7} {'0.6706':>7} {'0.6632':>7}")
print(f"{'Distilled T2a0.3 (prev best)':<42} {'37.6%':>7} {'0.6879':>7} {'0.6858':>7} {'0.6846':>7}")
print(f"{'Teacher (wear+CGM)':<42} {'45.4%':>7} {'0.7472':>7} {'—':>7} {'—':>7}")
print(f"{'Teacher+Survey (wear+CGM+survey)':<42} {t_acc*100:>6.1f}% {t_auc4:>7.4f} {t_auc3:>7.4f} {t_auc2:>7.4f}")
print()

print("Hybrid Students (wearable + survey, distilled):")
print("-" * 75)
print(f"{'  B1: orig teacher':<42} {b1[0]*100:>6.1f}% {b1[1]:>7.4f} {b1[2]:>7.4f} {b1[3]:>7.4f}")
print(f"{'  B2: survey teacher':<42} {b2[0]*100:>6.1f}% {b2[1]:>7.4f} {b2[2]:>7.4f} {b2[3]:>7.4f}")
print()

best_key = max(results, key=lambda k: results[k]['auc4'])
for (T, alpha), r in sorted(results.items()):
    tag = "  << BEST" if (T, alpha) == best_key else ""
    name = f"  C: T={T:.0f} alpha={alpha}"
    print(f"{name:<42} {r['acc']*100:>6.1f}% {r['auc4']:>7.4f} {r['auc3']:>7.4f} {r['auc2']:>7.4f}{tag}")

best_r = results[best_key]
print(f"\nBest hybrid student: T={best_key[0]:.0f}, alpha={best_key[1]}")
print(f"  vs prev best (0.6879):   {best_r['auc4']-0.6879:+.4f}")
print(f"  vs Teacher:              {best_r['auc4']-0.7472:+.4f}")

print("\nDone.", flush=True)
