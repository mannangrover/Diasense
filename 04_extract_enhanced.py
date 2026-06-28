"""
Extract enhanced features (Option C + B) and evaluate all models.
Optimized: loads raw JSON files once, extracts both C and B features in one pass.
"""
import sys, io, gc, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from config import (
    RANDOM_STATE, N_FOLDS, SEQUENCE_LENGTH, DATASET_PATH,
    CONTINUOUS_MODALITIES, STRESS_HIGH_THRESHOLD, SPO2_LOW_THRESHOLD,
    LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2, set_seed,
)
from src.data.cohort import build_cohort
from src.features.enhanced import (
    NUM_OPTION_C_FEATURES, NUM_OPTION_B_FEATURES,
    _hr_advanced_daily, _stress_advanced_daily, _spo2_advanced_daily,
    _sleep_advanced_daily, _activity_advanced_daily, _segment_hour,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

X_orig = np.load('outputs/features/X.npy')
y = np.load('outputs/features/y.npy')
lengths = np.load('outputs/features/lengths.npy')
pids = np.load('outputs/features/person_ids.npy')
n_participants = len(y)
print(f"Loaded {n_participants} participants, original: {X_orig.shape}", flush=True)


def _load_raw_fast(filepath, body_key, value_key, invalid_values):
    """Load raw JSON, return dict of date -> (timestamps, values)."""
    fp = Path(filepath)
    if not fp.exists():
        return {}
    with open(fp) as f:
        data = json.load(f)
    records = data["body"][body_key]
    invalid_set = set(invalid_values)
    by_date = defaultdict(lambda: ([], []))
    for r in records:
        val = r[value_key]["value"]
        if val in invalid_set:
            continue
        dt_str = r["effective_time_frame"]["date_time"].replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        ts_list, val_list = by_date[dt.date()]
        ts_list.append(dt)
        val_list.append(float(val))
    return dict(by_date)


# =====================================================================
# EXTRACT ENHANCED FEATURES
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("EXTRACTING ENHANCED FEATURES (optimized single-pass)", flush=True)
print("=" * 70, flush=True)

cohort = build_cohort(exclude_dead_sensors=True)
pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}

X_c = np.full((n_participants, SEQUENCE_LENGTH, NUM_OPTION_C_FEATURES), np.nan, dtype=np.float32)
X_b = np.full((n_participants, SEQUENCE_LENGTH, NUM_OPTION_B_FEATURES), np.nan, dtype=np.float32)

start_time = time.time()
processed = 0

for _, row in cohort.iterrows():
    pid = int(row["person_id"])
    if pid not in pid_to_idx:
        continue
    idx = pid_to_idx[pid]
    length = int(lengths[idx])
    if length == 0:
        continue

    try:
        # Load HR raw (once)
        hr_data = {}
        raw_path = row.get(CONTINUOUS_MODALITIES["heart_rate"]["filepath_col"])
        if not (raw_path is None or (isinstance(raw_path, float) and np.isnan(raw_path)) or raw_path == ""):
            hr_data = _load_raw_fast(
                str(DATASET_PATH) + str(raw_path),
                "heart_rate", "heart_rate", [0]
            )

        # Load stress raw (once)
        stress_data = {}
        raw_path = row.get(CONTINUOUS_MODALITIES["stress"]["filepath_col"])
        if not (raw_path is None or (isinstance(raw_path, float) and np.isnan(raw_path)) or raw_path == ""):
            stress_data = _load_raw_fast(
                str(DATASET_PATH) + str(raw_path),
                "stress", "stress", [-2, -1]
            )

        # Load SpO2 raw (once)
        spo2_data = {}
        raw_path = row.get(CONTINUOUS_MODALITIES["spo2"]["filepath_col"])
        if not (raw_path is None or (isinstance(raw_path, float) and np.isnan(raw_path)) or raw_path == ""):
            spo2_data = _load_raw_fast(
                str(DATASET_PATH) + str(raw_path),
                "breathing", "oxygen_saturation", [0]
            )

        # Pre-load activity file ONCE per participant
        act_by_date = defaultdict(list)  # date -> list of (activity, dur_min, steps, hour)
        act_path = row.get("physical_activity_filepath")
        if not (act_path is None or (isinstance(act_path, float) and np.isnan(act_path)) or act_path == ""):
            fp = Path(str(DATASET_PATH) + str(act_path))
            if fp.exists():
                with open(fp) as f:
                    adata = json.load(f)
                for ar in adata["body"]["activity"]:
                    if ar["activity_name"] == "":
                        continue
                    ti = ar["effective_time_frame"]["time_interval"]
                    sdt = datetime.fromisoformat(ti["start_date_time"].replace("Z", "+00:00"))
                    edt = datetime.fromisoformat(ti["end_date_time"].replace("Z", "+00:00"))
                    dur = (edt - sdt).total_seconds() / 60.0
                    sv = ar["base_movement_quantity"]["value"]
                    steps = int(sv) if sv != "" else 0
                    act_by_date[sdt.date()].append((ar["activity_name"], dur, steps, sdt.hour))

        # Pre-load sleep file ONCE per participant
        sleep_by_date = defaultdict(list)  # date -> list of (stage, dur_min)
        slp_path = row.get("sleep_filepath")
        if not (slp_path is None or (isinstance(slp_path, float) and np.isnan(slp_path)) or slp_path == ""):
            fp = Path(str(DATASET_PATH) + str(slp_path))
            if fp.exists():
                with open(fp) as f:
                    sdata = json.load(f)
                for sr in sdata["body"]["sleep"]:
                    ti = sr["effective_time_frame"]["time_interval"]
                    sdt = datetime.fromisoformat(ti["start_date_time"].replace("Z", "+00:00"))
                    edt = datetime.fromisoformat(ti["end_date_time"].replace("Z", "+00:00"))
                    dur = (edt - sdt).total_seconds() / 60.0
                    sleep_by_date[sdt.date()].append((sr["sleep_stage_state"], dur))

        # Get dates
        all_raw_dates = set()
        all_raw_dates.update(hr_data.keys())
        all_raw_dates.update(stress_data.keys())
        all_raw_dates.update(spo2_data.keys())
        dates = sorted(all_raw_dates)[:SEQUENCE_LENGTH]

        if not dates:
            processed += 1
            continue

        def _sm(d, s):
            return np.mean(d[s]) if len(d[s]) >= 3 else np.nan

        for d_i, date in enumerate(dates):
            if d_i >= length:
                break

            # === OPTION C ===
            hr_ts, hr_vals = hr_data.get(date, ([], []))
            c_hr = _hr_advanced_daily(hr_ts, hr_vals, date)

            st_ts, st_vals = stress_data.get(date, ([], []))
            c_stress = _stress_advanced_daily(st_ts, st_vals, date)

            sp_ts, sp_vals = spo2_data.get(date, ([], []))
            c_spo2 = _spo2_advanced_daily(sp_ts, sp_vals, date)

            # Activity advanced (from pre-loaded data)
            day_acts = act_by_date.get(date, [])
            if day_acts:
                sed_bouts = [dur for act, dur, st, hr in day_acts if act == "sedentary"]
                active_bouts = [dur for act, dur, st, hr in day_acts if act != "sedentary"]
                total_steps = sum(st for act, dur, st, hr in day_acts if act != "sedentary")
                total_active = sum(dur for act, dur, st, hr in day_acts if act != "sedentary")
                c_act = [
                    float(len(sed_bouts)),
                    max(sed_bouts) if sed_bouts else 0.0,
                    float(len(active_bouts)),
                    total_steps / total_active if total_active > 0 else 0.0,
                ]
            else:
                c_act = [np.nan] * 4

            # Sleep advanced (from pre-loaded data)
            day_sleep = sleep_by_date.get(date, [])
            if day_sleep:
                sleep_min = sum(d for s, d in day_sleep if s != "awake")
                awake_min = sum(d for s, d in day_sleep if s == "awake")
                total_min = sleep_min + awake_min
                awake_count = sum(1 for s, d in day_sleep if s == "awake")
                c_sleep = [
                    sleep_min / total_min * 100 if total_min > 0 else np.nan,
                    awake_min,
                    awake_count / (sleep_min / 60.0) if sleep_min > 0 else np.nan,
                ]
            else:
                c_sleep = [np.nan] * 3

            X_c[idx, d_i, :] = c_hr + c_stress + c_spo2 + c_act + c_sleep

            # === OPTION B (segment-level) ===
            hr_segs = defaultdict(list)
            for t, v in zip(hr_ts, hr_vals):
                hr_segs[_segment_hour(t.hour)].append(v)

            stress_segs = defaultdict(list)
            for t, v in zip(st_ts, st_vals):
                stress_segs[_segment_hour(t.hour)].append(v)

            spo2_segs = defaultdict(list)
            for t, v in zip(sp_ts, sp_vals):
                spo2_segs[_segment_hour(t.hour)].append(v)

            # Steps by segment (from pre-loaded activity)
            steps_morning = sum(st for act, dur, st, hr in day_acts if act != "sedentary" and 6 <= hr < 12)
            steps_afternoon = sum(st for act, dur, st, hr in day_acts if act != "sedentary" and 12 <= hr < 18)

            day_spo2 = [v for s in [1,2,3] for v in spo2_segs.get(s, [])]

            X_b[idx, d_i, :] = [
                _sm(hr_segs, 0), _sm(hr_segs, 1), _sm(hr_segs, 2), _sm(hr_segs, 3),
                _sm(stress_segs, 0), _sm(stress_segs, 1), _sm(stress_segs, 2), _sm(stress_segs, 3),
                _sm(spo2_segs, 0),
                np.mean(day_spo2) if len(day_spo2) >= 3 else np.nan,
                float(steps_morning), float(steps_afternoon),
            ]

    except Exception as e:
        if processed < 5:
            print(f"  ERROR PID {pid}: {e}", flush=True)

    processed += 1
    if processed % 200 == 0:
        elapsed = time.time() - start_time
        rate = processed / elapsed
        remaining = (n_participants - processed) / rate / 60
        print(f"  {processed}/{n_participants} ({elapsed:.0f}s, ~{remaining:.1f}min left)", flush=True)
        gc.collect()

elapsed = time.time() - start_time
print(f"\nExtraction done: {processed} in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

# Check coverage
c_valid = (~np.isnan(X_c)).any(axis=2).any(axis=1).sum()
b_valid = (~np.isnan(X_b)).any(axis=2).any(axis=1).sum()
print(f"  Option C: {c_valid}/{n_participants} participants have data", flush=True)
print(f"  Option B: {b_valid}/{n_participants} participants have data", flush=True)

np.save('outputs/features/X_option_c.npy', X_c)
np.save('outputs/features/X_option_b.npy', X_b)
print("Saved enhanced features.", flush=True)


# =====================================================================
# EVALUATION HELPERS
# =====================================================================

def flatten(X_seq, lens):
    n, _, nf = X_seq.shape
    Xm = np.zeros((n, nf), dtype=np.float32)
    Xs = np.zeros((n, nf), dtype=np.float32)
    for i in range(n):
        Xm[i] = np.nanmean(X_seq[i, :lens[i], :], axis=0)
        Xs[i] = np.nanstd(X_seq[i, :lens[i], :], axis=0)
    return np.hstack([Xm, Xs])


def eval_rf(X_flat, y, label):
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X_flat)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), 4))
    for _, (tr, va) in enumerate(skf.split(X_imp, y)):
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X_imp[tr]), sc.transform(X_imp[va])
        m = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                   class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(Xtr, y[tr])
        oof_pred[va] = m.predict(Xva)
        p = m.predict_proba(Xva)
        for j, c in enumerate(m.classes_):
            oof_proba[va, c] = p[:, j]
    acc = accuracy_score(y, oof_pred)
    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')
    y3 = np.array([LABEL_MAP_4_TO_3[i] for i in y])
    p3 = np.zeros((len(y),3)); p3[:,0]=oof_proba[:,0]; p3[:,1]=oof_proba[:,1]; p3[:,2]=oof_proba[:,2]+oof_proba[:,3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')
    y2 = np.array([LABEL_MAP_4_TO_2[i] for i in y])
    auc2 = roc_auc_score(y2, oof_proba[:,1]+oof_proba[:,2]+oof_proba[:,3])
    print(f"  {label} RF: acc={acc*100:.1f}% | 4-AUC={auc4:.4f} | 3-AUC={auc3:.4f} | 2-AUC={auc2:.4f}", flush=True)
    return acc, auc4, auc3, auc2


class LSTMAttn(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, drop=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, dropout=drop if layers>1 else 0, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.BatchNorm1d(hidden), nn.Dropout(drop),
                                  nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(drop/2), nn.Linear(64, 4))
    def forward(self, x, lens=None):
        if lens is not None:
            pk = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(pk); out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        sc = self.attn(out).squeeze(-1)
        if lens is not None:
            mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) >= lens.unsqueeze(1).to(out.device)
            sc = sc.masked_fill(mask, float('-inf'))
        w = torch.softmax(sc, dim=1).unsqueeze(-1)
        return self.head((out * w).sum(dim=1))


def eval_lstm(X_seq, y, lens, label):
    set_seed()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 4))
    nf = X_seq.shape[2]
    for _, (tr, va) in enumerate(skf.split(X_seq, y)):
        ntr, sl, nva = len(tr), X_seq.shape[1], len(va)
        Xtr2 = X_seq[tr].reshape(-1, nf).copy(); Xva2 = X_seq[va].reshape(-1, nf).copy()
        imp = SimpleImputer(strategy='median'); Xtr2 = imp.fit_transform(Xtr2); Xva2 = imp.transform(Xva2)
        sc = StandardScaler(); Xtr2 = sc.fit_transform(Xtr2); Xva2 = sc.transform(Xva2)
        Xt = torch.tensor(Xtr2.reshape(ntr,sl,nf).astype(np.float32))
        yt = torch.tensor(y[tr], dtype=torch.long); lt = torch.tensor(lens[tr], dtype=torch.long)
        Xv = torch.tensor(Xva2.reshape(nva,sl,nf).astype(np.float32)).to(device)
        yv = torch.tensor(y[va], dtype=torch.long).to(device); lv = torch.tensor(lens[va], dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt, lt), batch_size=64, shuffle=True)
        cc = np.bincount(y[tr], minlength=4).astype(float); cw = 1.0/(cc+1e-6); cw = cw/cw.sum()*4
        wt = torch.tensor(cw, dtype=torch.float32).to(device)
        set_seed(); model = LSTMAttn(nf).to(device)
        crit = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)
        best_vl, pat = float('inf'), 0
        for ep in range(80):
            model.train()
            for Xb, yb, lb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad(); loss = crit(model(Xb, lb), yb); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad(): vl = crit(model(Xv, lv), yv).item()
            sched.step(vl)
            if vl < best_vl:
                best_vl = vl; pat = 0; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= 20: break
        model.load_state_dict(bs); model.eval()
        with torch.no_grad(): oof_proba[va] = torch.softmax(model(Xv, lv), dim=1).cpu().numpy()
    pred = np.argmax(oof_proba, axis=1)
    acc = accuracy_score(y, pred)
    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')
    y3 = np.array([LABEL_MAP_4_TO_3[i] for i in y])
    p3 = np.zeros((len(y),3)); p3[:,0]=oof_proba[:,0]; p3[:,1]=oof_proba[:,1]; p3[:,2]=oof_proba[:,2]+oof_proba[:,3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')
    y2 = np.array([LABEL_MAP_4_TO_2[i] for i in y])
    auc2 = roc_auc_score(y2, oof_proba[:,1]+oof_proba[:,2]+oof_proba[:,3])
    print(f"  {label} LSTM: acc={acc*100:.1f}% | 4-AUC={auc4:.4f} | 3-AUC={auc3:.4f} | 2-AUC={auc2:.4f}", flush=True)
    return acc, auc4, auc3, auc2


# =====================================================================
# RUN ALL STAGES
# =====================================================================
results = []

print("\n" + "=" * 70, flush=True)
print("STAGE 0: ORIGINAL 22 FEATURES", flush=True)
print("=" * 70, flush=True)
Xf0 = flatten(X_orig, lengths)
r = eval_rf(Xf0, y, "Orig(22)"); results.append(("Original(22)", "RF", *r))
r = eval_lstm(X_orig, y, lengths, "Orig(22)"); results.append(("Original(22)", "LSTM", *r))

print("\n" + "=" * 70, flush=True)
print("STAGE 1: OPTION C — 22 + 19 = 41 features/day", flush=True)
print("=" * 70, flush=True)
Xc = np.concatenate([X_orig, X_c], axis=2)
print(f"  Shape: {Xc.shape}", flush=True)
Xf1 = flatten(Xc, lengths)
r = eval_rf(Xf1, y, "OptC(41)"); results.append(("+OptionC(41)", "RF", *r))
r = eval_lstm(Xc, y, lengths, "OptC(41)"); results.append(("+OptionC(41)", "LSTM", *r))

print("\n" + "=" * 70, flush=True)
print("STAGE 2: OPTION B — 22 + 12 = 34 features/day", flush=True)
print("=" * 70, flush=True)
Xb = np.concatenate([X_orig, X_b], axis=2)
print(f"  Shape: {Xb.shape}", flush=True)
Xf2 = flatten(Xb, lengths)
r = eval_rf(Xf2, y, "OptB(34)"); results.append(("+OptionB(34)", "RF", *r))
r = eval_lstm(Xb, y, lengths, "OptB(34)"); results.append(("+OptionB(34)", "LSTM", *r))

print("\n" + "=" * 70, flush=True)
print("STAGE 3: COMBINED — 22 + 19 + 12 = 53 features/day", flush=True)
print("=" * 70, flush=True)
Xcomb = np.concatenate([X_orig, X_c, X_b], axis=2)
print(f"  Shape: {Xcomb.shape}", flush=True)
np.save('outputs/features/X_combined.npy', Xcomb)
Xf3 = flatten(Xcomb, lengths)
r = eval_rf(Xf3, y, "Combined(53)"); results.append(("+Combined(53)", "RF", *r))
r = eval_lstm(Xcomb, y, lengths, "Combined(53)"); results.append(("+Combined(53)", "LSTM", *r))

# =====================================================================
# FINAL TABLE
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("FINAL COMPARISON TABLE", flush=True)
print("=" * 70, flush=True)
print(f"\n{'Features':<20s} {'Model':<6s} {'Acc':>7s} {'4-AUC':>7s} {'3-AUC':>7s} {'2-AUC':>7s}", flush=True)
print("-" * 55, flush=True)
for feat, model, acc, auc4, auc3, auc2 in results:
    print(f"{feat:<20s} {model:<6s} {acc*100:>6.1f}% {auc4:>7.4f} {auc3:>7.4f} {auc2:>7.4f}", flush=True)
