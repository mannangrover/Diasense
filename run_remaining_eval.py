"""Run remaining evaluations: Option B LSTM, Combined RF + LSTM."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from config import RANDOM_STATE, N_FOLDS, LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2, CLASS_NAMES_4, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

X_orig = np.load('outputs/features/X.npy')
y = np.load('outputs/features/y.npy')
lengths = np.load('outputs/features/lengths.npy')
X_c = np.load('outputs/features/X_option_c.npy')
X_b = np.load('outputs/features/X_option_b.npy')

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
    print(f"  Per-class:\n{classification_report(y, oof_pred, target_names=CLASS_NAMES_4, digits=3)}", flush=True)
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
    print(f"  Per-class:\n{classification_report(y, pred, target_names=CLASS_NAMES_4, digits=3)}", flush=True)
    return acc, auc4, auc3, auc2

# Option B LSTM (RF already done: acc=40.6%, 4-AUC=0.6595)
print("\n" + "=" * 70, flush=True)
print("OPTION B LSTM (22 + 12 = 34 features/day)", flush=True)
print("=" * 70, flush=True)
Xb = np.concatenate([X_orig, X_b], axis=2)
eval_lstm(Xb, y, lengths, "OptB(34)")

# Combined RF + LSTM (22 + 19 + 12 = 53 features/day)
print("\n" + "=" * 70, flush=True)
print("COMBINED (22 + 19 + 12 = 53 features/day)", flush=True)
print("=" * 70, flush=True)
Xcomb = np.concatenate([X_orig, X_c, X_b], axis=2)
print(f"  Shape: {Xcomb.shape}", flush=True)
np.save('outputs/features/X_combined.npy', Xcomb)
Xf = flatten(Xcomb, lengths)
eval_rf(Xf, y, "Combined(53)")
eval_lstm(Xcomb, y, lengths, "Combined(53)")

print("\n" + "=" * 70, flush=True)
print("DONE", flush=True)
print("=" * 70, flush=True)
