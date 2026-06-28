"""Tune LSTM hyperparameters."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from config import RANDOM_STATE, N_FOLDS, NUM_DAILY_FEATURES, SEQUENCE_LENGTH, set_seed

X = np.load('outputs/features/X.npy')
y = np.load('outputs/features/y.npy')
lengths = np.load('outputs/features/lengths.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=4, bidirectional=False, use_attention=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size

        if use_attention:
            self.attn_w = nn.Linear(lstm_out_size, 1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(lstm_out_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, _) = self.lstm(packed)
        else:
            output, (h_n, _) = self.lstm(x)

        if self.use_attention:
            # Unpack
            if lengths is not None:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            # Attention weights
            attn_scores = self.attn_w(output).squeeze(-1)  # (batch, seq_len)
            # Mask padding
            if lengths is not None:
                mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) >= lengths.unsqueeze(1).to(output.device)
                attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            context = (output * attn_weights).sum(dim=1)  # (batch, hidden)
            return self.head(context)
        else:
            if self.bidirectional:
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                last_hidden = h_n[-1]
            return self.head(last_hidden)


def train_config(config, verbose=True):
    """Train one LSTM configuration and return 4-class OOF AUC."""
    set_seed()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        len_tr, len_va = lengths[train_idx], lengths[val_idx]

        n_tr, seq_len, n_feat = X_tr_raw.shape
        n_va = X_va_raw.shape[0]

        # Impute + scale
        X_tr_2d = X_tr_raw.reshape(-1, n_feat).copy()
        X_va_2d = X_va_raw.reshape(-1, n_feat).copy()
        imp = SimpleImputer(strategy='median')
        X_tr_2d = imp.fit_transform(X_tr_2d)
        X_va_2d = imp.transform(X_va_2d)
        sc = StandardScaler()
        X_tr_2d = sc.fit_transform(X_tr_2d)
        X_va_2d = sc.transform(X_va_2d)

        X_train_t = torch.tensor(X_tr_2d.reshape(n_tr, seq_len, n_feat).astype(np.float32))
        y_train_t = torch.tensor(y_tr, dtype=torch.long)
        len_train_t = torch.tensor(len_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_va_2d.reshape(n_va, seq_len, n_feat).astype(np.float32)).to(device)
        len_val_t = torch.tensor(len_va, dtype=torch.long)
        y_val_t = torch.tensor(y_va, dtype=torch.long).to(device)

        train_ds = TensorDataset(X_train_t, y_train_t, len_train_t)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

        # Class weights
        cc = np.bincount(y_tr, minlength=4).astype(float)
        cw = 1.0 / (cc + 1e-6)
        cw = cw / cw.sum() * 4
        wt = torch.tensor(cw, dtype=torch.float32).to(device)

        set_seed()
        model = LSTMModel(
            input_size=n_feat,
            hidden_size=config['hidden'],
            num_layers=config['layers'],
            dropout=config['dropout'],
            bidirectional=config.get('bidirectional', False),
            use_attention=config.get('attention', False),
        ).to(device)

        label_smoothing = config.get('label_smoothing', 0.0)
        criterion = nn.CrossEntropyLoss(weight=wt, label_smoothing=label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('wd', 0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = config.get('patience', 15)

        for epoch in range(config['epochs']):
            model.train()
            for X_b, y_b, l_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b, l_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = criterion(model(X_val_t, len_val_t), y_val_t).item()
            scheduler.step(vl)

            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(X_val_t, len_val_t), dim=1).cpu().numpy()
        oof_proba[val_idx] = proba

    auc4 = roc_auc_score(y, oof_proba, multi_class='ovr', average='macro')

    # Also compute 3-class and 2-class
    from config import LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2
    y3 = np.array([LABEL_MAP_4_TO_3[yi] for yi in y])
    p3 = np.zeros((len(y), 3))
    p3[:, 0] = oof_proba[:, 0]
    p3[:, 1] = oof_proba[:, 1]
    p3[:, 2] = oof_proba[:, 2] + oof_proba[:, 3]
    auc3 = roc_auc_score(y3, p3, multi_class='ovr', average='macro')

    y2 = np.array([LABEL_MAP_4_TO_2[yi] for yi in y])
    p2_pos = oof_proba[:, 1] + oof_proba[:, 2] + oof_proba[:, 3]
    auc2 = roc_auc_score(y2, p2_pos)

    return auc4, auc3, auc2, oof_proba


# ===== LSTM CONFIGURATIONS =====
configs = [
    # Baseline (what we had)
    {"name": "baseline", "hidden": 128, "layers": 2, "dropout": 0.3, "lr": 0.001, "batch_size": 64, "epochs": 30},

    # Smaller model (less overfitting)
    {"name": "small", "hidden": 64, "layers": 1, "dropout": 0.3, "lr": 0.001, "batch_size": 64, "epochs": 50},
    {"name": "small-2L", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.001, "batch_size": 64, "epochs": 50},

    # Lower LR
    {"name": "low-lr", "hidden": 128, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60},
    {"name": "very-low-lr", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0003, "batch_size": 64, "epochs": 80},

    # More dropout
    {"name": "high-drop", "hidden": 128, "layers": 2, "dropout": 0.5, "lr": 0.001, "batch_size": 64, "epochs": 50},
    {"name": "high-drop-small", "hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.0005, "batch_size": 64, "epochs": 60},

    # Weight decay
    {"name": "wd", "hidden": 128, "layers": 2, "dropout": 0.3, "lr": 0.001, "batch_size": 64, "epochs": 50, "wd": 1e-4},
    {"name": "wd-small", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "wd": 1e-4},

    # Bidirectional
    {"name": "bidir", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "bidirectional": True},
    {"name": "bidir-small", "hidden": 32, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "bidirectional": True},

    # Attention
    {"name": "attn", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "attention": True},
    {"name": "attn-bidir", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "attention": True, "bidirectional": True},

    # Label smoothing
    {"name": "smooth", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "label_smoothing": 0.1},
    {"name": "smooth-attn", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 64, "epochs": 60, "label_smoothing": 0.1, "attention": True},

    # Larger batch
    {"name": "big-batch", "hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.001, "batch_size": 128, "epochs": 80},

    # Combo: best ideas together
    {"name": "combo1", "hidden": 64, "layers": 2, "dropout": 0.4, "lr": 0.0005, "batch_size": 64, "epochs": 80, "wd": 1e-4, "attention": True, "label_smoothing": 0.1, "patience": 20},
    {"name": "combo2", "hidden": 64, "layers": 2, "dropout": 0.4, "lr": 0.0003, "batch_size": 64, "epochs": 100, "wd": 5e-5, "bidirectional": True, "attention": True, "label_smoothing": 0.05, "patience": 25},
    {"name": "combo3", "hidden": 32, "layers": 2, "dropout": 0.3, "lr": 0.0005, "batch_size": 32, "epochs": 80, "wd": 1e-4, "attention": True, "label_smoothing": 0.1, "patience": 20},
]

print(f"\nTuning {len(configs)} LSTM configurations...\n")
print(f"{'Config':<20s}  {'4-AUC':>7s}  {'3-AUC':>7s}  {'2-AUC':>7s}", flush=True)
print("-" * 50, flush=True)

best_auc = 0
best_name = ""
best_proba = None
all_results = []

for cfg in configs:
    name = cfg['name']
    try:
        auc4, auc3, auc2, proba = train_config(cfg, verbose=False)
        marker = " ***BEST***" if auc4 > best_auc else ""
        if auc4 > best_auc:
            best_auc = auc4
            best_name = name
            best_proba = proba
        print(f"{name:<20s}  {auc4:>7.4f}  {auc3:>7.4f}  {auc2:>7.4f}{marker}", flush=True)
        all_results.append((name, auc4, auc3, auc2))
    except Exception as e:
        print(f"{name:<20s}  ERROR: {e}", flush=True)

print(f"\nBest LSTM: {best_name} = {best_auc:.4f}")

# Save best OOF predictions
if best_proba is not None:
    np.save('outputs/features/best_lstm_oof_proba.npy', best_proba)
    print("Saved best OOF predictions")
