"""
LSTM Model
==========
Daily-summary LSTM for diabetes classification.

Input:  (batch_size, 14, 22) — 14 days × 22 features per day
Output: (batch_size, 4)      — 4 class logits

Usage:
    from src.models.lstm import DiasenseLSTM
    model = DiasenseLSTM()
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    NUM_DAILY_FEATURES,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
)


class DiasenseLSTM(nn.Module):
    """
    2-layer LSTM → FC head for 4-class diabetes classification.

    The LSTM processes the 14-day sequence and we take the last
    hidden state as the participant representation. Then a small
    FC head maps to 4 class logits.

    Supports variable-length sequences via pack_padded_sequence.
    """

    def __init__(
        self,
        input_size=NUM_DAILY_FEATURES,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        num_classes=4,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, features) — padded sequences
            lengths: (batch,) — actual sequence lengths for packing

        Returns:
            logits: (batch, num_classes)
        """

        if lengths is not None:
            # Pack padded sequences so LSTM ignores padding
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # h_n shape: (num_layers, batch, hidden_size)
        # Take the last layer's hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)

        logits = self.head(last_hidden)
        return logits
