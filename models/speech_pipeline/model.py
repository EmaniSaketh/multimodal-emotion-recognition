"""
Speech Emotion Recognition Model
Architecture:
  Input  : (batch, MAX_FRAMES, N_MFCC)  — MFCC features
  Block 1: 1-D CNN feature refiner
  Block 2: Bidirectional LSTM temporal modeller
  Block 3: Attention pooling
  Block 4: FC classifier → NUM_CLASSES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import MAX_FRAMES, N_MFCC, NUM_CLASSES


class SpeechEmotionModel(nn.Module):
    def __init__(self,
                 n_mfcc:      int = N_MFCC,
                 cnn_channels: int = 128,
                 lstm_hidden:  int = 256,
                 lstm_layers:  int = 2,
                 num_classes:  int = NUM_CLASSES,
                 dropout:      float = 0.3):
        super().__init__()

        # ── 1-D CNN feature refiner ───────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        # ── BiLSTM temporal modeller ──────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Attention pooling ─────────────────────────────────────────────
        self.attn = nn.Linear(lstm_hidden * 2, 1)

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, n_mfcc)
        x = x.permute(0, 2, 1)          # (B, n_mfcc, T)  — for Conv1d
        x = self.cnn(x)                  # (B, C, T)
        x = x.permute(0, 2, 1)          # (B, T, C)  — for LSTM

        out, _ = self.lstm(x)            # (B, T, 2*H)
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        ctx    = (attn_w * out).sum(dim=1)              # (B, 2*H)

        return self.classifier(ctx)      # (B, num_classes)

    def get_representation(self, x):
        """Return latent vector before classifier (for t-SNE)."""
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        ctx    = (attn_w * out).sum(dim=1)
        return ctx
