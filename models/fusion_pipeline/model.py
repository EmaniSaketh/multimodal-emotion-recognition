"""
Multimodal Fusion Emotion Recognition Model
Architecture:
  Speech branch : CNN → BiLSTM → Attention  →  speech_repr (512-d)
  Text branch   : BERT → CLS pool           →  text_repr   (768-d)
  Fusion        : Cross-modal attention + Concatenation → fused_repr
  Classifier    : FC → NUM_CLASSES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from utils.dataset import NUM_CLASSES, BERT_MODEL, N_MFCC


class CrossModalAttention(nn.Module):
    """Lightweight cross-modal attention: query from one, key/value from other."""
    def __init__(self, dim_q: int, dim_kv: int, out_dim: int):
        super().__init__()
        self.q  = nn.Linear(dim_q,  out_dim)
        self.k  = nn.Linear(dim_kv, out_dim)
        self.v  = nn.Linear(dim_kv, out_dim)
        self.scale = out_dim ** -0.5

    def forward(self, query, context):
        # query: (B, dim_q)  context: (B, dim_kv)
        q = self.q(query).unsqueeze(1)           # (B, 1, out_dim)
        k = self.k(context).unsqueeze(1)         # (B, 1, out_dim)
        v = self.v(context).unsqueeze(1)         # (B, 1, out_dim)
        attn = torch.softmax((q * k) * self.scale, dim=-1)
        out  = (attn * v).squeeze(1)             # (B, out_dim)
        return out


class FusionEmotionModel(nn.Module):
    def __init__(self,
                 n_mfcc:       int   = N_MFCC,
                 cnn_channels: int   = 128,
                 lstm_hidden:  int   = 256,
                 lstm_layers:  int   = 2,
                 bert_name:    str   = BERT_MODEL,
                 freeze_bert:  int   = 10,
                 fusion_dim:   int   = 256,
                 num_classes:  int   = NUM_CLASSES,
                 dropout:      float = 0.3):
        super().__init__()

        # ── Speech branch ─────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.speech_attn = nn.Linear(lstm_hidden * 2, 1)
        speech_dim = lstm_hidden * 2   # 512

        # ── Text branch ───────────────────────────────────────────────────
        self.bert = BertModel.from_pretrained(bert_name)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_bert:
                for p in layer.parameters():
                    p.requires_grad = False
        text_dim = self.bert.config.hidden_size  # 768

        # ── Cross-modal attention (fusion) ────────────────────────────────
        self.s2t_attn = CrossModalAttention(speech_dim, text_dim,  fusion_dim)
        self.t2s_attn = CrossModalAttention(text_dim,   speech_dim, fusion_dim)

        fused_dim = speech_dim + text_dim + fusion_dim * 2

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    # ── Forward helpers ───────────────────────────────────────────────────
    def _speech_repr(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        out, _ = self.lstm(x)
        w = torch.softmax(self.speech_attn(out), dim=1)
        return (w * out).sum(dim=1)   # (B, 512)

    def _text_repr(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state[:, 0, :]   # (B, 768)

    def forward(self, x, input_ids, attention_mask):
        s = self._speech_repr(x)
        t = self._text_repr(input_ids, attention_mask)

        s_aug = self.s2t_attn(s, t)   # speech attends to text
        t_aug = self.t2s_attn(t, s)   # text attends to speech

        fused = torch.cat([s, t, s_aug, t_aug], dim=-1)
        return self.classifier(fused)

    def get_representation(self, x, input_ids, attention_mask):
        s     = self._speech_repr(x)
        t     = self._text_repr(input_ids, attention_mask)
        s_aug = self.s2t_attn(s, t)
        t_aug = self.t2s_attn(t, s)
        return torch.cat([s, t, s_aug, t_aug], dim=-1)
