"""
Text Emotion Recognition Model
Architecture:
  Input  : token ids + attention mask  (BERT tokenizer output)
  Block 1: BERT encoder (contextual modelling, frozen base + fine-tune top layers)
  Block 2: CLS-token pooling
  Block 3: FC classifier → NUM_CLASSES
"""

import torch
import torch.nn as nn
from transformers import BertModel
from utils.dataset import NUM_CLASSES, BERT_MODEL


class TextEmotionModel(nn.Module):
    def __init__(self,
                 bert_name:   str = BERT_MODEL,
                 num_classes: int = NUM_CLASSES,
                 dropout:     float = 0.3,
                 freeze_layers: int = 10):   # freeze bottom N BERT layers
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)

        # Freeze embedding + bottom layers for efficiency
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden = self.bert.config.hidden_size   # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # CLS token  (B, 768)
        return self.classifier(cls)

    def get_representation(self, input_ids, attention_mask):
        """Return CLS embedding before classifier (for t-SNE)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]
