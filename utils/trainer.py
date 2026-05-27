"""Shared training utilities: metrics, checkpointing, early stopping."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from utils.dataset import IDX2EMO, NUM_CLASSES


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7, delta=1e-4):
        self.patience  = patience
        self.delta     = delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  ✔ checkpoint saved → {path}")


def load_checkpoint(model, path: str, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device, modality="speech"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            if modality == "speech":
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            elif modality == "text":
                ids, mask, y = batch
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                logits = model(ids, mask)
            else:   # fusion
                x, ids, mask, y = batch
                x, ids, mask, y = x.to(device), ids.to(device), mask.to(device), y.to(device)
                logits = model(x, ids, mask)

            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs,
                title: str, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train"); ax1.plot(val_losses, label="val")
    ax1.set_title(f"{title} — Loss"); ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_accs, label="train"); ax2.plot(val_accs, label="val")
    ax2.set_title(f"{title} — Accuracy"); ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✔ curves saved → {save_path}")


def plot_confusion(labels, preds, title: str, save_path: str):
    cm = confusion_matrix(labels, preds)
    emo_names = [IDX2EMO[i] for i in range(NUM_CLASSES)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=emo_names, yticklabels=emo_names)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✔ confusion matrix saved → {save_path}")


def print_report(labels, preds):
    emo_names = [IDX2EMO[i] for i in range(NUM_CLASSES)]
    print(classification_report(labels, preds, target_names=emo_names))
