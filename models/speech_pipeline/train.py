"""
Speech Pipeline — Training Script
Usage:
    python -m models.speech_pipeline.train --data_dir data/TESS --epochs 50
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    get_splits, SpeechDataset, get_device,
    EarlyStopping, save_checkpoint, evaluate,
    plot_curves, plot_confusion, print_report,
)
from models.speech_pipeline.model import SpeechEmotionModel


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = get_splits(args.data_dir)
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    train_loader = DataLoader(SpeechDataset(train_df), batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(SpeechDataset(val_df),   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(SpeechDataset(test_df),  batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = SpeechEmotionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper   = EarlyStopping(patience=args.patience)

    ckpt_path    = "checkpoints/speech_best.pt"
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        t_loss = epoch_loss / len(train_loader)
        t_acc  = correct / total
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device, "speech")
        scheduler.step(v_loss)

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)

        print(f"  train loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"val loss {v_loss:.4f} acc {v_acc:.4f}")

        if stopper.best_loss is None or v_loss < stopper.best_loss:
            save_checkpoint(model, ckpt_path)
        stopper(v_loss)
        if stopper.stop:
            print("Early stopping triggered.")
            break

    # ── Test ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device, "speech")
    print(f"\n=== Test Accuracy (Speech): {test_acc:.4f} ===\n")
    print_report(labels, preds)

    # ── Save results ──────────────────────────────────────────────────────
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                "Speech Pipeline", "Results/plots/speech_curves.png")
    plot_confusion(labels, preds, "Speech Confusion Matrix",
                   "Results/plots/speech_confusion.png")

    import pandas as pd, json
    os.makedirs("Results/tables", exist_ok=True)
    pd.DataFrame({"epoch": range(1, len(train_losses)+1),
                  "train_loss": train_losses, "val_loss": val_losses,
                  "train_acc":  train_accs,  "val_acc":  val_accs}
                ).to_csv("Results/tables/speech_history.csv", index=False)
    with open("Results/tables/speech_test_acc.json", "w") as f:
        json.dump({"test_accuracy": round(test_acc, 4)}, f)
    print("Results saved to Results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="data/TESS")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--patience",   type=int,   default=10)
    train(ap.parse_args())
