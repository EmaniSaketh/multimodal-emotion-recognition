"""
Speech Pipeline — Test / Inference Script
Usage:
    python -m models.speech_pipeline.test --data_dir data/TESS
    python -m models.speech_pipeline.test --wav_file path/to/file.wav
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import (
    get_splits, SpeechDataset, get_device,
    load_checkpoint, evaluate, plot_confusion,
    print_report, IDX2EMO, load_audio, extract_mfcc,
)
from models.speech_pipeline.model import SpeechEmotionModel


def test_dataset(args):
    device = get_device()
    _, _, test_df = get_splits(args.data_dir)
    test_loader   = DataLoader(SpeechDataset(test_df),
                               batch_size=32, shuffle=False)
    model = SpeechEmotionModel().to(device)
    load_checkpoint(model, args.ckpt, device)

    criterion = torch.nn.CrossEntropyLoss()
    _, acc, preds, labels = evaluate(model, test_loader, criterion, device, "speech")
    print(f"\n=== Speech Test Accuracy: {acc:.4f} ===\n")
    print_report(labels, preds)
    plot_confusion(labels, preds, "Speech Confusion Matrix",
                   "Results/plots/speech_confusion_test.png")


def predict_single(wav_path: str, ckpt: str):
    device = get_device()
    model  = SpeechEmotionModel().to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()

    wav  = load_audio(wav_path)
    feat = torch.tensor(extract_mfcc(wav)).unsqueeze(0).to(device)  # (1, T, F)
    with torch.no_grad():
        logits = model(feat)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred   = probs.argmax()
    print(f"Predicted emotion: {IDX2EMO[pred]}  (confidence {probs[pred]:.2%})")
    for i, p in enumerate(probs):
        print(f"  {IDX2EMO[i]:12s}: {p:.4f}")
    return IDX2EMO[pred], probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/TESS")
    ap.add_argument("--ckpt",     default="checkpoints/speech_best.pt")
    ap.add_argument("--wav_file", default=None, help="single-file inference")
    args = ap.parse_args()

    if args.wav_file:
        predict_single(args.wav_file, args.ckpt)
    else:
        test_dataset(args)
