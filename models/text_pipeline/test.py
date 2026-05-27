"""
Text Pipeline — Test / Inference Script
Usage:
    python -m models.text_pipeline.test --data_dir data/TESS
    python -m models.text_pipeline.test --text "I am so happy today"
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils import (
    get_splits, TextDataset, get_device,
    load_checkpoint, evaluate, plot_confusion,
    print_report, IDX2EMO, BERT_MODEL, MAX_LEN,
)
from models.text_pipeline.model import TextEmotionModel


def test_dataset(args):
    device    = get_device()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    _, _, test_df = get_splits(args.data_dir)
    test_loader   = DataLoader(TextDataset(test_df, tokenizer),
                               batch_size=32, shuffle=False)
    model = TextEmotionModel().to(device)
    load_checkpoint(model, args.ckpt, device)

    criterion = torch.nn.CrossEntropyLoss()
    _, acc, preds, labels = evaluate(model, test_loader, criterion, device, "text")
    print(f"\n=== Text Test Accuracy: {acc:.4f} ===\n")
    print_report(labels, preds)
    plot_confusion(labels, preds, "Text Confusion Matrix",
                   "Results/plots/text_confusion_test.png")


def predict_single(text: str, ckpt: str):
    device    = get_device()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model     = TextEmotionModel().to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()

    enc  = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, mask)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred   = probs.argmax()

    print(f"Predicted emotion: {IDX2EMO[pred]}  (confidence {probs[pred]:.2%})")
    for i, p in enumerate(probs):
        print(f"  {IDX2EMO[i]:12s}: {p:.4f}")
    return IDX2EMO[pred], probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/TESS")
    ap.add_argument("--ckpt",     default="checkpoints/text_best.pt")
    ap.add_argument("--text",     default=None)
    args = ap.parse_args()

    if args.text:
        predict_single(args.text, args.ckpt)
    else:
        test_dataset(args)
