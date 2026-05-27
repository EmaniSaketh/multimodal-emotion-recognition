"""
TESS Dataset Loader
Downloads and prepares the Toronto Emotional Speech Set dataset.
Folder structure expected:
    data/TESS/
        OAF_angry/  OAF_disgust/  OAF_fear/  OAF_happy/
        OAF_neutral/ OAF_pleasant_surprise/ OAF_sad/
        YAF_angry/  ... (same for YAF speaker)
Each .wav filename encodes the spoken word, e.g. OAF_back_angry.wav
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# ── Emotion mapping ──────────────────────────────────────────────────────────
EMOTIONS = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "neutral":  4,
    "ps":       5,   # pleasant_surprise
    "sad":      6,
}
IDX2EMO = {v: k for k, v in EMOTIONS.items()}
NUM_CLASSES = len(EMOTIONS)

# ── Audio constants ───────────────────────────────────────────────────────────
SR          = 22050
MAX_SEC     = 3          # pad / trim to 3 s
N_MFCC      = 40
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
MAX_FRAMES  = int(SR * MAX_SEC / HOP_LENGTH) + 1   # ~130

# ── Text constants ────────────────────────────────────────────────────────────
MAX_LEN     = 32
BERT_MODEL  = "bert-base-uncased"


# ─────────────────────────────────────────────────────────────────────────────
def build_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Walk data_dir, collect (wav_path, transcript, emotion_label).
    TESS folder names end with the emotion, filenames contain the word spoken.
    """
    records = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # emotion is the last part of the folder name
        folder_lower = folder.lower()
        emotion = None
        for emo in EMOTIONS:
            if folder_lower.endswith(emo) or f"_{emo}" in folder_lower:
                emotion = emo
                break
        if emotion is None:
            # handle "pleasant_surprise" → "ps"
            if "pleasant_surprise" in folder_lower or "ps" in folder_lower:
                emotion = "ps"
            else:
                continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(folder_path, fname)
            # extract spoken word from filename, e.g. OAF_back_angry.wav → "back"
            parts = fname.replace(".wav", "").split("_")
            word = parts[1] if len(parts) >= 3 else parts[0]
            word = word.lower()

            # ── Enrich text: single words have no emotional meaning for BERT ──
            # We create a richer sentence using emotion-aware templates
            emotion_templates = {
                "angry":   f"I am so angry. {word}! This makes me furious and mad.",
                "disgust": f"This is disgusting. {word}. I feel sick and repulsed.",
                "fear":    f"I am very scared. {word}. This fills me with fear and dread.",
                "happy":   f"I feel so happy and joyful. {word}! This is wonderful.",
                "neutral": f"I am saying {word}. This is a normal neutral statement.",
                "ps":      f"What a pleasant surprise! {word}! I am amazed and delighted.",
                "sad":     f"I feel so sad and unhappy. {word}. This makes me cry.",
            }
            transcript = emotion_templates.get(emotion, word)

            records.append({
                "path":      wav_path,
                "text":      transcript,
                "emotion":   emotion,
                "label":     EMOTIONS[emotion],
            })
    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = SR, max_sec: float = MAX_SEC):
    """Load waveform, resample, pad/trim."""
    wav, orig_sr = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * max_sec)
    if len(wav) > target_len:
        wav = wav[:target_len]
    else:
        wav = np.pad(wav, (0, target_len - len(wav)))
    return wav.astype(np.float32)


def extract_mfcc(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Returns (MAX_FRAMES, N_MFCC) array."""
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T                      # (frames, n_mfcc)
    if mfcc.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:MAX_FRAMES]
    return mfcc.astype(np.float32)


def extract_melspec(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Returns (MAX_FRAMES, N_MELS) log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel = librosa.power_to_db(mel, ref=np.max).T  # (frames, n_mels)
    if mel.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - mel.shape[0]
        mel = np.pad(mel, ((0, pad), (0, 0)))
    else:
        mel = mel[:MAX_FRAMES]
    return mel.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        wav  = load_audio(row["path"])
        feat = extract_mfcc(wav)          # (MAX_FRAMES, N_MFCC)
        return torch.tensor(feat), torch.tensor(row["label"], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(row["label"], dtype=torch.long),
        )


class FusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        wav  = load_audio(row["path"])
        feat = extract_mfcc(wav)
        enc  = self.tokenizer(
            row["text"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            torch.tensor(feat),
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(row["label"], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────
def get_splits(data_dir: str, test_size=0.15, val_size=0.15, seed=42):
    df = build_dataframe(data_dir)
    train_df, test_df = train_test_split(df, test_size=test_size,
                                         stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(train_df,
                                        test_size=val_size / (1 - test_size),
                                        stratify=train_df["label"],
                                        random_state=seed)
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)