"""
Multimodal Emotion Recognition — Streamlit App
Run: streamlit run app.py
"""

import os, sys, io, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from transformers import BertTokenizer
import gdown

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model Auto Download ───────────────────────────────────────────────────────

MODEL_FILES = {
    "speech_best.pt": "1Kqs56nUuzDURS5wQakpZAQj-ywbS-Elr",
    "text_best.pt": "1FzRuyQbfN0MasWE_UtUQzCAlwQqzyc8l",
    "fusion_best.pt": "1_Kp3_j3ofKKPoD-oNZk8zhFMALvkGF-s",
}

os.makedirs("checkpoints", exist_ok=True)

for filename, file_id in MODEL_FILES.items():
    path = f"checkpoints/{filename}"

    if not os.path.exists(path):
        with st.spinner(f"Downloading {filename}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main {
    background: #0d0d0f;
}

[data-testid="stSidebar"] {
    background: #111116;
    border-right: 1px solid #222;
}

.emotion-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 8px 0;
}

.emotion-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: white;
}

.confidence-badge {
    background: #e94560;
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Emotion Metadata ──────────────────────────────────────────────────────────

EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "ps": "😲",
    "sad": "😢",
}

EMOTION_COLOR = {
    "angry": "#e94560",
    "disgust": "#7b2d8b",
    "fear": "#f5a623",
    "happy": "#27ae60",
    "neutral": "#3498db",
    "ps": "#e67e22",
    "sad": "#2980b9",
}

IDX2EMO = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "ps",
    6: "sad"
}

BERT_MODEL = "bert-base-uncased"
MAX_LEN = 32
SR = 22050
MAX_SEC = 3
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
MAX_FRAMES = int(SR * MAX_SEC / HOP_LENGTH) + 1

# ── Cached Resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(BERT_MODEL)

@st.cache_resource
def load_speech_model():
    from models.speech_pipeline.model import SpeechEmotionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechEmotionModel().to(device)

    model.load_state_dict(
        torch.load("checkpoints/speech_best.pt", map_location=device)
    )

    model.eval()
    return model, device

@st.cache_resource
def load_text_model():
    from models.text_pipeline.model import TextEmotionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextEmotionModel().to(device)

    model.load_state_dict(
        torch.load("checkpoints/text_best.pt", map_location=device)
    )

    model.eval()
    return model, device

@st.cache_resource
def load_fusion_model():
    from models.fusion_pipeline.model import FusionEmotionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionEmotionModel().to(device)

    model.load_state_dict(
        torch.load("checkpoints/fusion_best.pt", map_location=device)
    )

    model.eval()
    return model, device

# ── Feature Processing ────────────────────────────────────────────────────────

def process_audio(audio_bytes):
    import librosa

    wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

    target = int(SR * MAX_SEC)

    if len(wav) > target:
        wav = wav[:target]
    else:
        wav = np.pad(wav, (0, target - len(wav)))

    mfcc = librosa.feature.mfcc(
        y=wav.astype(np.float32),
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    ).T

    if mfcc.shape[0] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, MAX_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_FRAMES]

    return mfcc.astype(np.float32)

def process_text(text, tokenizer):
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return enc["input_ids"], enc["attention_mask"]

# ── Predictions ───────────────────────────────────────────────────────────────

def predict_speech(audio_bytes):

    try:
        model, device = load_speech_model()

        feat = torch.tensor(
            process_audio(audio_bytes)
        ).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(feat)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()

        return IDX2EMO[np.argmax(probs)], probs

    except Exception as e:
        st.error(f"Speech inference error: {str(e)}")
        return None, None

# ── Result Renderer ───────────────────────────────────────────────────────────
import streamlit.components.v1 as components

def render_result(emotion, probs, label):

    conf = probs.max()
    emoji = EMOTION_EMOJI.get(emotion, "🎭")

    html_code = f"""
    <div style="
        background: linear-gradient(135deg,#1a1a2e,#16213e);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #0f3460;
        color: white;
        font-family: Arial;
    ">

        <div style="font-size:4rem;">
            {emoji}
        </div>

        <h1>
            {emotion.upper()}
        </h1>

        <h3 style="color:#e94560;">
            {conf:.1%} confidence
        </h3>

        <p style="color:#aaa;">
            {label}
        </p>

    </div>
    """

    components.html(html_code, height=300)
# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎭 EmotionAI")

    mode = st.radio(
        "Choose Mode",
        [
            "🎤 Speech Only",
            "📝 Text Only",
            "🔀 Multimodal"
        ]
    )
def predict_text(text):

    try:
        model, device = load_text_model()
        tokenizer = load_tokenizer()

        ids, mask = process_text(text, tokenizer)

        ids = ids.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            output = model(ids, mask)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()

        return IDX2EMO[np.argmax(probs)], probs

    except Exception as e:
        st.error(f"Text inference error: {str(e)}")
        return None, None

def predict_fusion(audio_bytes, text):

    try:
        model, device = load_fusion_model()
        tokenizer = load_tokenizer()

        feat = torch.tensor(
            process_audio(audio_bytes)
        ).unsqueeze(0).float().to(device)

        ids, mask = process_text(text, tokenizer)

        ids = ids.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            output = model(feat, ids, mask)
            probs = torch.softmax(output, dim=-1).squeeze().cpu().numpy()

        return IDX2EMO[np.argmax(probs)], probs

    except Exception as e:
        st.error(f"Fusion inference error: {str(e)}")
        return None, None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎭 Multimodal Emotion Recognition")
st.write("Analyze emotions using Speech, Text, or Both.")

# ── Speech Only ───────────────────────────────────────────────────────────────

# ── Speech Only ───────────────────────────────────────────────────────────────

if mode == "🎤 Speech Only":

    st.subheader("🎤 Speech Emotion Recognition")

    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "ogg"]
    )

    if audio_file is not None:

        audio_bytes = audio_file.read()

        st.audio(audio_bytes)

        if st.button("Analyze Emotion"):

            with st.spinner("Running speech inference..."):

                emotion, probs = predict_speech(audio_bytes)

            if emotion is not None:
                render_result(
                    emotion,
                    probs,
                    "Speech Pipeline · CNN-BiLSTM"
                )

# ── Text Only ─────────────────────────────────────────────────────────────────

elif mode == "📝 Text Only":

    st.subheader("📝 Text Emotion Recognition")

    text_input = st.text_area(
        "Enter text",
        height=120
    )

    if st.button("Analyze Text Emotion") and text_input.strip():

        with st.spinner("Analyzing text..."):
            emotion, probs = predict_text(text_input.strip())

        render_result(
            emotion,
            probs,
            "Text Pipeline · BERT"
        )

# ── Multimodal ────────────────────────────────────────────────────────────────

elif mode == "🔀 Multimodal":

    st.subheader("🔀 Multimodal Emotion Recognition")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "ogg"]
    )

    text_input = st.text_area(
        "Enter Transcript/Text",
        height=120
    )

    if st.button("Analyze Multimodal Emotion"):

        if audio_file is not None and text_input.strip():

            audio_bytes = audio_file.read()

            with st.spinner("Running multimodal fusion inference..."):

                emotion, probs = predict_fusion(
                    audio_bytes,
                    text_input.strip()
                )

            if emotion is not None:
                render_result(
                    emotion,
                    probs,
                    "Fusion Pipeline · Cross Modal Attention"
                )

        else:
            st.error("Please provide both audio and text.")

st.divider()

st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "TESS Dataset · CNN-BiLSTM · BERT · Multimodal Fusion"
    "</p>",
    unsafe_allow_html=True,
)