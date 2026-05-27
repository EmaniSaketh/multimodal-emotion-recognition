"""
Multimodal Emotion Recognition — Streamlit App
Run:  streamlit run app.py
"""

import os, sys, io, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from transformers import BertTokenizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0d0d0f; }
[data-testid="stSidebar"] { background: #111116; border-right: 1px solid #222; }

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
    background: linear-gradient(90deg, #e94560, #0f3460);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
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
.mode-badge {
    background: #0f3460;
    color: #a0c4ff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 4px;
}
.metric-box {
    background: #111116;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c1121f);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "angry":   "😠", "disgust": "🤢", "fear":    "😨",
    "happy":   "😊", "neutral": "😐", "ps":      "😲", "sad": "😢",
}
EMOTION_COLOR = {
    "angry":   "#e94560", "disgust": "#7b2d8b", "fear":    "#f5a623",
    "happy":   "#27ae60", "neutral": "#3498db", "ps":      "#e67e22",
    "sad":     "#2980b9",
}
IDX2EMO = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"ps", 6:"sad"}
BERT_MODEL = "bert-base-uncased"
MAX_LEN    = 32
SR         = 22050
MAX_SEC    = 3
N_MFCC     = 40
HOP_LENGTH = 512
N_FFT      = 2048
MAX_FRAMES = int(SR * MAX_SEC / HOP_LENGTH) + 1


# ── Model loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(BERT_MODEL)

@st.cache_resource
def load_speech_model(ckpt="checkpoints/speech_best.pt"):
    from models.speech_pipeline.model import SpeechEmotionModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpeechEmotionModel().to(device)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model, device
    return None, device

@st.cache_resource
def load_text_model(ckpt="checkpoints/text_best.pt"):
    from models.text_pipeline.model import TextEmotionModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TextEmotionModel().to(device)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model, device
    return None, device

@st.cache_resource
def load_fusion_model(ckpt="checkpoints/fusion_best.pt"):
    from models.fusion_pipeline.model import FusionEmotionModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FusionEmotionModel().to(device)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model, device
    return None, device


# ── Feature helpers ───────────────────────────────────────────────────────────
def process_audio(audio_bytes):
    import librosa, soundfile as sf
    wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    target = int(SR * MAX_SEC)
    wav    = wav[:target] if len(wav) > target else np.pad(wav, (0, target - len(wav)))
    mfcc   = librosa.feature.mfcc(y=wav.astype(np.float32), sr=SR,
                                   n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
    if mfcc.shape[0] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, MAX_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_FRAMES]
    return mfcc.astype(np.float32)


def process_text(text, tokenizer):
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


# ── Inference functions ───────────────────────────────────────────────────────
def predict_speech(audio_bytes):
    model, device = load_speech_model()
    if model is None:
        return None, None
    feat   = torch.tensor(process_audio(audio_bytes)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(feat), -1).squeeze().cpu().numpy()
    return IDX2EMO[probs.argmax()], probs

def predict_text(text):
    model, device = load_text_model()
    tokenizer     = load_tokenizer()
    if model is None:
        return None, None
    ids, mask = process_text(text, tokenizer)
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(ids, mask), -1).squeeze().cpu().numpy()
    return IDX2EMO[probs.argmax()], probs

def predict_fusion(audio_bytes, text):
    model, device = load_fusion_model()
    tokenizer     = load_tokenizer()
    if model is None:
        return None, None
    feat      = torch.tensor(process_audio(audio_bytes)).unsqueeze(0).to(device)
    ids, mask = process_text(text, tokenizer)
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(feat, ids, mask), -1).squeeze().cpu().numpy()
    return IDX2EMO[probs.argmax()], probs


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_result(emotion, probs, mode_label):
    if emotion is None:
        st.warning("⚠️ Model checkpoint not found. Please train the model first.")
        return
    emoji = EMOTION_EMOJI.get(emotion, "🎭")
    conf  = probs.max()
    st.markdown(f"""
    <div class="emotion-card">
        <div style="font-size:3.5rem">{emoji}</div>
        <p class="emotion-title">{emotion.upper()}</p>
        <span class="confidence-badge">{conf:.1%} confidence</span><br>
        <span class="mode-badge">{mode_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart
    labels = [f"{EMOTION_EMOJI[IDX2EMO[i]]} {IDX2EMO[i]}" for i in range(len(probs))]
    colors = [EMOTION_COLOR[IDX2EMO[i]] for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font=dict(color="white", family="Space Grotesk"),
        xaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=40, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_comparison(s_probs, t_probs, f_probs):
    """Side-by-side radar chart comparing all 3 models."""
    emos = [IDX2EMO[i] for i in range(7)]
    fig  = go.Figure()
    for name, probs, color in [
        ("Speech", s_probs, "#e94560"),
        ("Text",   t_probs, "#0f3460"),
        ("Fusion", f_probs, "#27ae60"),
    ]:
        if probs is not None:
            fig.add_trace(go.Scatterpolar(
                r=list(probs) + [probs[0]],
                theta=emos + [emos[0]],
                fill="toself", name=name,
                line_color=color,
                fillcolor=color,
                opacity=0.4,
            ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font=dict(color="white"),
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 EmotionAI")
    st.markdown("*Multimodal Emotion Recognition*")
    st.divider()

    mode = st.radio("**Analysis Mode**", [
        "🎤 Speech Only",
        "📝 Text Only",
        "🔀 Multimodal (Speech + Text)",
        "📊 Compare All Models",
    ])
    st.divider()

    # Model status
    st.markdown("**Model Status**")
    for label, path in [("Speech", "checkpoints/speech_best.pt"),
                         ("Text",   "checkpoints/text_best.pt"),
                         ("Fusion", "checkpoints/fusion_best.pt")]:
        icon = "✅" if os.path.exists(path) else "❌"
        st.markdown(f"{icon} {label} model")

    st.divider()
    st.markdown("""
    **How to train:**
    ```bash
    python -m models.speech_pipeline.train
    python -m models.text_pipeline.train
    python -m models.fusion_pipeline.train
    ```
    """)

    # Load results if available
    results = {}
    for key, path in [("Speech", "Results/tables/speech_test_acc.json"),
                       ("Text",   "Results/tables/text_test_acc.json"),
                       ("Fusion", "Results/tables/fusion_test_acc.json")]:
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)["test_accuracy"]

    if results:
        st.divider()
        st.markdown("**Test Accuracies**")
        for k, v in results.items():
            st.metric(k, f"{v:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🎭 Multimodal Emotion Recognition")
st.markdown("Detect emotions from **speech**, **text**, or **both combined**.")
st.divider()

# ── Speech Only ───────────────────────────────────────────────────────────────
if mode == "🎤 Speech Only":
    st.subheader("🎤 Speech Emotion Recognition")
    audio_file = st.file_uploader("Upload a WAV audio file", type=["wav", "mp3", "ogg"])
    if audio_file:
        st.audio(audio_file)
        if st.button("Analyze Emotion"):
            with st.spinner("Extracting features & running inference..."):
                emotion, probs = predict_speech(audio_file.read())
            render_result(emotion, probs, "Speech Pipeline · CNN-BiLSTM-Attention")

# ── Text Only ─────────────────────────────────────────────────────────────────
elif mode == "📝 Text Only":
    st.subheader("📝 Text Emotion Recognition")
    text_input = st.text_area("Enter text to analyze", height=120,
                               placeholder="e.g. I am so happy today!")
    if st.button("Analyze Emotion") and text_input.strip():
        with st.spinner("Running BERT inference..."):
            emotion, probs = predict_text(text_input.strip())
        render_result(emotion, probs, "Text Pipeline · BERT Fine-tuned")

# ── Multimodal ────────────────────────────────────────────────────────────────
elif mode == "🔀 Multimodal (Speech + Text)":
    st.subheader("🔀 Multimodal Emotion Recognition")
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file)
    with col2:
        text_input = st.text_area("Enter Transcript / Text", height=120,
                                   placeholder="Type what is said in the audio...")

    if st.button("Analyze with Fusion Model"):
        if audio_file and text_input.strip():
            with st.spinner("Running multimodal fusion inference..."):
                emotion, probs = predict_fusion(audio_file.read(), text_input.strip())
            render_result(emotion, probs, "Fusion Pipeline · Cross-Modal Attention")
        else:
            st.error("Please provide both audio and text.")

# ── Compare All ───────────────────────────────────────────────────────────────
elif mode == "📊 Compare All Models":
    st.subheader("📊 Compare All Three Pipelines")
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file)
    with col2:
        text_input = st.text_area("Enter Text", height=120,
                                   placeholder="Transcript or any text...")

    if st.button("Run All Models"):
        if audio_file and text_input.strip():
            audio_bytes = audio_file.read()
            text        = text_input.strip()

            c1, c2, c3 = st.columns(3)
            s_probs = t_probs = f_probs = None

            with c1:
                st.markdown("#### 🎤 Speech")
                with st.spinner():
                    s_emo, s_probs = predict_speech(audio_bytes)
                render_result(s_emo, s_probs, "Speech")

            with c2:
                st.markdown("#### 📝 Text")
                with st.spinner():
                    t_emo, t_probs = predict_text(text)
                render_result(t_emo, t_probs, "Text")

            with c3:
                st.markdown("#### 🔀 Fusion")
                with st.spinner():
                    f_emo, f_probs = predict_fusion(audio_bytes, text)
                render_result(f_emo, f_probs, "Fusion")

            st.divider()
            st.markdown("#### 🕸️ Model Comparison Radar")
            render_comparison(s_probs, t_probs, f_probs)
        else:
            st.error("Please provide both audio and text.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "TESS Dataset · CNN-BiLSTM · BERT · Cross-Modal Attention Fusion"
    "</p>",
    unsafe_allow_html=True,
)
