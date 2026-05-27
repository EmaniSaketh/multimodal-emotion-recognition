# 🎭 Multimodal Emotion Recognition

Recognize emotions from **speech**, **text**, or **both combined** using deep learning.

## Architecture

| Pipeline | Feature Extraction | Temporal/Contextual Modelling | Classifier |
|---|---|---|---|
| Speech | MFCC (40 coefficients) | 1D-CNN → BiLSTM → Attention | FC layers |
| Text | BERT tokenizer | BERT fine-tuned (top 2 layers) | FC layers |
| Fusion | MFCC + BERT | CNN-BiLSTM + BERT | Cross-Modal Attention → FC |

**Emotions:** angry, disgust, fear, happy, neutral, pleasant_surprise (ps), sad

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download TESS Dataset
- Go to: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- Download and extract to `data/TESS/`

Expected structure:
```
data/TESS/
    OAF_angry/
    OAF_disgust/
    OAF_fear/
    OAF_happy/
    OAF_neutral/
    OAF_pleasant_surprise/
    OAF_sad/
    YAF_angry/
    ...
```

---

## Training

```bash
# Train Speech pipeline
python -m models.speech_pipeline.train --data_dir data/TESS --epochs 50

# Train Text pipeline
python -m models.text_pipeline.train --data_dir data/TESS --epochs 20

# Train Fusion pipeline
python -m models.fusion_pipeline.train --data_dir data/TESS --epochs 30
```

Checkpoints are saved to `checkpoints/`.

---

## Testing

```bash
# Test on dataset
python -m models.speech_pipeline.test --data_dir data/TESS
python -m models.text_pipeline.test   --data_dir data/TESS
python -m models.fusion_pipeline.test --data_dir data/TESS

# Single file inference
python -m models.speech_pipeline.test --wav_file path/to/audio.wav
python -m models.text_pipeline.test   --text "I am so happy today"
python -m models.fusion_pipeline.test --wav_file audio.wav --text "hello"
```

---

## Streamlit App (Deployment)

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### App Features:
- 🎤 **Speech Only** — upload WAV, get emotion
- 📝 **Text Only** — type text, get emotion
- 🔀 **Multimodal** — audio + text combined
- 📊 **Compare All** — run all 3 models side by side with radar chart

---

## Project Structure

```
emotion_recognition/
├── app.py                          # Streamlit deployment app
├── requirements.txt
├── README.md
├── utils/
│   ├── dataset.py                  # TESS loader, feature extraction, datasets
│   └── trainer.py                  # Training helpers, metrics, plots
├── models/
│   ├── speech_pipeline/
│   │   ├── model.py                # CNN-BiLSTM-Attention
│   │   ├── train.py
│   │   └── test.py
│   ├── text_pipeline/
│   │   ├── model.py                # BERT fine-tuned
│   │   ├── train.py
│   │   └── test.py
│   └── fusion_pipeline/
│       ├── model.py                # Cross-Modal Attention Fusion
│       ├── train.py
│       └── test.py
├── checkpoints/                    # Saved model weights
└── Results/
    ├── plots/                      # Loss curves, confusion matrices
    └── tables/                     # CSV histories, accuracy JSONs
```

---

## Expected Results (TESS Dataset)

| Model | Expected Accuracy |
|---|---|
| Speech Only | ~85–90% |
| Text Only | ~75–82% |
| Fusion | ~90–95% |

---

## Deployment on Streamlit Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo
4. Set main file as `app.py`
5. Deploy!

> **Note:** Pre-train models locally, then push checkpoints to the repo before deploying.
