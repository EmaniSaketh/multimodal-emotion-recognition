🎭 Multimodal Emotion Recognition System

Recognise human emotions using Speech · Text · Multimodal Fusion

PyTorch · CNN+BiLSTM · BERT · Streamlit · Deep Learning

🔗 Project Links
GitHub Repository

https://github.com/EmaniSaketh/multimodal-emotion-recognition

Live Demo

https://emanisaketh-multimodal-emotion-recognition-app-ojgbxw.streamlit.app/

🚀 Quick Start

Run these commands to start the project locally.

Step 1 — Clone Repository
git clone https://github.com/EmaniSaketh/multimodal-emotion-recognition.git

cd multimodal-emotion-recognition
Step 2 — Create Virtual Environment
Windows
python -m venv venv

venv\Scripts\activate
Linux / macOS
python3 -m venv venv

source venv/bin/activate
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Run Streamlit App
streamlit run app.py

Open:

http://localhost:8501
📌 Project Overview

This project implements a complete Multimodal Emotion Recognition System capable of detecting emotions using:

🎤 Speech
✍️ Text
🔀 Speech + Text Fusion

The project combines:

Speech Signal Processing
Natural Language Processing
Deep Learning
Multimodal AI Fusion
Real-Time Deployment
✨ Features
🎤 Speech Emotion Recognition
✍️ Text Emotion Recognition
🔀 Multimodal Fusion Prediction
📊 Interactive Streamlit Dashboard
⚡ Real-Time Inference
☁️ Cloud Deployment using Streamlit
🧠 Deep Learning based prediction
📁 Automatic Model Downloading from Google Drive
🧠 Architecture
🎤 Speech Pipeline
Audio Input
     │
     ▼
MFCC Feature Extraction
     │
     ▼
CNN Layers
     │
     ▼
BiLSTM Temporal Modelling
     │
     ▼
Dense Layers
     │
     ▼
Emotion Prediction
Technologies Used
Librosa
MFCC Features
CNN
BiLSTM
PyTorch
✍️ Text Pipeline
Text Input
     │
     ▼
Text Cleaning & Tokenization
     │
     ▼
BERT Tokenizer
     │
     ▼
Transformer/BERT Embeddings
     │
     ▼
Dense Layers
     │
     ▼
Emotion Prediction
Technologies Used
HuggingFace Transformers
BERT
PyTorch
🔀 Multimodal Fusion Pipeline
Speech Features        Text Features
       │                     │
       ▼                     ▼
Feature Projection Layers
              │
              ▼
Cross Modal Attention
              │
              ▼
Fusion Layer
              │
              ▼
Emotion Prediction
📚 Dataset
Toronto Emotional Speech Set (TESS)

The project uses the TESS Dataset for training and evaluation.

Dataset Information
Property	Value
Total Samples	5600
Emotions	7
Dataset Type	Speech Emotion Dataset
Format	WAV Audio
Labels	Emotion Categories
😃 Supported Emotions
Emotion	Emoji
Angry	😠
Disgust	🤢
Fear	😨
Happy	😊
Neutral	😐
Surprise	😲
Sad	😢
🛠️ Tech Stack
Category	Technology
Frontend	Streamlit
Backend	Python
Deep Learning	PyTorch
NLP	Transformers / BERT
Audio Processing	Librosa
Visualization	Plotly
Deployment	Streamlit Cloud
Version Control	Git & GitHub
📂 Project Structure
multimodal-emotion-recognition/
│
├── app.py
├── requirements.txt
├── README.md
│
├── checkpoints/
│   ├── speech_best.pt
│   ├── text_best.pt
│   └── fusion_best.pt
│
├── models/
│   ├── speech_pipeline/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test.py
│   │
│   ├── text_pipeline/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── fusion_pipeline/
│       ├── model.py
│       ├── train.py
│       └── test.py
│
└── utils/
⚡ Model Checkpoints

Due to large model sizes, checkpoints are hosted using Google Drive and automatically downloaded during runtime.

Models Included:

Speech Model
Text Model
Fusion Model
🖥️ Streamlit Demo

The Streamlit application provides:

Mode	Description
🎤 Speech	Upload audio and predict emotion
✍️ Text	Enter text and predict emotion
🔀 Multimodal	Combine speech + text prediction
📊 Example Predictions
🎤 Speech Prediction
Input
Audio File (.wav)
Output
Emotion: Happy
Confidence: 98%
✍️ Text Prediction
Input
"I am feeling amazing today!"
Output
Emotion: Happy
Confidence: 99%
🔀 Multimodal Prediction
Input
Audio File
Transcript/Text
Output
Emotion: Happy
Confidence: 99%
☁️ Deployment

The application is deployed using Streamlit Cloud.

Deployment Challenges Solved
Large Model Hosting
Cloud Inference Setup
Dynamic Model Downloading
Streamlit Deployment Optimization
Dependency Management
GitHub Repository Optimization
🔬 Key Learnings
Speech carries strong emotional information through pitch, energy, and tone.
BERT improves contextual understanding in text analysis.
Multimodal Fusion helps combine complementary information from speech and text.
Cloud deployment of AI systems requires dependency optimization and model hosting strategies.
🔮 Future Improvements
Real-time microphone recording
Webcam facial emotion recognition
Attention visualization
Mobile application deployment
API integration
HuggingFace deployment
📖 Research Areas Covered
Deep Learning
Natural Language Processing
Speech Signal Processing
Emotion AI
Multimodal Learning
Human Computer Interaction
👨‍💻 Author
Saketh Emani

B.Tech Computer Science Engineering Student

AI · Deep Learning · NLP · Multimodal AI

GitHub:
https://github.com/EmaniSaketh

📜 License

This project is for academic and research purposes.

🙏 Acknowledgements
PyTorch
HuggingFace Transformers
Streamlit
Librosa
TESS Dataset
Open Source AI Community