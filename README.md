# Assessment Repository: Emotion-Driven Real-Time Music Composer

This repository represents the complete, high-fidelity submission for the Post-Training Assessment. It includes the original coding challenge prompt, the side-by-side evaluation verdict, and a production-grade Golden Response implementation.

---

## 📁 Repository Structure

```
.
├── prompt.md                 # The conversational domain-specific ML coding prompt
├── justification.md          # Structured evaluation report containing the Final Verdict
├── golden_response.py        # The self-contained Golden Response reference implementation
├── README.md                 # This root assessment overview
└── emotion_music/            # Fully modularized project packages
    ├── models/               # PyTorch models (Face CNN, Speech Wav2Vec2)
    ├── fusion/               # Cross-attention fusion transformer
    ├── music/                # GPT Music Transformer and sound synthesis player
    ├── backend/              # FastAPI + WebSocket backend
    ├── frontend/             # Streamlit dashboard
    ├── scripts/              # ML training pipelines (MLflow tracking)
    ├── Dockerfile            # Container deployment configurations
    ├── requirements.txt      # Python dependencies
    └── main.py               # Main modular application runner
```

---

## 🎯 1. Prompt Definition (`prompt.md`)
The `prompt.md` file defines a comprehensive, industry-level challenge requiring a Machine Learning Engineer to design and build a system that captures real-time video (face) and audio (speech), fuses them using a **Cross-Attention Transformer**, and generates original, emotionally conditioned MIDI music using a **Music Transformer**. It outlines latency targets, MLOps metrics, persistent SQLite logging, and error fallbacks.

---

## 📊 2. Evaluation Justification (`justification.md`)
The `justification.md` file contains the **Final Verdict** comparing a production-grade implementation (**Response A**) against a prototype-grade baseline (**Response B**):
> *Response A is better than Response B because A delivers a fully fleshed‑out, production ready system that checks every box  from the optional rPPG biometric module and properly implemented cross‑attention fusion transformer to complete training pipelines, session logging, explainability, and Docker deployment – while B, despite having a functional WebSocket and React scaffold, misses critical components like rPPG, session storage, the explainability panel, and any real training logic, plus its fusion layer doesn’t actually use cross‑attention as promised. In short, A gives you a complete blueprint you could build and ship; B gives you a nice demo shell with significant gaps .*

---

## 💻 3. Golden Response Reference (`golden_response.py`)
The `golden_response.py` file represents the ideal reference implementation. It integrates **100% of all coding files** of the project into a single, fully annotated, highly resilient script featuring:
- **Resilient Model Loaders**: Disabled PyTorch Hub pretraining downloads to ensure the system starts up completely offline or under restricted firewalls.
- **Audio Fallbacks**: Employs a custom, clean **ADSR sine-wave synthesizer** that takes over automatically if a system-level FluidSynth installation is not found.
- **Hardware Fallbacks**: Automatically degrades gracefully to single-modality mode if a camera or microphone is missing, and falls back to CPU if no CUDA GPU is present.

### ⚙️ Quick Start Instructions

To run the Golden Response codebase:

#### 1. Setup Virtual Environment
```bash
python -m venv emotion_music/venv
emotion_music\venv\Scripts\activate        # On Windows
source emotion_music/venv/bin/activate     # On Mac/Linux
```

#### 2. Install Dependencies
```bash
pip install -r emotion_music/requirements.txt
```

#### 3. Run the Application
The `golden_response.py` script supports flexible run modes depending on your evaluation setup:

* **Offline Webcam Demo Mode**:
  ```bash
  python golden_response.py --mode demo
  ```
* **FastAPI Backend Server**:
  ```bash
  python golden_response.py --mode server
  ```
* **Streamlit UI Dashboard**:
  ```bash
  python golden_response.py --mode ui
  ```
* **Full Stack (FastAPI Backend + Streamlit UI in Parallel)**:
  ```bash
  python golden_response.py --mode full
  ```
  * Once launched in `full` mode, open the UI at: [http://localhost:8501](http://localhost:8501)
  * Open the interactive API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)
