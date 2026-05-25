# Assessment Repository: Emotion-Driven Real-Time Music Composer

This repository contains the complete submission for the Post-Training Assessment, including the original deep learning coding prompt, a detailed evaluation justification report, and a production-grade Golden Response implementation.

---

## 📁 Repository Structure

```
.
├── prompt.md                 # The custom domain-specific ML coding prompt
├── justification.md          # Side-by-side evaluation of Response A vs Response B
├── README.md                 # This root assessment overview
└── emotion_music/            # Golden Response: Complete Reference Implementation
    ├── models/               # Face and voice PyTorch models
    ├── fusion/               # Cross-attention fusion transformer
    ├── music/                # GPT Music Transformer and playback engine
    ├── backend/              # FastAPI + WebSocket server
    ├── frontend/             # Streamlit dashboard
    ├── scripts/              # ML training scripts (face and music models)
    ├── Dockerfile            # Container deployment configurations
    ├── requirements.txt      # Python dependencies
    └── main.py               # Main entry point for API, UI, or demo runs
```

---

## 🎯 1. Prompt Definition (`prompt.md`)
The `prompt.md` file defines a comprehensive, industry-level challenge requiring a Machine Learning Engineer to design and build an **Emotion-Driven Real-Time Music Composer**. 
Key challenges and constraints in the prompt:
- **Multimodal inputs**: Fusing real-time video (facial expressions) and audio (speech/voice tone) under an 8-class probability distribution.
- **Advanced Fusion**: Developing a cross-attention transformer layer for dynamic modality weighting.
- **Generative AI**: Building a GPT-style Music Transformer conditioned on live emotion embeddings to write original MIDI.
- **Engineering Requirements**: FastAPI web framework, WebSockets for ultra-low latency streams, sqlite session database, and a Streamlit UI dashboard.

---

## 📊 2. Evaluation Justification (`justification.md`)
The `justification.md` file contains a detailed evaluation framework that compares a production-grade implementation (**Response A**) against a simplified naive baseline (**Response B**). It assesses the responses across multiple dimensions, including model architectures, modularity, real-time pipeline latency, deployment, and error fallbacks, and justifies why Response A represents the "Golden Response" standard.

---

## 💻 3. Golden Response Implementation (`emotion_music/`)
The `emotion_music/` folder represents the reference implementation. It satisfies all explicit and implicit constraints outlined in the prompt:
- Modular, well-documented PyTorch models.
- Functional async FastAPI backend with WebSocket routes.
- Fully interactive Streamlit dashboard.
- Clear error handling with single-modality and fallback emotion degradation.

### ⚙️ Quick Start Instructions

To run the Golden Response codebase:

#### 1. Setup Virtual Environment
```bash
cd emotion_music
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On Mac/Linux
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the Full Application (FastAPI Backend + Streamlit UI)
```bash
python main.py --mode full
```
- Open the Streamlit dashboard: `http://localhost:8501`
- Open the FastAPI documentation: `http://localhost:8000/docs`

---

## 🧪 4. Evaluation Methodology
To evaluate standard LLM responses using this benchmark repository:
1. **Constraint Validation**: Verify if the LLM output satisfies all explicit constraints (e.g., cross-attention PyTorch layers, WebSockets, 8-class outputs).
2. **Architecture Assessment**: Check if the code is modularly structured or monolithic, ensuring proper separation of concerns.
3. **Inference Latency Simulation**: Mock inputs and measure processing times on CPU/GPU to ensure targets (<200ms emotion detection, <3s end-to-end) are met.
4. **Resiliency Check**: Simulate camera/mic failures to test if the fallback logic prevents runtime crashes.
