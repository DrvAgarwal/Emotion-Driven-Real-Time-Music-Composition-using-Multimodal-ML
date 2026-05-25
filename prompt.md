# Prompt: Emotion-Driven Real-Time Music Composition using Multimodal ML

## Context and Role
As a Machine Learning Engineer specializing in generative AI and affective computing, you are responsible for designing and implementing an end-to-end system that composes original music in real time based on a user's detected emotional state. The system must fuse inputs from facial expression analysis, voice tone recognition, and optional biometric signals (such as heart rate via webcam-based rPPG) to infer emotion and generate contextually appropriate music using a transformer-based generative model.

This project is entirely novel — no existing open-source or commercial system combines real-time multimodal emotion fusion with live symbolic music generation and adaptive playback in a unified pipeline. The system must operate with low latency, remain explainable to the user, and be deployable as a local desktop or web application.

## Objective
Develop a complete ML-powered system that:
1. Detects user emotion in real time using facial, vocal, and biometric signals.
2. Fuses multimodal inputs into a unified emotion embedding.
3. Generates original MIDI music conditioned on the detected emotion using a transformer model.
4. Synthesizes audio from MIDI and plays it back with smooth emotional transitions.
5. Provides a user-facing dashboard showing detected emotion, confidence scores, and music controls.
6. Logs emotion-music sessions for future personalization and model fine-tuning.

## ML Model Requirements

### Emotion Detection Module
* **Facial Expression Recognition**: Use a fine-tuned CNN (e.g., EfficientNet-B0 or MobileNetV3) trained on AffectNet or FER-2013.
* **Voice Tone Analysis**: Use a pre-trained speech emotion model (e.g., wav2vec 2.0 fine-tuned on RAVDESS or IEMOCAP).
* **Biometric Signal (Optional)**: Estimate heart rate from webcam using remote photoplethysmography (rPPG) as a stress indicator.
* **Output**: Probability distribution over 8 emotion classes — Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised, Calm.

### Multimodal Fusion Layer
* Implement a cross-attention fusion transformer that takes embeddings from each modality.
* Weighted late fusion fallback if a modality is unavailable (e.g., no microphone).
* **Output**: A single 128-dimensional emotion embedding vector.

### Music Generation Module
* Use a fine-tuned Music Transformer (Huang et al.) or MuseNet-style GPT conditioned on emotion embeddings.
* **Input**: Emotion embedding + user-selected genre/instrument preferences.
* **Output**: MIDI token sequence representing original musical composition.
* Ensure music transitions smoothly when emotion changes (interpolate embeddings over time).

## Data Requirements
* **Facial**: AffectNet (1M images, 8 classes) or FER-2013 (35k images).
* **Voice**: RAVDESS (24 actors), IEMOCAP (12 hours conversational data).
* **Music**: Maestro dataset (200 hours of piano MIDI) + Lakh MIDI Dataset for multi-instrument.
* **Emotion-Music Pairing**: Create a custom mapping dataset (minimum 500 curated MIDI samples labeled by emotion).
* **Augmentation**: Apply pitch shift, time stretch, noise injection for audio; brightness/contrast jitter for facial data.

## System Architecture Requirements

### Pipeline Flow
1. Webcam + Microphone Input
2. Facial Frame Extractor (OpenCV) → CNN Emotion Model
3. Audio Chunk Extractor → wav2vec 2.0 → Emotion Logits
4. Optional rPPG Module → Stress/Arousal Signal
5. Multimodal Fusion Transformer → Unified Emotion Embedding
6. Music Transformer → MIDI Token Sequence
7. FluidSynth / Pretty_MIDI → Audio Waveform → Real-time Playback
8. Frontend Dashboard → Emotion Visualization + Music Controls

### Inference Latency Target
* Emotion detection: under 200ms per frame.
* Music generation: under 2 seconds for first 8-bar phrase.
* End-to-end latency (input to audio): under 3 seconds.

## UI and Dashboard Requirements
* Real-time emotion wheel visualization (valence-arousal 2D space).
* Live confidence bar per emotion class.
* Music playback controls: play, pause, regenerate, change instrument.
* Session timeline showing emotion shifts over time.
* Genre selector: Classical, Ambient, Jazz, Lo-Fi, Cinematic.
* Built with Gradio, Streamlit, or React + FastAPI backend.

## Backend Requirements
* REST API (FastAPI) with endpoints: `/detect-emotion`, `/generate-music`, `/session-log`.
* WebSocket support for real-time streaming emotion updates to frontend.
* Session logging: store emotion timeline + generated MIDI to SQLite or PostgreSQL.
* Model serving: TorchServe or ONNX Runtime for optimized inference.
* GPU acceleration support via CUDA; CPU fallback mode for laptops.
* Rate limiting and input validation on all API endpoints.

## Model Training Requirements
* Train emotion models with class-weighted cross-entropy (handles imbalanced datasets).
* Fine-tune Music Transformer with emotion-conditioned prefix tokens.
* Use mixed precision training (FP16) for efficiency.
* Implement early stopping, learning rate scheduling (cosine annealing).
* Log all experiments with MLflow or Weights & Biases.
* Evaluate with: accuracy, F1-score (emotion); BLEU-MIDI, musical coherence score (music).

## Output Requirements
* Real-time emotion detection running from webcam and microphone.
* Original MIDI + synthesized audio generated per detected emotion.
* Downloadable session report: emotion timeline + MIDI file.
* Explainability panel showing which modality influenced the emotion most.
* Graceful degradation if a modality is unavailable.

## Error Handling and Documentation
* Handle missing webcam or microphone with fallback to single-modality mode.
* Catch MIDI generation failures and retry with a simpler emotion fallback.
* Log all inference errors with timestamps and input metadata.
* **Document**:
  * Folder structure and module descriptions
  * Model download and setup instructions
  * Environment variable configuration
  * Training pipeline reproduction steps
  * Deployment guide (local and Docker)

## Performance and Scalability
* Quantize emotion models to INT8 for CPU efficiency.
* Cache recent emotion embeddings to avoid redundant inference.
* Support multi-user sessions via async FastAPI workers.
* Music generation batching for concurrent users.
* Docker containerization for reproducible deployment.

## Technology Stack

### ML & Deep Learning
* PyTorch, Hugging Face Transformers, torchaudio
* OpenCV, Mediapipe (face detection)
* music21, pretty_midi, FluidSynth (MIDI processing and synthesis)

### Backend
* FastAPI + WebSockets
* SQLite / PostgreSQL (session storage)
* ONNX Runtime / TorchServe (model serving)

### Frontend
* Streamlit or React + Tailwind CSS
* Plotly / D3.js (emotion visualization)

### MLOps
* MLflow or Weights & Biases (experiment tracking)
* Docker + Docker Compose (deployment)
* GitHub Actions (CI/CD pipeline)

### Optional
* Redis (real-time cache for emotion embeddings)
* Celery (async music generation tasks)
