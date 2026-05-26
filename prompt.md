# Prompt: Emotion-Driven Real-Time Music Composition using Multimodal ML

## Context and Role
You're a machine learning engineer who knows generative AI and affective computing — that's the field where computers try to understand human emotions. Your job is to build something pretty wild: a system that watches a person's face, listens to their voice, and then composes music on the spot that matches how they're feeling. Like a musical chameleon.

The system needs to take in video from a webcam, audio from a mic, and optionally even estimate heart rate from the webcam feed (there's a clever trick called rPPG that does that). Then it blends all that information together, figures out the emotion, and generates original music using a transformer model. And here's the thing — nobody's really built an end‑to‑end system like this before. So you're basically making something new.

It has to be fast, it has to be able to explain itself to the user ("why did you play happy music?"), and it should work as a local desktop app or a web app.

## Objective
Build a complete ML system that:
* Detects emotion in real time from face, voice, and optional biometric signals
* Merges those inputs into a single emotion "embedding" (just a fancy vector of numbers that represents the feeling)
* Generates original MIDI music based on that emotion using a transformer
* Turns that MIDI into actual sound and plays it back, with smooth transitions when the emotion changes
* Shows the user a live dashboard with the detected emotion, confidence scores, and music controls
* Logs everything so you can later personalize the model or fine‑tune it

## ML Model Requirements

### Emotion Detection Module
* **Face**: Use a fine‑tuned CNN like EfficientNet‑B0 or MobileNetV3, trained on AffectNet or FER-2013.
* **Voice**: Use something like wav2vec 2.0 fine‑tuned on RAVDESS or IEMOCAP.
* **Biometric (optional)**: Estimate heart rate from the webcam using rPPG — think of it as a stress indicator.
* **Output**: Probabilities for 8 emotions: Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised, Calm.

### Multimodal Fusion Layer
* Build a cross‑attention fusion transformer that takes the embeddings from each modality (face, voice, biometrics).
* If a modality is missing (say, no microphone), fall back to weighted late fusion — still works, just less accurate.
* Output a single 128‑dimensional emotion embedding.

### Music Generation Module
* Use a fine‑tuned Music Transformer (Huang et al.) or a MuseNet‑style GPT, conditioned on the emotion embedding.
* **Inputs**: emotion embedding + what the user wants (genre, instrument).
* **Output**: a sequence of MIDI tokens that represent original music.
* **Smooth transitions**: when the emotion changes, interpolate the embeddings over time so the music doesn't jump around awkwardly.

## Data Requirements
* **Face**: AffectNet (1 million images) or FER‑2013 (35k images).
* **Voice**: RAVDESS (24 actors), IEMOCAP (12 hours of conversation).
* **Music**: Maestro (200 hours of piano MIDI) + Lakh MIDI Dataset (multi‑instrument).
* **Emotion‑Music pairing**: Create your own custom dataset — at least 500 MIDI samples labeled by emotion.
* **Augmentation**: For audio, use pitch shift, time stretch, noise injection. For face, brightness/contrast jitter.

## System Architecture Requirements

### Pipeline Flow (step‑by‑step)
1. Webcam + microphone start capturing
2. Face frames go through OpenCV, then a CNN emotion model
3. Audio chunks go through wav2vec 2.0, get emotion logits
4. Optional rPPG module gives stress/arousal signal
5. Multimodal fusion transformer produces one unified emotion embedding
6. Music transformer turns that into MIDI tokens
7. FluidSynth / pretty_midi converts to audio, plays in real time
8. Frontend dashboard shows emotion visualization + music controls

### Latency Targets (non‑negotiable)
* **Emotion detection**: under 200ms per frame
* **Music generation (first 8 bars)**: under 2 seconds
* **End‑to‑end from input to audio**: under 3 seconds

## UI and Dashboard Requirements
The dashboard needs to show:
* A real‑time emotion wheel (valence‑arousal 2D space)
* Live confidence bars for each of the 8 emotions
* Music playback controls: play, pause, regenerate, change instrument
* A session timeline showing how emotion shifts over time
* Genre selector: Classical, Ambient, Jazz, Lo-Fi, Cinematic

You can build this with Gradio, Streamlit, or React + FastAPI.

## Backend Requirements
* REST API with FastAPI — endpoints: `/detect‑emotion`, `/generate‑music`, `/session‑log`
* WebSocket support for streaming real‑time emotion updates to the frontend
* Session logging to SQLite or PostgreSQL (store emotion timeline + generated MIDI)
* Model serving via TorchServe or ONNX Runtime for optimized inference
* GPU acceleration (CUDA) and CPU fallback for laptops
* Rate limiting and input validation on all API endpoints

## Model Training Requirements
* Train emotion models with class‑weighted cross‑entropy (because datasets are often imbalanced)
* Fine‑tune Music Transformer with emotion‑conditioned prefix tokens
* Use mixed precision training (FP16) to save memory and speed things up
* Early stopping and cosine annealing learning rate scheduler
* Log every experiment with MLflow or Weights & Biases
* **Evaluate with**: accuracy, F1‑score (for emotion); BLEU‑MIDI and a musical coherence score (for music)

## Output Requirements
* Real‑time emotion detection running from webcam + mic
* Original MIDI + synthesized audio generated per detected emotion
* Downloadable session report: emotion timeline + MIDI file
* **Explainability panel** showing which modality (face, voice, or rPPG) influenced the emotion the most
* **Graceful degradation** — if a modality is missing, the system still works fine

## Error Handling and Documentation
* If webcam or microphone is missing, fall back to single‑modality mode and tell the user
* If MIDI generation fails, catch it and retry with a simpler emotion fallback
* Log all inference errors with timestamps and input metadata
* **Documentation must include**:
  * Folder structure and module descriptions
  * Model download and setup instructions
  * Environment variable configuration
  * Training pipeline reproduction steps
  * Deployment guide (local and Docker)

## Performance and Scalability
* Quantize emotion models to INT8 for CPU efficiency
* Cache recent emotion embeddings to avoid redundant inference
* Support multi‑user sessions via async FastAPI workers
* Batch music generation for concurrent users
* Docker containerization for reproducible deployment

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

### Optional but helpful
* Redis for real‑time cache of emotion embeddings
* Celery for async music generation tasks
