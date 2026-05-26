# 🎵 Emotion-Driven Real-Time Music Composer

A unique ML system that detects your emotion from face + voice in real time
and generates original MIDI music conditioned on that emotion using a
transformer model — composed fresh every time, never retrieved.

---

## 📁 Project Structure

```
emotion_music/
│
├── configs/
│   └── config.py               # All config loaded from .env
│
├── models/
│   ├── face_emotion.py         # EfficientNet-B0 face emotion detector
│   └── voice_emotion.py        # wav2vec 2.0 voice emotion detector
│
├── fusion/
│   └── emotion_fusion.py       # Cross-attention multimodal fusion transformer
│
├── music/
│   ├── music_transformer.py    # GPT-style Music Transformer (MIDI generation)
│   └── audio_player.py         # MIDI → WAV synthesis + pygame playback
│
├── backend/
│   └── api.py                  # FastAPI: REST endpoints + WebSocket
│
├── frontend/
│   └── app.py                  # Streamlit dashboard
│
├── utils/
│   └── database.py             # SQLAlchemy session + emotion logging
│
├── scripts/
│   ├── train_face_model.py     # Train face emotion model
│   └── train_music_model.py    # Train music transformer
│
├── data/                       # Put your datasets here
├── models/                     # Saved model weights go here
├── outputs/                    # Generated MIDI and WAV files
├── logs/                       # SQLite DB + log files
│
├── main.py                     # Entry point (demo / server / ui / full)
├── requirements.txt
└── .env
```

---

## ⚙️ Setup Instructions

### 1. Clone and create virtual environment
```bash
git clone <your-repo>
cd emotion_music
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Edit `.env` file:
```
DEVICE=cpu
DATABASE_URL=sqlite:///./logs/sessions.db
SAMPLE_RATE=16000
MAX_MIDI_TOKENS=512
```

### 4. Run in demo mode (webcam required)
```bash
python main.py --mode demo
```

### 5. Run full stack (API + UI)
```bash
python main.py --mode full
```
Then open: http://localhost:8501

---

## 🏋️ Training the Models

### Face Emotion Model
Prepare a CSV file:
```
image_path,label
data/images/img001.jpg,1
data/images/img002.jpg,3
```
Label mapping: 0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Fearful, 5=Disgusted, 6=Surprised, 7=Calm

```bash
python scripts/train_face_model.py
```

### Music Transformer
Prepare a JSON file:
```json
[
  {"midi_path": "data/midi/happy_01.mid", "emotion": "Happy",  "genre": "Classical"},
  {"midi_path": "data/midi/sad_01.mid",   "emotion": "Sad",    "genre": "Ambient"}
]
```

```bash
python scripts/train_music_model.py
```

---

## 🌐 API Endpoints

| Method | Endpoint                     | Description                    |
|--------|------------------------------|--------------------------------|
| GET    | /health                      | Health check                   |
| POST   | /session/create              | Create new session             |
| POST   | /detect-emotion              | Detect emotion from frame      |
| POST   | /generate-music              | Generate music from emotion    |
| GET    | /session-log/{session_id}    | Get emotion timeline           |
| GET    | /download/midi/{session_id}  | Download MIDI file             |
| GET    | /download/wav/{session_id}   | Download WAV file              |
| POST   | /playback/stop               | Stop music playback            |
| WS     | /ws/{session_id}             | Real-time emotion streaming    |

---

## 🎭 Emotion → Music Mapping

| Emotion   | Tempo (BPM) | Key         | Character         |
|-----------|-------------|-------------|-------------------|
| Happy     | 140         | Major       | Bright, energetic |
| Sad       | 65          | Minor       | Slow, soft        |
| Angry     | 175         | Diminished  | Sharp, heavy      |
| Calm      | 72          | Major       | Smooth, wide      |
| Fearful   | 110         | Minor       | Tense, uncertain  |
| Disgusted | 90          | Minor       | Dark, irregular   |
| Surprised | 155         | Major       | Fast, unexpected  |
| Neutral   | 100         | Major       | Balanced          |

---

## 🐳 Docker Deployment

```bash
docker build -t emotion-music .
docker run -p 8000:8000 -p 8501:8501 emotion-music
```

---

## 📊 MLflow Experiment Tracking

```bash
mlflow ui --port 5000
```
Open: http://localhost:5000

---

## 🔧 Environment Variables

| Variable              | Default                          | Description                    |
|-----------------------|----------------------------------|--------------------------------|
| DEVICE                | cpu                              | cpu or cuda                    |
| FACE_MODEL_PATH       | models/face_emotion.pth          | Face model weights path        |
| VOICE_MODEL_PATH      | models/voice_emotion.pth         | Voice model weights path       |
| FUSION_MODEL_PATH     | models/fusion.pth                | Fusion model weights path      |
| MUSIC_MODEL_PATH      | models/music_transformer.pth     | Music model weights path       |
| API_HOST              | 0.0.0.0                          | FastAPI host                   |
| API_PORT              | 8000                             | FastAPI port                   |
| DATABASE_URL          | sqlite:///./logs/sessions.db     | Database connection string     |
| SAMPLE_RATE           | 16000                            | Audio sample rate              |
| MAX_MIDI_TOKENS       | 512                              | Max tokens per generation      |
| MUSIC_TEMPERATURE     | 0.95                             | Sampling temperature           |
| EMOTION_EMBEDDING_DIM | 128                              | Fusion embedding size          |

---

## 📝 Notes

- Without pretrained weights, the system runs in **random init mode** (demo purposes).
- For real results, train the models using the scripts provided.
- Windows CPU mode is fully supported — no GPU required.
- The music generation uses nucleus sampling (top-p=0.9) for diversity.
