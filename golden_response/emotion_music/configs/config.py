import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Device
    DEVICE                  = os.getenv("DEVICE", "cpu")
    USE_GPU                 = os.getenv("USE_GPU", "false").lower() == "true"

    # Model paths
    FACE_MODEL_PATH         = os.getenv("FACE_MODEL_PATH", "models/face_emotion.pth")
    VOICE_MODEL_PATH        = os.getenv("VOICE_MODEL_PATH", "models/voice_emotion.pth")
    FUSION_MODEL_PATH       = os.getenv("FUSION_MODEL_PATH", "models/fusion.pth")
    MUSIC_MODEL_PATH        = os.getenv("MUSIC_MODEL_PATH", "models/music_transformer.pth")

    # API
    API_HOST                = os.getenv("API_HOST", "0.0.0.0")
    API_PORT                = int(os.getenv("API_PORT", 8000))
    SECRET_KEY              = os.getenv("SECRET_KEY", "secret")

    # Database
    DATABASE_URL            = os.getenv("DATABASE_URL", "sqlite:///./logs/sessions.db")

    # Audio
    SAMPLE_RATE             = int(os.getenv("SAMPLE_RATE", 16000))
    AUDIO_CHUNK_DURATION    = int(os.getenv("AUDIO_CHUNK_DURATION", 2))
    SOUNDFONT_PATH          = os.getenv("SOUNDFONT_PATH", "data/soundfont.sf2")

    # Music Generation
    MAX_MIDI_TOKENS         = int(os.getenv("MAX_MIDI_TOKENS", 512))
    MUSIC_TEMPERATURE       = float(os.getenv("MUSIC_TEMPERATURE", 0.95))
    EMOTION_EMBEDDING_DIM   = int(os.getenv("EMOTION_EMBEDDING_DIM", 128))

    # Emotions
    EMOTION_CLASSES         = [
        "Neutral", "Happy", "Sad", "Angry",
        "Fearful", "Disgusted", "Surprised", "Calm"
    ]
    NUM_EMOTIONS            = len(EMOTION_CLASSES)

    # Logging
    LOG_LEVEL               = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR                 = os.getenv("LOG_DIR", "logs/")

config = Config()
