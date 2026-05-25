"""
Voice Emotion Detection Module
Uses wav2vec 2.0 base fine-tuned for 8-class speech emotion recognition.
"""

import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd
import soundfile as sf
import librosa
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from configs.config import config

logger = logging.getLogger(__name__)


# ── Model Definition ────────────────────────────────────────────────────────

class VoiceEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.wav2vec    = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_size     = self.wav2vec.config.hidden_size        # 768

        # Freeze first 6 transformer layers — fine-tune the rest
        for i, layer in enumerate(self.wav2vec.encoder.layers):
            if i < 6:
                for p in layer.parameters():
                    p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.embedding_head = nn.Linear(hidden_size, 128)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask
        )
        # Mean pool over time
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)

    def get_embedding(self, input_values, attention_mask=None):
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.embedding_head(hidden)


# ── Voice Detector ────────────────────────────────────────────────────────

class VoiceEmotionDetector:
    def __init__(self):
        self.device     = torch.device(config.DEVICE)
        self.sr         = config.SAMPLE_RATE
        self.duration   = config.AUDIO_CHUNK_DURATION
        self.emotions   = config.EMOTION_CLASSES
        self.processor  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model      = self._load_model()
        self.available  = self._check_mic()
        logger.info("VoiceEmotionDetector initialized. Mic available: %s", self.available)

    def _load_model(self) -> VoiceEmotionModel:
        model = VoiceEmotionModel(num_classes=config.NUM_EMOTIONS)
        if config.VOICE_MODEL_PATH and torch.os.path.exists(config.VOICE_MODEL_PATH):
            state = torch.load(config.VOICE_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state)
            logger.info("Loaded voice model from %s", config.VOICE_MODEL_PATH)
        else:
            logger.warning("No pretrained voice weights — using base wav2vec2 init.")
        model.to(self.device)
        model.eval()
        return model

    def _check_mic(self) -> bool:
        try:
            devices = sd.query_devices()
            return any(d['max_input_channels'] > 0 for d in devices)
        except Exception:
            return False

    def record_chunk(self) -> np.ndarray:
        """Records AUDIO_CHUNK_DURATION seconds from microphone."""
        if not self.available:
            return None
        try:
            audio = sd.rec(
                int(self.duration * self.sr),
                samplerate=self.sr,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            return audio.squeeze()
        except Exception as e:
            logger.error("Mic recording failed: %s", e)
            return None

    def load_audio_file(self, path: str) -> np.ndarray:
        """Load audio from file — useful for testing."""
        audio, _ = librosa.load(path, sr=self.sr)
        return audio

    def preprocess(self, audio: np.ndarray) -> dict:
        inputs = self.processor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def predict(self, audio: np.ndarray = None) -> dict:
        """
        If audio is None, records from mic.
        Returns emotion probabilities and embedding.
        """
        if audio is None:
            audio = self.record_chunk()

        if audio is None or len(audio) == 0:
            return self._empty_result()

        try:
            inputs      = self.preprocess(audio)
            logits      = self.model(**inputs)
            probs       = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            embedding   = self.model.get_embedding(**inputs).squeeze().cpu().numpy()

            return {
                "probabilities": {e: float(p) for e, p in zip(self.emotions, probs)},
                "dominant_emotion": self.emotions[int(np.argmax(probs))],
                "confidence": float(np.max(probs)),
                "embedding": embedding.tolist(),
                "voice_detected": True
            }
        except Exception as e:
            logger.error("Voice prediction failed: %s", e)
            return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "probabilities": {e: 1/8 for e in self.emotions},
            "dominant_emotion": "Neutral",
            "confidence": 0.0,
            "embedding": [0.0] * 128,
            "voice_detected": False
        }
