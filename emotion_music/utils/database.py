"""
Database Models and Session Logging
SQLAlchemy models for storing emotion sessions and MIDI outputs.
"""

import os
import json
import logging
from datetime import datetime
from sqlalchemy import (create_engine, Column, Integer, String,
                        Float, Text, DateTime, JSON)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from configs.config import config

logger  = logging.getLogger(__name__)
Base    = declarative_base()
os.makedirs("logs", exist_ok=True)
engine  = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── ORM Models ──────────────────────────────────────────────────────────────

class EmotionSession(Base):
    __tablename__ = "emotion_sessions"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(String, unique=True, index=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    genre           = Column(String, default="Classical")
    total_duration  = Column(Float, default=0.0)
    midi_path       = Column(String, nullable=True)
    wav_path        = Column(String, nullable=True)


class EmotionEvent(Base):
    __tablename__ = "emotion_events"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(String, index=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    dominant_emotion= Column(String)
    confidence      = Column(Float)
    probabilities   = Column(JSON)
    dominant_source = Column(String)
    face_weight     = Column(Float)
    voice_weight    = Column(Float)
    embedding_norm  = Column(Float)


class MusicEvent(Base):
    __tablename__ = "music_events"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(String, index=True)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    emotion         = Column(String)
    genre           = Column(String)
    tempo           = Column(Float)
    num_tokens      = Column(Integer)
    midi_path       = Column(String, nullable=True)


# ── Create Tables ────────────────────────────────────────────────────────────

Base.metadata.create_all(bind=engine)


# ── Session Logger ────────────────────────────────────────────────────────────

class SessionLogger:
    def __init__(self):
        self.db = SessionLocal()
        logger.info("SessionLogger initialized with DB: %s", config.DATABASE_URL)

    def create_session(self, session_id: str, genre: str = "Classical") -> EmotionSession:
        s = EmotionSession(session_id=session_id, genre=genre)
        self.db.add(s)
        self.db.commit()
        self.db.refresh(s)
        return s

    def log_emotion(self, session_id: str, fusion_result: dict) -> EmotionEvent:
        import numpy as np
        emb_norm = float(np.linalg.norm(fusion_result.get("embedding", [0])))
        event = EmotionEvent(
            session_id      = session_id,
            dominant_emotion= fusion_result.get("dominant_emotion", "Neutral"),
            confidence      = fusion_result.get("confidence", 0.0),
            probabilities   = fusion_result.get("probabilities", {}),
            dominant_source = fusion_result.get("dominant_source", "none"),
            face_weight     = fusion_result.get("face_weight", 0.0),
            voice_weight    = fusion_result.get("voice_weight", 0.0),
            embedding_norm  = emb_norm
        )
        self.db.add(event)
        self.db.commit()
        return event

    def log_music(self, session_id: str, music_result: dict) -> MusicEvent:
        event = MusicEvent(
            session_id  = session_id,
            emotion     = music_result.get("emotion", "Neutral"),
            genre       = music_result.get("genre", "Classical"),
            tempo       = float(music_result.get("tempo", 100)),
            num_tokens  = music_result.get("num_tokens", 0),
            midi_path   = music_result.get("output_path")
        )
        self.db.add(event)
        self.db.commit()
        return event

    def get_session_timeline(self, session_id: str) -> list:
        events = (self.db.query(EmotionEvent)
                  .filter(EmotionEvent.session_id == session_id)
                  .order_by(EmotionEvent.timestamp)
                  .all())
        return [
            {
                "timestamp":        e.timestamp.isoformat(),
                "dominant_emotion": e.dominant_emotion,
                "confidence":       e.confidence,
                "probabilities":    e.probabilities,
                "dominant_source":  e.dominant_source
            }
            for e in events
        ]

    def update_session(self, session_id: str, **kwargs):
        s = self.db.query(EmotionSession).filter(
            EmotionSession.session_id == session_id
        ).first()
        if s:
            for k, v in kwargs.items():
                setattr(s, k, v)
            self.db.commit()

    def close(self):
        self.db.close()
