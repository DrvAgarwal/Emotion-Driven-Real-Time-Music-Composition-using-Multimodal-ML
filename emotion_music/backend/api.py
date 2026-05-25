"""
FastAPI Backend
Endpoints: /detect-emotion, /generate-music, /session-log
WebSocket: /ws/{session_id} for real-time emotion streaming
"""

import os
import uuid
import time
import logging
import asyncio
import base64
import json
import numpy as np
import cv2
from datetime import datetime
from typing import Optional

from fastapi import (FastAPI, WebSocket, WebSocketDisconnect,
                     HTTPException, UploadFile, File, Depends)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from configs.config import config
from models.face_emotion import FaceEmotionDetector
from models.voice_emotion import VoiceEmotionDetector
from fusion.emotion_fusion import EmotionFusion
from music.music_transformer import MusicGenerator
from music.audio_player import AudioPlayer
from utils.database import SessionLogger

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotion-Driven Music Composer API",
    description="Real-time emotion detection and music generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Lazy-loaded singletons ───────────────────────────────────────────────────

_face_detector  = None
_voice_detector = None
_fusion         = None
_music_gen      = None
_audio_player   = None
_db_logger      = None

def get_face_detector() -> FaceEmotionDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceEmotionDetector()
    return _face_detector

def get_voice_detector() -> VoiceEmotionDetector:
    global _voice_detector
    if _voice_detector is None:
        _voice_detector = VoiceEmotionDetector()
    return _voice_detector

def get_fusion() -> EmotionFusion:
    global _fusion
    if _fusion is None:
        _fusion = EmotionFusion()
    return _fusion

def get_music_gen() -> MusicGenerator:
    global _music_gen
    if _music_gen is None:
        _music_gen = MusicGenerator()
    return _music_gen

def get_audio_player() -> AudioPlayer:
    global _audio_player
    if _audio_player is None:
        _audio_player = AudioPlayer()
    return _audio_player

def get_db() -> SessionLogger:
    global _db_logger
    if _db_logger is None:
        _db_logger = SessionLogger()
    return _db_logger


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class EmotionDetectRequest(BaseModel):
    frame_b64:  str             # base64 encoded JPEG frame
    audio_b64:  Optional[str]   # base64 encoded WAV audio chunk
    session_id: Optional[str]

class GenerateMusicRequest(BaseModel):
    session_id:     str
    emotion_result: dict
    genre:          str = "Classical"
    save_midi:      bool = True

class SessionLogRequest(BaseModel):
    session_id: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def decode_frame(b64_str: str) -> np.ndarray:
    data    = base64.b64decode(b64_str)
    arr     = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def decode_audio(b64_str: str) -> np.ndarray:
    import io
    import soundfile as sf
    data    = base64.b64decode(b64_str)
    audio, _= sf.read(io.BytesIO(data))
    return audio.astype(np.float32)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/detect-emotion")
async def detect_emotion(req: EmotionDetectRequest):
    """
    Accepts a base64 webcam frame + optional audio chunk.
    Returns fused emotion result.
    """
    try:
        face_detector   = get_face_detector()
        voice_detector  = get_voice_detector()
        fusion          = get_fusion()

        # Decode frame
        frame       = decode_frame(req.frame_b64)
        face_result = face_detector.predict(frame)

        # Decode audio if provided
        if req.audio_b64:
            audio           = decode_audio(req.audio_b64)
            voice_result    = voice_detector.predict(audio)
        else:
            voice_result    = voice_detector._empty_result()

        # Fuse
        fusion_result = fusion.fuse(face_result, voice_result)

        # Log to DB if session_id provided
        if req.session_id:
            db = get_db()
            db.log_emotion(req.session_id, fusion_result)

        return {
            "success":      True,
            "session_id":   req.session_id,
            "result":       fusion_result,
            "face":         face_result,
            "voice":        voice_result
        }

    except Exception as e:
        logger.exception("detect-emotion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-music")
async def generate_music(req: GenerateMusicRequest):
    """
    Generates original MIDI from emotion embedding.
    Returns MIDI file path and music metadata.
    """
    try:
        music_gen   = get_music_gen()
        audio_player= get_audio_player()
        db          = get_db()

        os.makedirs("outputs", exist_ok=True)
        midi_path   = f"outputs/{req.session_id}_music.mid" if req.save_midi else None

        result = music_gen.generate(
            fusion_result=req.emotion_result,
            genre=req.genre,
            output_path=midi_path
        )

        # Synthesize and play
        wav_path = audio_player.play_midi(result["midi"], session_id=req.session_id)
        result["wav_path"] = wav_path

        # Log
        db.log_music(req.session_id, result)
        db.update_session(req.session_id, midi_path=midi_path, wav_path=wav_path)

        return {
            "success":      True,
            "session_id":   req.session_id,
            "emotion":      result["emotion"],
            "genre":        result["genre"],
            "tempo":        result["tempo"],
            "num_tokens":   result["num_tokens"],
            "midi_path":    midi_path,
            "wav_path":     wav_path
        }

    except Exception as e:
        logger.exception("generate-music failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/create")
def create_session(genre: str = "Classical"):
    session_id = str(uuid.uuid4())
    db = get_db()
    db.create_session(session_id, genre=genre)
    return {"session_id": session_id, "genre": genre}


@app.get("/session-log/{session_id}")
def get_session_log(session_id: str):
    db       = get_db()
    timeline = db.get_session_timeline(session_id)
    return {"session_id": session_id, "timeline": timeline, "count": len(timeline)}


@app.get("/download/midi/{session_id}")
def download_midi(session_id: str):
    path = f"outputs/{session_id}_music.mid"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="MIDI not found")
    return FileResponse(path, media_type="audio/midi",
                        filename=f"emotion_music_{session_id}.mid")


@app.get("/download/wav/{session_id}")
def download_wav(session_id: str):
    path = f"outputs/{session_id}_music.wav"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="WAV not found")
    return FileResponse(path, media_type="audio/wav",
                        filename=f"emotion_music_{session_id}.wav")


@app.post("/playback/stop")
def stop_playback():
    audio_player = get_audio_player()
    audio_player.stop()
    return {"success": True, "message": "Playback stopped"}


@app.post("/playback/pause")
def pause_playback():
    get_audio_player().pause()
    return {"success": True}


@app.post("/playback/resume")
def resume_playback():
    get_audio_player().resume()
    return {"success": True}


# ── WebSocket ────────────────────────────────────────────────────────────────

active_connections: dict = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time emotion streaming.
    Client sends base64 frames; server returns emotion results.
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info("WebSocket connected: %s", session_id)

    face_detector   = get_face_detector()
    voice_detector  = get_voice_detector()
    fusion          = get_fusion()

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            frame_b64   = payload.get("frame_b64")
            audio_b64   = payload.get("audio_b64")

            if not frame_b64:
                continue

            frame       = decode_frame(frame_b64)
            face_result = face_detector.predict(frame)

            if audio_b64:
                audio           = decode_audio(audio_b64)
                voice_result    = voice_detector.predict(audio)
            else:
                voice_result    = voice_detector._empty_result()

            fusion_result = fusion.fuse(face_result, voice_result)

            db = get_db()
            db.log_emotion(session_id, fusion_result)

            await websocket.send_text(json.dumps({
                "type":         "emotion_update",
                "session_id":   session_id,
                "result":       fusion_result,
                "timestamp":    datetime.utcnow().isoformat()
            }))

    except WebSocketDisconnect:
        active_connections.pop(session_id, None)
        logger.info("WebSocket disconnected: %s", session_id)
    except Exception as e:
        logger.error("WebSocket error [%s]: %s", session_id, e)
        active_connections.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app",
                host=config.API_HOST,
                port=config.API_PORT,
                reload=True)
