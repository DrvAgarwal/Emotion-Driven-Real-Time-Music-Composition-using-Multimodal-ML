"""
Main Entry Point
Runs the full Emotion-Driven Music Composer pipeline.
- Mode 1: demo   — runs offline demo without API
- Mode 2: server — starts FastAPI backend
- Mode 3: ui     — starts Streamlit frontend
- Mode 4: full   — starts both backend + frontend
"""

import sys
import os
import subprocess
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs",    exist_ok=True)
os.makedirs("data",    exist_ok=True)


# ── Demo Mode ─────────────────────────────────────────────────────────────────

def run_demo():
    """
    Offline demo: webcam → emotion detection → MIDI generation → playback.
    Does not require the API server.
    """
    import cv2
    import time
    from models.face_emotion import FaceEmotionDetector
    from models.voice_emotion import VoiceEmotionDetector
    from fusion.emotion_fusion import EmotionFusion
    from music.music_transformer import MusicGenerator
    from music.audio_player import AudioPlayer
    from utils.database import SessionLogger
    import uuid

    logger.info("🎵 Starting Emotion Music Composer — Demo Mode")

    face_detector   = FaceEmotionDetector()
    voice_detector  = VoiceEmotionDetector()
    fusion          = EmotionFusion()
    music_gen       = MusicGenerator()
    audio_player    = AudioPlayer()
    db              = SessionLogger()

    session_id      = str(uuid.uuid4())[:8]
    db.create_session(session_id, genre="Classical")
    logger.info("Session: %s", session_id)

    cap             = cv2.VideoCapture(0)
    last_music_time = 0
    music_interval  = 12   # seconds

    if not cap.isOpened():
        logger.error("Cannot open webcam. Running in headless demo mode.")
        _run_headless_demo(music_gen, audio_player)
        return

    logger.info("Press 'q' to quit | 'r' to regenerate music | 's' to save MIDI")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_result     = face_detector.predict(frame)
            voice_result    = voice_detector._empty_result()  # No blocking mic in demo
            fusion_result   = fusion.fuse(face_result, voice_result)
            db.log_emotion(session_id, fusion_result)

            # Overlay
            emotion = fusion_result["dominant_emotion"]
            conf    = fusion_result["confidence"]
            cv2.putText(frame, f"{emotion} ({conf:.0%})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2)
            cv2.putText(frame, f"Session: {session_id}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Show probs
            y = 110
            for e, p in sorted(fusion_result["probabilities"].items(),
                                key=lambda x: -x[1]):
                bar = int(p * 150)
                cv2.rectangle(frame, (10, y), (10 + bar, y + 14), (0, 180, 100), -1)
                cv2.putText(frame, f"{e}: {p:.0%}",
                            (170, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y += 20

            cv2.imshow("🎵 Emotion Music Composer", frame)

            # Auto-generate music every N seconds
            now = time.time()
            if now - last_music_time > music_interval:
                logger.info("🎼 Generating music for emotion: %s", emotion)
                midi_path   = f"outputs/{session_id}_music.mid"
                result      = music_gen.generate(
                    fusion_result=fusion_result,
                    genre="Classical",
                    output_path=midi_path
                )
                audio_player.play_midi(result["midi"], session_id=session_id)
                db.log_music(session_id, result)
                last_music_time = now
                logger.info("✅ Playing music — %d tokens, tempo %d BPM",
                            result["num_tokens"], result["tempo"])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                last_music_time = 0   # Force regeneration
            elif key == ord('s'):
                midi_path = f"outputs/{session_id}_saved.mid"
                music_gen.generate(fusion_result=fusion_result,
                                   genre="Classical", output_path=midi_path)
                logger.info("💾 Saved MIDI to %s", midi_path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_player.stop()
        db.close()
        logger.info("Session %s ended. Files in outputs/", session_id)


def _run_headless_demo(music_gen, audio_player):
    """Demo without webcam — generates music for each emotion."""
    import time
    from configs.config import config

    logger.info("Running headless demo — generating music for all emotions")
    for emotion in config.EMOTION_CLASSES:
        dummy_result = {
            "dominant_emotion": emotion,
            "embedding": [0.1 * (i % 10) for i in range(128)],
            "confidence": 0.85
        }
        result = music_gen.generate(
            fusion_result=dummy_result,
            genre="Classical",
            output_path=f"outputs/demo_{emotion.lower()}.mid"
        )
        logger.info("✅ %s — %d tokens, %d BPM, saved to outputs/",
                    emotion, result["num_tokens"], result["tempo"])
        time.sleep(1)


# ── Server Mode ──────────────────────────────────────────────────────────────

def run_server():
    import uvicorn
    from configs.config import config
    logger.info("🚀 Starting FastAPI backend on %s:%d", config.API_HOST, config.API_PORT)
    uvicorn.run(
        "backend.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False
    )


# ── UI Mode ───────────────────────────────────────────────────────────────────

def run_ui():
    logger.info("🎨 Starting Streamlit dashboard")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])


# ── Full Mode ─────────────────────────────────────────────────────────────────

def run_full():
    import threading
    logger.info("🚀 Starting full stack: API + Frontend")

    api_thread = threading.Thread(target=run_server, daemon=True)
    api_thread.start()

    import time
    time.sleep(2)   # Wait for API to start
    run_ui()        # Blocking


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-Driven Music Composer")
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["demo", "server", "ui", "full"],
        help="Run mode: demo | server | ui | full"
    )
    args = parser.parse_args()

    modes = {
        "demo":   run_demo,
        "server": run_server,
        "ui":     run_ui,
        "full":   run_full
    }
    modes[args.mode]()
