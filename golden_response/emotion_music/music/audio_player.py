"""
Audio Synthesis and Playback
Converts MIDI to audio using pygame and pretty_midi.
Handles real-time playback with smooth transitions.
"""

import os
import time
import logging
import threading
import numpy as np
import pretty_midi
import pygame

from configs.config import config

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self):
        self._init_pygame()
        self._playing       = False
        self._lock          = threading.Lock()
        self._current_path  = None
        logger.info("AudioPlayer initialized")

    def _init_pygame(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            logger.info("pygame mixer initialized")
        except Exception as e:
            logger.error("pygame mixer init failed: %s", e)

    def midi_to_audio(self, midi: pretty_midi.PrettyMIDI,
                      output_wav: str = "outputs/temp_audio.wav") -> str:
        """
        Synthesizes MIDI to WAV using pretty_midi's built-in FluidSynth wrapper.
        Falls back to a simple sine-wave synthesis if FluidSynth not available.
        """
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        try:
            audio = midi.fluidsynth(fs=44100)
            import scipy.io.wavfile as wav
            wav.write(output_wav, 44100, audio.astype(np.int16))
            logger.info("Synthesized audio saved to %s", output_wav)
            return output_wav
        except Exception as e:
            logger.warning("FluidSynth failed (%s) — using fallback synthesis", e)
            return self._fallback_synthesis(midi, output_wav)

    def _fallback_synthesis(self, midi: pretty_midi.PrettyMIDI,
                             output_wav: str) -> str:
        """Simple sine-wave synthesis fallback."""
        import scipy.io.wavfile as wav

        sr          = 44100
        duration    = max(n.end for inst in midi.instruments for n in inst.notes) if \
                      any(inst.notes for inst in midi.instruments) else 5.0
        audio       = np.zeros(int(sr * (duration + 1)), dtype=np.float32)

        for instrument in midi.instruments:
            for note in instrument.notes:
                freq    = pretty_midi.note_number_to_hz(note.pitch)
                start   = int(note.start * sr)
                end     = int(note.end * sr)
                t       = np.linspace(0, note.end - note.start, end - start)
                sine    = (note.velocity / 127.0) * 0.3 * np.sin(2 * np.pi * freq * t)
                # ADSR envelope
                env     = np.ones_like(sine)
                attack  = min(int(0.01 * sr), len(env))
                release = min(int(0.05 * sr), len(env))
                env[:attack]     = np.linspace(0, 1, attack)
                env[-release:]   = np.linspace(1, 0, release)
                audio[start:end] += (sine * env)

        audio = np.clip(audio, -1, 1)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(output_wav, sr, audio_int16)
        return output_wav

    def play(self, wav_path: str, fade_in_ms: int = 500):
        """Plays a WAV file with optional fade-in."""
        if not os.path.exists(wav_path):
            logger.error("Audio file not found: %s", wav_path)
            return

        with self._lock:
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.fadeout(300)
                    time.sleep(0.3)

                pygame.mixer.music.load(wav_path)
                pygame.mixer.music.play(fade_ms=fade_in_ms)
                self._playing       = True
                self._current_path  = wav_path
                logger.info("Playing: %s", wav_path)
            except Exception as e:
                logger.error("Playback failed: %s", e)

    def stop(self, fade_out_ms: int = 500):
        with self._lock:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(fade_out_ms)
            self._playing = False

    def pause(self):
        pygame.mixer.music.pause()

    def resume(self):
        pygame.mixer.music.unpause()

    def is_playing(self) -> bool:
        return pygame.mixer.music.get_busy()

    def set_volume(self, volume: float):
        """volume: 0.0 to 1.0"""
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))

    def play_midi(self, midi: pretty_midi.PrettyMIDI,
                  session_id: str = "session",
                  fade_in_ms: int = 500) -> str:
        """One-shot: synthesize and play."""
        wav_path = f"outputs/{session_id}_music.wav"
        wav_path = self.midi_to_audio(midi, wav_path)
        self.play(wav_path, fade_in_ms=fade_in_ms)
        return wav_path
