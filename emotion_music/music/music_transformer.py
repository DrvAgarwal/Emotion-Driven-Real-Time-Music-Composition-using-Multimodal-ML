"""
Music Transformer
GPT-style transformer conditioned on emotion embeddings.
Generates MIDI token sequences representing original music.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi
import logging
import os

from configs.config import config

logger = logging.getLogger(__name__)

# ── MIDI Vocabulary ──────────────────────────────────────────────────────────
# Token space: 128 note-on + 128 note-off + 100 time-shift + 32 velocity = 388
NOTE_ON_OFFSET      = 0
NOTE_OFF_OFFSET     = 128
TIME_SHIFT_OFFSET   = 256
VELOCITY_OFFSET     = 356
VOCAB_SIZE          = 388
PAD_TOKEN           = VOCAB_SIZE        # 388
BOS_TOKEN           = VOCAB_SIZE + 1   # 389
EOS_TOKEN           = VOCAB_SIZE + 2   # 390
FULL_VOCAB          = VOCAB_SIZE + 3   # 391

# Genre conditioning tokens mapped to prefix embeddings
GENRE_MAP = {
    "Classical": 0,
    "Ambient":   1,
    "Jazz":      2,
    "Lo-Fi":     3,
    "Cinematic": 4
}

# Emotion → music parameter guidance
EMOTION_MUSIC_PARAMS = {
    "Happy":     {"tempo": 140, "velocity_scale": 0.85, "key": "major"},
    "Sad":       {"tempo": 65,  "velocity_scale": 0.45, "key": "minor"},
    "Angry":     {"tempo": 175, "velocity_scale": 1.0,  "key": "diminished"},
    "Calm":      {"tempo": 72,  "velocity_scale": 0.5,  "key": "major"},
    "Fearful":   {"tempo": 110, "velocity_scale": 0.6,  "key": "minor"},
    "Disgusted": {"tempo": 90,  "velocity_scale": 0.7,  "key": "minor"},
    "Surprised": {"tempo": 155, "velocity_scale": 0.9,  "key": "major"},
    "Neutral":   {"tempo": 100, "velocity_scale": 0.65, "key": "major"},
}


# ── Music Transformer Model ──────────────────────────────────────────────────

class MusicTransformer(nn.Module):
    def __init__(self,
                 vocab_size:    int = FULL_VOCAB,
                 embed_dim:     int = 256,
                 num_heads:     int = 8,
                 num_layers:    int = 6,
                 max_seq_len:   int = 512,
                 emotion_dim:   int = 128,
                 num_genres:    int = 5,
                 dropout:       float = 0.1):
        super().__init__()
        self.embed_dim  = embed_dim
        self.max_seq    = max_seq_len

        # Token + positional embeddings
        self.token_emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_emb    = nn.Embedding(max_seq_len, embed_dim)

        # Condition on emotion embedding
        self.emotion_proj = nn.Sequential(
            nn.Linear(emotion_dim, embed_dim),
            nn.GELU()
        )

        # Genre embedding
        self.genre_emb  = nn.Embedding(num_genres, embed_dim)

        # Transformer decoder layers
        decoder_layer   = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)
        self.dropout     = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                token_ids:      torch.Tensor,
                emotion_emb:    torch.Tensor,
                genre_id:       torch.Tensor = None) -> torch.Tensor:
        B, T = token_ids.shape
        pos  = torch.arange(T, device=token_ids.device).unsqueeze(0)

        # Token + position
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Build memory from emotion + genre conditioning
        emotion_ctx = self.emotion_proj(emotion_emb).unsqueeze(1)  # (B, 1, D)
        if genre_id is not None:
            genre_ctx   = self.genre_emb(genre_id).unsqueeze(1)    # (B, 1, D)
            memory      = torch.cat([emotion_ctx, genre_ctx], dim=1) # (B, 2, D)
        else:
            memory = emotion_ctx

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        out = self.transformer(x, memory, tgt_mask=causal_mask)
        out = self.norm(out)
        return self.output_head(out)


# ── Music Generator ──────────────────────────────────────────────────────────

class MusicGenerator:
    def __init__(self):
        self.device  = torch.device(config.DEVICE)
        self.model   = self._load_model()
        logger.info("MusicGenerator initialized")

    def _load_model(self) -> MusicTransformer:
        model = MusicTransformer(
            vocab_size=FULL_VOCAB,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            max_seq_len=config.MAX_MIDI_TOKENS,
            emotion_dim=config.EMOTION_EMBEDDING_DIM
        )
        if config.MUSIC_MODEL_PATH and os.path.exists(config.MUSIC_MODEL_PATH):
            state = torch.load(config.MUSIC_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state)
            logger.info("Loaded music model from %s", config.MUSIC_MODEL_PATH)
        else:
            logger.warning("No music model weights found — using random init (demo mode).")
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def generate_tokens(self,
                        emotion_emb:    list,
                        genre:          str = "Classical",
                        max_tokens:     int = None,
                        temperature:    float = None) -> list:
        """Autoregressively generates MIDI token sequence."""
        max_tokens  = max_tokens  or config.MAX_MIDI_TOKENS
        temperature = temperature or config.MUSIC_TEMPERATURE

        emb     = torch.tensor(emotion_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        genre_t = torch.tensor([GENRE_MAP.get(genre, 0)], dtype=torch.long).to(self.device)

        tokens  = [BOS_TOKEN]
        ids     = torch.tensor([tokens], dtype=torch.long).to(self.device)

        for _ in range(max_tokens):
            logits  = self.model(ids, emb, genre_t)
            next_logits = logits[:, -1, :] / temperature

            # Nucleus sampling (top-p = 0.9)
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove    = cum_probs > 0.9
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0]  = False
            next_logits[0][sorted_idx[0][remove[0]]] = float('-inf')

            probs   = F.softmax(next_logits, dim=-1)
            next_t  = torch.multinomial(probs, 1)
            token   = next_t.item()

            if token == EOS_TOKEN:
                break

            tokens.append(token)
            ids = torch.cat([ids, next_t], dim=1)

            if ids.shape[1] >= config.MAX_MIDI_TOKENS:
                break

        return tokens[1:]  # Strip BOS

    def tokens_to_midi(self,
                       tokens:          list,
                       emotion:         str = "Neutral",
                       output_path:     str = None) -> pretty_midi.PrettyMIDI:
        """Converts token sequence to pretty_midi object."""
        params      = EMOTION_MUSIC_PARAMS.get(emotion, EMOTION_MUSIC_PARAMS["Neutral"])
        midi        = pretty_midi.PrettyMIDI(initial_tempo=params["tempo"])
        instrument  = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time    = 0.0
        current_vel     = int(80 * params["velocity_scale"])
        active_notes    = {}
        time_per_shift  = 0.01  # 10ms per time-shift token

        for token in tokens:
            if token < NOTE_ON_OFFSET + 128:  # Note ON
                pitch = token - NOTE_ON_OFFSET
                active_notes[pitch] = (current_time, current_vel)

            elif token < NOTE_OFF_OFFSET + 128:  # Note OFF
                pitch = token - NOTE_OFF_OFFSET
                if pitch in active_notes:
                    start, vel = active_notes.pop(pitch)
                    duration   = max(0.05, current_time - start)
                    note = pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=start,
                        end=start + duration
                    )
                    instrument.notes.append(note)

            elif token < TIME_SHIFT_OFFSET + 100:  # Time shift
                steps        = token - TIME_SHIFT_OFFSET + 1
                current_time += steps * time_per_shift

            elif token < VELOCITY_OFFSET + 32:   # Velocity change
                vel_idx      = token - VELOCITY_OFFSET
                current_vel  = int((vel_idx / 31) * 127 * params["velocity_scale"])
                current_vel  = max(1, min(127, current_vel))

        # Close any still-open notes
        for pitch, (start, vel) in active_notes.items():
            instrument.notes.append(pretty_midi.Note(
                velocity=vel, pitch=pitch,
                start=start, end=max(start + 0.1, current_time)
            ))

        instrument.notes.sort(key=lambda n: n.start)
        midi.instruments.append(instrument)

        if output_path:
            midi.write(output_path)
            logger.info("MIDI saved to %s", output_path)

        return midi

    def generate(self,
                 fusion_result:  dict,
                 genre:          str = "Classical",
                 output_path:    str = None) -> dict:
        """
        Full generation pipeline.
        fusion_result: output from EmotionFusion.fuse()
        Returns: dict with midi object, token list, emotion params
        """
        emotion     = fusion_result.get("dominant_emotion", "Neutral")
        embedding   = fusion_result.get("embedding", [0.0] * 128)

        tokens  = self.generate_tokens(embedding, genre=genre)
        midi    = self.tokens_to_midi(tokens, emotion=emotion, output_path=output_path)

        return {
            "midi":         midi,
            "tokens":       tokens,
            "num_tokens":   len(tokens),
            "emotion":      emotion,
            "genre":        genre,
            "tempo":        EMOTION_MUSIC_PARAMS[emotion]["tempo"],
            "output_path":  output_path
        }
