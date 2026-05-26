"""
Training Script — Music Transformer
Fine-tunes Music Transformer on MIDI data conditioned on emotion labels.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import numpy as np
import logging
import mlflow
from pathlib import Path

from music.music_transformer import (MusicTransformer, FULL_VOCAB, BOS_TOKEN,
                                      EOS_TOKEN, PAD_TOKEN, GENRE_MAP,
                                      NOTE_ON_OFFSET, NOTE_OFF_OFFSET,
                                      TIME_SHIFT_OFFSET, VELOCITY_OFFSET)
from configs.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── MIDI Tokenizer ────────────────────────────────────────────────────────────

def midi_to_tokens(midi_path: str, max_tokens: int = 512) -> list:
    """Converts a MIDI file to a list of integer tokens."""
    try:
        midi        = pretty_midi.PrettyMIDI(midi_path)
        tokens      = []
        events      = []

        for instrument in midi.instruments:
            for note in instrument.notes:
                events.append((note.start,  "note_on",  note.pitch, note.velocity))
                events.append((note.end,    "note_off", note.pitch, 0))

        events.sort(key=lambda x: x[0])

        current_time = 0.0
        for time_, etype, pitch, vel in events:
            # Time shift
            delta = time_ - current_time
            if delta > 0:
                steps = min(int(delta / 0.01), 99)
                tokens.append(TIME_SHIFT_OFFSET + steps)
            current_time = time_

            if etype == "note_on":
                vel_idx = min(int(vel / 4), 31)
                tokens.append(VELOCITY_OFFSET + vel_idx)
                tokens.append(NOTE_ON_OFFSET + pitch)
            else:
                tokens.append(NOTE_OFF_OFFSET + pitch)

        return tokens[:max_tokens]
    except Exception as e:
        logger.warning("Failed to tokenize %s: %s", midi_path, e)
        return []


# ── Dataset ──────────────────────────────────────────────────────────────────

class MIDIEmotionDataset(Dataset):
    """
    Expects a JSON file with entries:
    [{"midi_path": "...", "emotion": "Happy", "genre": "Classical"}, ...]
    """
    EMOTION_MAP = {
        "Neutral": 0, "Happy": 1, "Sad": 2, "Angry": 3,
        "Fearful": 4, "Disgusted": 5, "Surprised": 6, "Calm": 7
    }

    def __init__(self, json_path: str, max_seq: int = 512):
        with open(json_path) as f:
            self.data = json.load(f)
        self.max_seq = max_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry   = self.data[idx]
        tokens  = midi_to_tokens(entry["midi_path"], self.max_seq)
        if not tokens:
            tokens = [BOS_TOKEN, EOS_TOKEN]

        # Build input/target pairs (teacher forcing)
        seq     = [BOS_TOKEN] + tokens + [EOS_TOKEN]
        seq     = seq[:self.max_seq + 1]
        input_  = seq[:-1]
        target  = seq[1:]

        # Pad
        pad_len = self.max_seq - len(input_)
        input_  = input_  + [PAD_TOKEN] * pad_len
        target  = target  + [PAD_TOKEN] * pad_len

        emotion_id = self.EMOTION_MAP.get(entry.get("emotion", "Neutral"), 0)
        genre_id   = GENRE_MAP.get(entry.get("genre", "Classical"), 0)

        return {
            "input_ids":    torch.tensor(input_,     dtype=torch.long),
            "labels":       torch.tensor(target,     dtype=torch.long),
            "emotion_id":   torch.tensor(emotion_id, dtype=torch.long),
            "genre_id":     torch.tensor(genre_id,   dtype=torch.long)
        }


# ── Dummy Emotion Embeddings from IDs ─────────────────────────────────────────

class EmotionEmbedder(nn.Module):
    def __init__(self, num_emotions: int = 8, embed_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_emotions, embed_dim)

    def forward(self, emotion_ids):
        return self.emb(emotion_ids)


# ── Training ──────────────────────────────────────────────────────────────────

def train_music_transformer(
    train_json:     str,
    val_json:       str,
    output_path:    str = "models/music_transformer.pth",
    epochs:         int = 50,
    batch_size:     int = 16,
    lr:             float = 3e-4
):
    device      = torch.device(config.DEVICE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train_ds    = MIDIEmotionDataset(train_json)
    val_ds      = MIDIEmotionDataset(val_json)
    train_loader= DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model       = MusicTransformer(
        vocab_size=FULL_VOCAB, embed_dim=256, num_heads=8,
        num_layers=6, max_seq_len=512, emotion_dim=128
    ).to(device)

    embedder    = EmotionEmbedder(num_emotions=8, embed_dim=128).to(device)

    optimizer   = optim.AdamW(
        list(model.parameters()) + list(embedder.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss   = float("inf")
    patience        = 7
    no_improve      = 0

    mlflow.set_experiment("music_transformer_training")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "device": config.DEVICE
        })

        for epoch in range(epochs):
            # Train
            model.train()
            embedder.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids   = batch["input_ids"].to(device)
                labels      = batch["labels"].to(device)
                emotion_ids = batch["emotion_id"].to(device)
                genre_ids   = batch["genre_id"].to(device)

                emotion_emb = embedder(emotion_ids)
                optimizer.zero_grad()
                logits  = model(input_ids, emotion_emb, genre_ids)
                B, T, V = logits.shape
                loss    = criterion(logits.reshape(B * T, V), labels.reshape(B * T))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            embedder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids   = batch["input_ids"].to(device)
                    labels      = batch["labels"].to(device)
                    emotion_ids = batch["emotion_id"].to(device)
                    genre_ids   = batch["genre_id"].to(device)
                    emotion_emb = embedder(emotion_ids)
                    logits      = model(input_ids, emotion_emb, genre_ids)
                    B, T, V     = logits.shape
                    val_loss   += criterion(logits.reshape(B*T, V),
                                            labels.reshape(B*T)).item()

            avg_train   = train_loss / len(train_loader)
            avg_val     = val_loss   / len(val_loader)
            scheduler.step()

            logger.info("Epoch %02d | Train Loss: %.4f | Val Loss: %.4f",
                        epoch + 1, avg_train, avg_val)
            mlflow.log_metrics({"train_loss": avg_train, "val_loss": avg_val}, step=epoch)

            if avg_val < best_val_loss:
                best_val_loss   = avg_val
                no_improve      = 0
                torch.save(model.state_dict(), output_path)
                logger.info("✅ Saved best music model (Val Loss: %.4f)", best_val_loss)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)

    return best_val_loss


if __name__ == "__main__":
    train_music_transformer(
        train_json="data/midi_emotion_train.json",
        val_json="data/midi_emotion_val.json",
        output_path="models/music_transformer.pth",
        epochs=50,
        batch_size=16,
        lr=3e-4
    )
