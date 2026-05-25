"""
Multimodal Fusion Module
Cross-attention transformer that fuses face + voice embeddings
into a single 128-dim emotion vector.
"""

import numpy as np
import torch
import torch.nn as nn
import logging

from configs.config import config

logger = logging.getLogger(__name__)


# ── Cross-Attention Fusion Transformer ──────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_modalities: int = 2):
        super().__init__()
        self.embed_dim  = embed_dim
        self.proj_face  = nn.Linear(128, embed_dim)
        self.proj_voice = nn.Linear(128, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                face_emb: torch.Tensor,
                voice_emb: torch.Tensor,
                face_weight: float = 0.6,
                voice_weight: float = 0.4) -> torch.Tensor:
        """
        face_emb  : (B, 128)
        voice_emb : (B, 128)
        Returns   : (B, 128) fused emotion embedding
        """
        f = self.proj_face(face_emb).unsqueeze(1)   # (B, 1, D)
        v = self.proj_voice(voice_emb).unsqueeze(1) # (B, 1, D)

        # Stack as sequence of tokens: [face, voice]
        tokens = torch.cat([f, v], dim=1)           # (B, 2, D)

        # Cross attention: face attends to voice and vice versa
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Weighted pool
        fused = face_weight * tokens[:, 0, :] + voice_weight * tokens[:, 1, :]
        return self.output_proj(fused)              # (B, 128)


class EmotionFusionModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_emotions: int = 8):
        super().__init__()
        self.fusion     = CrossAttentionFusion(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions)
        )

    def forward(self, face_emb, voice_emb, face_w=0.6, voice_w=0.4):
        fused  = self.fusion(face_emb, voice_emb, face_w, voice_w)
        logits = self.classifier(fused)
        return fused, logits


# ── Fusion Orchestrator ──────────────────────────────────────────────────────

class EmotionFusion:
    def __init__(self):
        self.device   = torch.device(config.DEVICE)
        self.emotions = config.EMOTION_CLASSES
        self.model    = self._load_model()
        self._prev_embedding = None
        logger.info("EmotionFusion initialized")

    def _load_model(self) -> EmotionFusionModel:
        model = EmotionFusionModel(
            embed_dim=config.EMOTION_EMBEDDING_DIM,
            num_emotions=config.NUM_EMOTIONS
        )
        if config.FUSION_MODEL_PATH and torch.os.path.exists(config.FUSION_MODEL_PATH):
            state = torch.load(config.FUSION_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state)
            logger.info("Loaded fusion model from %s", config.FUSION_MODEL_PATH)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def fuse(self,
             face_result: dict,
             voice_result: dict,
             interpolation_alpha: float = 0.85) -> dict:
        """
        Fuses face and voice results into unified emotion embedding.
        interpolation_alpha: smoothing factor for temporal transitions (0-1).
        """
        face_available  = face_result.get("face_detected", False)
        voice_available = voice_result.get("voice_detected", False)

        # Determine weights based on modality availability
        if face_available and voice_available:
            face_w, voice_w = 0.6, 0.4
        elif face_available:
            face_w, voice_w = 1.0, 0.0
        elif voice_available:
            face_w, voice_w = 0.0, 1.0
        else:
            return self._empty_result()

        face_emb  = torch.tensor(face_result["embedding"],  dtype=torch.float32).unsqueeze(0).to(self.device)
        voice_emb = torch.tensor(voice_result["embedding"], dtype=torch.float32).unsqueeze(0).to(self.device)

        fused_emb, logits = self.model(face_emb, voice_emb, face_w, voice_w)
        fused_np  = fused_emb.squeeze().cpu().numpy()
        probs     = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Smooth transitions between embeddings over time
        if self._prev_embedding is not None:
            fused_np = (interpolation_alpha * self._prev_embedding
                        + (1 - interpolation_alpha) * fused_np)
        self._prev_embedding = fused_np.copy()

        # Determine which modality was dominant
        if face_available and voice_available:
            dominant_source = "face+voice"
        elif face_available:
            dominant_source = "face"
        else:
            dominant_source = "voice"

        return {
            "embedding": fused_np.tolist(),
            "probabilities": {e: float(p) for e, p in zip(self.emotions, probs)},
            "dominant_emotion": self.emotions[int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
            "dominant_source": dominant_source,
            "face_weight": face_w,
            "voice_weight": voice_w
        }

    def reset_temporal(self):
        self._prev_embedding = None

    def _empty_result(self) -> dict:
        return {
            "embedding": [0.0] * config.EMOTION_EMBEDDING_DIM,
            "probabilities": {e: 1/8 for e in self.emotions},
            "dominant_emotion": "Neutral",
            "confidence": 0.0,
            "dominant_source": "none",
            "face_weight": 0.0,
            "voice_weight": 0.0
        }
