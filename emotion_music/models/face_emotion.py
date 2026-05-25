"""
Face Emotion Detection Module
Uses EfficientNet-B0 fine-tuned for 8-class emotion recognition.
Falls back to a lightweight CNN if pretrained weights not found.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import mediapipe as mp
import logging

from configs.config import config

logger = logging.getLogger(__name__)


# ── Model Definition ────────────────────────────────────────────────────────

class FaceEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_embedding(self, x):
        """Returns 128-dim embedding before classification head."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features[:, :128]


# ── Transforms ──────────────────────────────────────────────────────────────

def get_transforms(train: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ])


# ── Face Detector ────────────────────────────────────────────────────────────

class FaceEmotionDetector:
    def __init__(self):
        self.device     = torch.device(config.DEVICE)
        self.model      = self._load_model()
        self.transform  = get_transforms(train=False)
        self.mp_face    = mp.solutions.face_detection
        self.detector   = self.mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.emotions   = config.EMOTION_CLASSES
        logger.info("FaceEmotionDetector initialized on %s", config.DEVICE)

    def _load_model(self) -> FaceEmotionModel:
        model = FaceEmotionModel(num_classes=config.NUM_EMOTIONS, pretrained=True)
        if config.FACE_MODEL_PATH and torch.os.path.exists(config.FACE_MODEL_PATH):
            state = torch.load(config.FACE_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state)
            logger.info("Loaded face model from %s", config.FACE_MODEL_PATH)
        else:
            logger.warning("No pretrained face weights found — using ImageNet init.")
        model.to(self.device)
        model.eval()
        return model

    def detect_faces(self, frame: np.ndarray) -> list:
        """Returns list of (x, y, w, h) face bounding boxes."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        boxes = []
        if results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x  = max(0, int(bb.xmin * w))
                y  = max(0, int(bb.ymin * h))
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                boxes.append((x, y, bw, bh))
        return boxes

    def preprocess_face(self, frame: np.ndarray, box: tuple) -> torch.Tensor:
        x, y, w, h = box
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            return None
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        return self.transform(face_pil).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> dict:
        """
        Main inference call.
        Returns dict with emotion probabilities and embedding.
        """
        boxes = self.detect_faces(frame)
        if not boxes:
            return self._empty_result()

        # Use the largest face
        box = max(boxes, key=lambda b: b[2] * b[3])
        tensor = self.preprocess_face(frame, box)
        if tensor is None:
            return self._empty_result()

        logits    = self.model(tensor)
        probs     = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        embedding = self.model.get_embedding(tensor).squeeze().cpu().numpy()

        return {
            "probabilities": {e: float(p) for e, p in zip(self.emotions, probs)},
            "dominant_emotion": self.emotions[int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
            "embedding": embedding.tolist(),
            "face_detected": True
        }

    def _empty_result(self) -> dict:
        return {
            "probabilities": {e: 1/8 for e in self.emotions},
            "dominant_emotion": "Neutral",
            "confidence": 0.0,
            "embedding": [0.0] * 128,
            "face_detected": False
        }

    def draw_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draws emotion label on frame."""
        if result["face_detected"]:
            label = f"{result['dominant_emotion']} ({result['confidence']:.0%})"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame
