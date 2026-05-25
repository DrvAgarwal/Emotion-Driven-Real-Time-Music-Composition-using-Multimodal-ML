"""
Training Script — Face Emotion Model
Trains EfficientNet-B0 on FER-2013 or AffectNet dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import logging
import mlflow
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

from models.face_emotion import FaceEmotionModel, get_transforms
from configs.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class FERDataset(Dataset):
    """
    Expects a CSV with columns: 'image_path', 'label' (0-7)
    Compatible with FER-2013 and AffectNet formats.
    """
    def __init__(self, csv_path: str, transform=None):
        self.df         = pd.read_csv(csv_path)
        self.transform  = transform
        self.labels     = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        image   = Image.open(row["image_path"]).convert("RGB")
        label   = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_weighted_sampler(dataset: FERDataset) -> WeightedRandomSampler:
    """Handles class imbalance with weighted sampling."""
    counts  = Counter(dataset.labels)
    weights = [1.0 / counts[l] for l in dataset.labels]
    return WeightedRandomSampler(weights, len(weights))


# ── Training ──────────────────────────────────────────────────────────────────

def train_face_model(
    train_csv:      str,
    val_csv:        str,
    output_path:    str = "models/face_emotion.pth",
    epochs:         int = 30,
    batch_size:     int = 32,
    lr:             float = 1e-4
):
    device      = torch.device(config.DEVICE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Datasets
    train_ds    = FERDataset(train_csv, transform=get_transforms(train=True))
    val_ds      = FERDataset(val_csv,   transform=get_transforms(train=False))
    sampler     = get_weighted_sampler(train_ds)

    train_loader= DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # Model
    model       = FaceEmotionModel(num_classes=config.NUM_EMOTIONS).to(device)

    # Class-weighted loss
    counts      = Counter(train_ds.labels)
    weights     = torch.tensor(
        [1.0 / counts.get(i, 1) for i in range(config.NUM_EMOTIONS)],
        dtype=torch.float32
    ).to(device)
    criterion   = nn.CrossEntropyLoss(weight=weights)

    optimizer   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1     = 0.0
    patience    = 5
    no_improve  = 0

    mlflow.set_experiment("face_emotion_training")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "device": config.DEVICE
        })

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss  = 0.0
            all_preds   = []
            all_labels  = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss  += loss.item()
                preds        = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

            train_f1    = f1_score(all_labels, all_preds, average="macro")

            # Validate
            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_f1  = f1_score(val_labels, val_preds, average="macro")
            val_acc = accuracy_score(val_labels, val_preds)

            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %02d | Train Loss: %.4f | Train F1: %.4f | Val F1: %.4f | Val Acc: %.4f | LR: %.6f",
                epoch + 1, train_loss / len(train_loader), train_f1, val_f1, val_acc, lr_now
            )

            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "train_f1":   train_f1,
                "val_f1":     val_f1,
                "val_acc":    val_acc,
                "lr":         lr_now
            }, step=epoch)

            # Save best
            if val_f1 > best_f1:
                best_f1     = val_f1
                no_improve  = 0
                torch.save(model.state_dict(), output_path)
                logger.info("✅ Saved best model (Val F1: %.4f)", best_f1)
                mlflow.log_artifact(output_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        mlflow.log_metric("best_val_f1", best_f1)
        logger.info("Training complete. Best Val F1: %.4f", best_f1)

    return best_f1


if __name__ == "__main__":
    train_face_model(
        train_csv="data/fer_train.csv",
        val_csv="data/fer_val.csv",
        output_path="models/face_emotion.pth",
        epochs=30,
        batch_size=32,
        lr=1e-4
    )
