
"""
Train a ResNet18 model for real/fake face-frame classification.

Run from the repository root:
    python ml/train_baseline.py --data "C:/path/to/Dataset" --epochs 8 --batch 16 --out ml/out

Expected dataset:
    Dataset/
        real/
        fake/

Outputs:
    best.pt
    forgery_resnet18.pt
    metrics.json

Backend preprocessing must match:
    frame size: 224
    normalization: ImageNet
    input: RGB face crops
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from split import scan_dataset, split_by_video, summarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    """Apply JPEG compression to simulate lower-quality frames."""

    def __call__(self, image):
        quality = random.randint(35, 95)
        buffer = io.BytesIO()

        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class FrameDataset(Dataset):
    """Dataset of face-crop images and binary labels."""

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label, _video_id = self.rows[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, torch.tensor(float(label), dtype=torch.float32)


def make_transforms(size):
    """Use augmentation only during training."""

    train_transform = T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),
        T.RandomApply(
            [RandomJPEGCompression()],
            p=0.5,
        ),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.15,
        ),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    # Deterministic validation and test preprocessing.
    eval_transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    return train_transform, eval_transform


def set_seed(seed):
    """Seed common random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    """Return sigmoid probabilities for the positive (label 1) class."""

    model.eval()
    probabilities = []

    for images, _labels in loader:
        images = images.to(device)
        logits = model(images).reshape(-1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    if not probabilities:
        return np.array([], dtype=np.float32)

    return np.concatenate(probabilities)


def metrics(y_true, probabilities, threshold=0.5):
    """Calculate classification metrics without inventing undefined AUC."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on an empty split.")

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths.")

    predictions = (probabilities >= threshold).astype(int)

    auc = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))

    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, predictions)), 4
        ),
        "precision": round(
            float(precision_score(y_true, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, predictions, zero_division=0)), 4
        ),
        "f1": round(
            float(f1_score(y_true, predictions, zero_division=0)), 4
        ),
        "auc": round(auc, 4) if auc is not None else None,
        "majority_class_accuracy": round(
            float(max(np.mean(y_true), 1.0 - np.mean(y_true))), 4
        ),
    }


def video_level(rows, probabilities):
    """Average frame probabilities so each video receives one prediction."""

    aggregated = defaultdict(list)
    video_labels = {}

    for (_path, label, video_id), probability in zip(rows, probabilities):
        aggregated[video_id].append(float(probability))

        if video_id in video_labels and video_labels[video_id] != label:
            raise ValueError(
                f"Inconsistent labels found for video {video_id!r}."
            )

        video_labels[video_id] = label

    videos = sorted(aggregated)

    labels = [int(video_labels[video_id]) for video_id in videos]
    scores = [
        float(np.mean(aggregated[video_id]))
        for video_id in videos
    ]

    return labels, scores


def choose_threshold(y_true, probabilities):
    """
    Select the threshold maximizing F1 on validation videos only.
    Ties are resolved using balanced accuracy.
    """

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Invalid validation labels or probabilities.")

    best_threshold = 0.5
    best_f1 = -1.0
    best_balanced_accuracy = -1.0

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        current_f1 = f1_score(
            y_true, predictions, zero_division=0
        )
        current_balanced_accuracy = balanced_accuracy_score(
            y_true, predictions
        )

        if (
            current_f1 > best_f1
            or (
                current_f1 == best_f1
                and current_balanced_accuracy > best_balanced_accuracy
            )
        ):
            best_f1 = float(current_f1)
            best_balanced_accuracy = float(current_balanced_accuracy)
            best_threshold = float(threshold)

    return round(best_threshold, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a ResNet18 video-forgery baseline."
    )

    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1 or args.size < 1:
        parser.error("--epochs, --batch, and --size must be positive.")

    set_seed(args.seed)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Dataset:", Path(args.data).resolve())

    # The existing splitter must group all frames from a video together.
    train_rows, val_rows, test_rows = split_by_video(
        scan_dataset(args.data),
        seed=args.seed,
    )

    for name, rows in (
        ("train", train_rows),
        ("validation", val_rows),
        ("test", test_rows),
    ):
        if not rows:
            raise ValueError(f"The {name} split is empty.")

        print(summarize(name, rows))

    train_transform, eval_transform = make_transforms(args.size)

    def make_loader(rows, transform, shuffle):
        return DataLoader(
            FrameDataset(rows, transform),
            batch_size=args.batch,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_rows, train_transform, True)
    val_loader = make_loader(val_rows, eval_transform, False)
    test_loader = make_loader(test_rows, eval_transform, False)

    # Binary classification: label 1 is assumed to mean fake.
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    n_fake = sum(int(label) for _, label, _ in train_rows)
    n_real = len(train_rows) - n_fake

    if n_fake == 0 or n_real == 0:
        raise ValueError(
            "Training split must contain both real (0) and fake (1) "
            "examples. Check your dataset labels."
        )

    pos_weight = torch.tensor(
        [n_real / n_fake],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_score = -float("inf")
    best_checkpoint = output_dir / "best.pt"
    history = []

    # Select the checkpoint using validation data, never test data.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images).reshape(-1)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)
            seen += len(images)

        val_probabilities = predict(model, val_loader, device)
        val_labels = [int(label) for _, label, _ in val_rows]

        validation_metrics = metrics(
            val_labels,
            val_probabilities,
        )

        train_loss = total_loss / max(seen, 1)

        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            **{
                f"val_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_auc={validation_metrics['auc']} | "
            f"val_balanced_accuracy="
            f"{validation_metrics['balanced_accuracy']} | "
            f"val_f1={validation_metrics['f1']}"
        )

        # Use AUC when defined; otherwise use balanced accuracy.
        score = validation_metrics["auc"]
        if score is None:
            score = validation_metrics["balanced_accuracy"]

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_checkpoint)

    # Reload the best validation-selected checkpoint.
    model.load_state_dict(
        torch.load(best_checkpoint, map_location=device)
    )

    # Select threshold using aggregated validation-video scores.
    val_probabilities = predict(model, val_loader, device)
    val_video_labels, val_video_probabilities = video_level(
        val_rows, val_probabilities
    )

    threshold = choose_threshold(
        val_video_labels,
        val_video_probabilities,
    )

    print("Selected validation-video threshold:", threshold)

    # Evaluate the test set exactly once after model/threshold selection.
    test_probabilities = predict(model, test_loader, device)

    test_labels = [int(label) for _, label, _ in test_rows]
    frame_metrics = metrics(
        test_labels,
        test_probabilities,
        threshold=threshold,
    )

    test_video_labels, test_video_probabilities = video_level(
        test_rows,
        test_probabilities,
    )
    video_metrics = metrics(
        test_video_labels,
        test_video_probabilities,
        threshold=threshold,
    )

    print("\nTEST frame-level metrics:", frame_metrics)
    print(
        "TEST video-level metrics:",
        video_metrics,
        f"({len(test_video_labels)} videos)",
    )

    metrics_data = {
        "model": "ResNet18",
        "seed": args.seed,
        "image_size": args.size,
        "threshold": threshold,
        "threshold_selection": "validation_video_f1",
        "positive_label": "fake",
        "test_frame": frame_metrics,
        "test_video": video_metrics,
        "n_test_videos": len(test_video_labels),
        "history": history,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics_data,
            file,
            indent=2,
            allow_nan=False,
        )

    # Export raw logits; the backend should apply sigmoid and the
    # same validation-selected threshold when making a prediction.
    model.eval().cpu()

    example_input = torch.randn(1, 3, args.size, args.size)
    traced_model = torch.jit.trace(model, example_input)

    torchscript_path = output_dir / "forgery_resnet18.pt"
    traced_model.save(str(torchscript_path))

    print("\nSaved best weights:", best_checkpoint)
    print("Saved metrics:", metrics_path)
    print("Saved TorchScript model:", torchscript_path)


if __name__ == "__main__":
    main()
