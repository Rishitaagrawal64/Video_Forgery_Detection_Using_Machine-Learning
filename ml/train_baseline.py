"""Baseline: fine-tune a pretrained ResNet18 on real/fake face frames, report honest metrics, export TorchScript.

Run (Colab / Kaggle / local):
    python train_baseline.py --data /path/to/Dataset --epochs 8 --out out

Outputs in --out:  best.pt (weights), forgery_resnet18.pt (TorchScript for the backend), metrics.json

Backend settings that must match this training (put in .env):
    FORGERY_PREDICTOR=torch
    FORGERY_MODEL_PATH=<path to forgery_resnet18.pt>
    FORGERY_FRAME_SIZE=224
    FORGERY_NORMALIZE_MODE=imagenet
    FORGERY_DETECT_FACES=true      # training images are face crops
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from split import scan_dataset, split_by_video, summarize

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class FrameDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label, _ = self.rows[i]
        return self.tf(Image.open(path).convert("RGB")), torch.tensor(float(label))


def make_transforms(size):
    train = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.15, 0.15, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    evalt = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evalt


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.sigmoid(model(x.to(device)).reshape(-1)).cpu().numpy())
    return np.concatenate(probs)


def metrics(y, p, thr=0.5):
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "majority_class_accuracy": max(y.mean(), 1 - y.mean()),
    }
    out["auc"] = roc_auc_score(y, p) if len(set(y.tolist())) == 2 else float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def video_level(rows, probs):
    """Average frame probabilities per video -> what the real system does."""
    agg, lab = defaultdict(list), {}
    for (_, l, v), p in zip(rows, probs):
        agg[v].append(p)
        lab[v] = l
    vids = sorted(agg)
    return [lab[v] for v in vids], [float(np.mean(agg[v])) for v in vids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train, val, test = split_by_video(scan_dataset(a.data), seed=a.seed)
    for n, r in (("train", train), ("val", val), ("test", test)):
        print(summarize(n, r))

    tf_train, tf_eval = make_transforms(a.size)
    mk = lambda rows, tf, sh: DataLoader(FrameDataset(rows, tf), batch_size=a.batch, shuffle=sh, num_workers=a.workers)
    dl_train, dl_val, dl_test = mk(train, tf_train, True), mk(val, tf_eval, False), mk(test, tf_eval, False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    n_fake = sum(l for _, l, _ in train)
    pos_weight = torch.tensor((len(train) - n_fake) / max(n_fake, 1), device=device)  # offsets class imbalance
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    best_auc, history = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).reshape(-1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        pv = predict(model, dl_val, device)
        m = metrics([l for _, l, _ in val], pv)
        history.append({"epoch": ep, "train_loss": round(total / len(train), 4), **{f"val_{k}": v for k, v in m.items()}})
        print(f"epoch {ep}: loss {total / len(train):.4f} | val AUC {m['auc']} bal.acc {m['balanced_accuracy']} F1 {m['f1']}")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    pt = predict(model, dl_test, device)
    frame_m = metrics([l for _, l, _ in test], pt)
    vy, vp = video_level(test, pt)
    video_m = metrics(vy, vp)
    print("\nTEST frame-level:", frame_m)
    print("TEST video-level:", video_m, f"({len(vy)} videos)")

    json.dump({"test_frame": frame_m, "test_video": video_m, "n_test_videos": len(vy),
               "history": history}, open(out / "metrics.json", "w"), indent=2)

    model.eval().cpu()
    ts = torch.jit.trace(model, torch.randn(1, 3, a.size, a.size))
    ts.save(str(out / "forgery_resnet18.pt"))
    print("saved TorchScript ->", out / "forgery_resnet18.pt")


if __name__ == "__main__":
    main()
