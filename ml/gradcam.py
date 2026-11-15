"""Grad-CAM explainability for the ResNet18 forgery detector.

Shows WHICH image regions pushed the model towards "forged". Uses the plain weights (best.pt),
not the TorchScript file, because hooks are easy on a normal nn.Module.

Single face crop:
    python ml/gradcam.py --weights ml/out/best.pt --image face.png --out docs/result/gradcam/face.png

Whole video (same frame sampling + face crop as the backend, saves the top-k most suspicious frames):
    python ml/gradcam.py --weights ml/out/best.pt --video clip.mp4 --out docs/result/gradcam/clip --topk 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # must match training (ImageNet)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


def load_model(weights: str | Path) -> nn.Module:
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)  # same head as ml/train_baseline.py
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def gradcam(model: nn.Module, rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """rgb: uint8 (H, W, 3) RGB face crop. Returns (heatmap in 0..1 of shape SIZE x SIZE, forged probability)."""
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    x = torch.from_numpy(x)

    store: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = model.layer4.register_forward_hook(forward_hook)
    try:
        logit = model(x).reshape(-1)[0]   # raw logit for the "fake" class
        model.zero_grad()
        logit.backward()
    finally:
        handle.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store["act"]).sum(dim=1))[0].detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return cam, float(torch.sigmoid(logit))


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def side_by_side(rgb: np.ndarray, cam: np.ndarray, prob: float) -> np.ndarray:
    left = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    panel = np.concatenate([left, overlay(rgb, cam)], axis=1)
    cv2.putText(panel, f"forged p={prob:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return panel


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="an already cropped face image")
    src.add_argument("--video", type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output file (--image) or folder (--video)")
    ap.add_argument("--frames", type=int, default=16, help="frames sampled from the video")
    ap.add_argument("--topk", type=int, default=3, help="most suspicious frames to save")
    ap.add_argument("--no-face-crop", action="store_true", help="skip the Haar face crop for --video")
    args = ap.parse_args()

    model = load_model(args.weights)

    if args.image:
        rgb = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
        cam, prob = gradcam(model, rgb)
        _write_rgb(args.out, side_by_side(rgb, cam, prob))
        print(f"forged probability: {prob:.3f} -> {args.out}")
        return

    from app.services.frame_extractor import extract_uniform
    from app.services.preprocessing import crop_face

    results = []
    for i, frame in enumerate(extract_uniform(args.video, args.frames)):
        face = frame if args.no_face_crop else crop_face(frame)
        cam, prob = gradcam(model, face)
        results.append((prob, i, face, cam))
    if not results:
        raise SystemExit("No frames could be read from the video")
    results.sort(key=lambda r: r[0], reverse=True)
    for prob, i, face, cam in results[: args.topk]:
        path = args.out / f"frame{i:02d}_p{prob:.2f}.png"
        _write_rgb(path, side_by_side(face, cam, prob))
        print(f"frame {i}: forged probability {prob:.3f} -> {path}")


if __name__ == "__main__":
    main()
