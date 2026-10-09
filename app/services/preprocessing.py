"""Preprocessing pipeline (Sprint 1): resize, optional face crop, normalize, batch."""
from __future__ import annotations

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def crop_face(frame: np.ndarray, margin: float = 0.25) -> np.ndarray:
    """Crop to the largest detected face (with margin). Returns the frame unchanged if none found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return frame
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, frame.shape[1]), min(y + h + my, frame.shape[0])
    return frame[y0:y1, x0:x1]


def resize_frame(frame: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def normalize_frame(frame: np.ndarray, mode: str = "unit") -> np.ndarray:
    """uint8 RGB -> float32. Modes: 'unit' [0,1], 'tanh' [-1,1], 'imagenet' (mean/std)."""
    x = frame.astype(np.float32) / 255.0
    if mode == "unit":
        return x
    if mode == "tanh":
        return x * 2.0 - 1.0
    if mode == "imagenet":
        return (x - _IMAGENET_MEAN) / _IMAGENET_STD
    raise ValueError(f"Unknown normalize mode: {mode!r}")


def preprocess_frames(
    frames: list[np.ndarray],
    size: int = 224,
    mode: str = "unit",
    detect_faces: bool = False,
) -> np.ndarray:
    """Return a float32 batch of shape (N, size, size, 3), channels-last (Keras/TensorFlow layout).

    For PyTorch use `batch.transpose(0, 3, 1, 2)`.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    out = []
    for f in frames:
        if detect_faces:
            f = crop_face(f)
        out.append(normalize_frame(resize_frame(f, size), mode))
    return np.stack(out, axis=0)
