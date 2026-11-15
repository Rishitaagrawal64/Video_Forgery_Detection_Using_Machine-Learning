"""Build a training-frame dataset using the SAME frame sampling and face crop the backend uses.

Why: the model card notes training used pre-cropped faces while the API uses an OpenCV Haar cascade.
Training on frames produced by the serving code removes that train/serve mismatch.

    python ml/make_training_frames.py --videos /path/to/videos --out /path/to/Dataset_serving

Input : <videos>/real/*.mp4 and <videos>/fake/*.mp4
Output: <out>/real/<video>_<n>.png and <out>/fake/<video>_<n>.png   (the layout ml/split.py expects)
Then : python ml/train_baseline.py --data /path/to/Dataset_serving --out ml/out_serving
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.frame_extractor import extract_uniform  # noqa: E402
from app.services.preprocessing import crop_face, resize_frame  # noqa: E402

EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    for cls in ("real", "fake"):
        src = args.videos / cls
        if not src.is_dir():
            raise SystemExit(f"Missing folder: {src}")
        dst = args.out / cls
        dst.mkdir(parents=True, exist_ok=True)
        videos = [p for p in sorted(src.iterdir()) if p.suffix.lower() in EXTS]
        for n, video in enumerate(videos, 1):
            try:
                frames = extract_uniform(video, args.frames)
            except Exception as exc:  # noqa: BLE001 - skip unreadable videos but say so
                print(f"skip {video.name}: {exc}")
                continue
            for i, frame in enumerate(frames):
                face = resize_frame(crop_face(frame), args.size)
                # video stem must not end in _<digits> ambiguity: split.py strips only the last _<n>
                cv2.imwrite(str(dst / f"{video.stem}_{i}.png"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"[{cls}] {n}/{len(videos)} {video.name}: {len(frames)} frames")


if __name__ == "__main__":
    main()
