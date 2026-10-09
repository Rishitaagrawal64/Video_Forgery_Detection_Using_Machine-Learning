"""CLI: extract + preprocess frames from a video, optionally saving them and running the predictor.

Usage:
    python scripts/extract_frames.py path/to/video.mp4 --num-frames 16 --out data/frames/demo
    python scripts/extract_frames.py video.mp4 --predict
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.registry import get_predictor  # noqa: E402
from app.services import frame_extractor as fx  # noqa: E402
from app.services.pipeline import run_pipeline  # noqa: E402
from app.services.preprocessing import preprocess_frames  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--mode", choices=["unit", "tanh", "imagenet"], default="unit")
    p.add_argument("--faces", action="store_true", help="crop to the largest detected face")
    p.add_argument("--out", help="directory to save extracted frames as JPEG")
    p.add_argument("--predict", action="store_true", help="run the configured predictor")
    a = p.parse_args()

    info = fx.get_video_info(a.video)
    print(f"{info.width}x{info.height} @ {info.fps:.2f} fps, {info.frame_count} frames, {info.duration_sec:.1f}s")

    if a.predict:
        res = run_pipeline(a.video, get_predictor(), num_frames=a.num_frames, size=a.size,
                           mode=a.mode, detect_faces=a.faces, save=bool(a.out), frames_out=a.out)
        print(f"{res.prediction.label}  p(forged)={res.prediction.forged_probability:.3f}  frames={res.num_frames}")
        return

    frames = fx.extract_uniform(a.video, a.num_frames)
    batch = preprocess_frames(frames, a.size, a.mode, a.faces)
    print(f"batch shape={batch.shape} dtype={batch.dtype} range=[{batch.min():.2f}, {batch.max():.2f}]")
    if a.out:
        paths = fx.save_frames(frames, a.out)
        print(f"saved {len(paths)} frames to {a.out}")


if __name__ == "__main__":
    main()
