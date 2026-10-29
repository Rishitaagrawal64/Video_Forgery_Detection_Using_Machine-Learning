"""Evaluate the *deployed* inference pipeline (frame sampling -> preprocessing -> model -> aggregation)
on a folder of videos, so training metrics can be checked against what the API actually returns.

    <root>/real/*.mp4      label 0
    <root>/fake/*.mp4      label 1

Run from the repo root with the same FORGERY_* settings you use for the API:

    FORGERY_PREDICTOR=torch FORGERY_MODEL_PATH=ml/out/forgery_resnet18.pt \
    FORGERY_NORMALIZE_MODE=imagenet FORGERY_DETECT_FACES=true FORGERY_DECISION_THRESHOLD=0.42 \
    python scripts/evaluate_pipeline.py --data /path/to/videos --out docs/result/pipeline_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.models.registry import get_predictor  # noqa: E402
from app.services.evaluation import compute_metrics  # noqa: E402
from app.services.pipeline import run_pipeline  # noqa: E402

VIDEO_EXT = set(settings.allowed_extensions)


def collect(root: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for name, label in (("real", 0), ("fake", 1)):
        d = root / name
        if not d.is_dir():
            raise SystemExit(f"Missing folder: {d}")
        items += [(p, label) for p in sorted(d.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not items:
        raise SystemExit(f"No videos found under {root}/real and {root}/fake")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write the report here as JSON")
    ap.add_argument("--limit", type=int, default=None, help="max videos per class (quick smoke test)")
    args = ap.parse_args()

    predictor = get_predictor()
    items = collect(args.data)
    if args.limit:
        items = [it for label in (0, 1) for it in [i for i in items if i[1] == label][: args.limit]]

    y_true, probs, failures = [], [], []
    t0 = time.perf_counter()
    for i, (path, label) in enumerate(items, 1):
        try:
            res = run_pipeline(path, predictor)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failures.append({"video": str(path), "error": str(exc)})
            continue
        y_true.append(label)
        probs.append(res.prediction.forged_probability)
        print(f"[{i}/{len(items)}] {path.name}: p_forged={probs[-1]:.3f} (label={label})")

    if not y_true:
        raise SystemExit("No video could be processed")
    report = {
        "predictor": predictor.name,
        "settings": {
            "num_frames": settings.num_frames, "frame_size": settings.frame_size,
            "normalize_mode": settings.normalize_mode, "detect_faces": settings.detect_faces,
            "aggregation": settings.aggregation, "decision_threshold": settings.decision_threshold,
        },
        "metrics": compute_metrics(y_true, probs, settings.decision_threshold),
        "failed_videos": failures,
        "seconds_per_video": round((time.perf_counter() - t0) / len(items), 3),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
