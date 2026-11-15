"""Compare frame->video aggregation methods (mean / median / topk) with the DEPLOYED pipeline.

Threshold is chosen on the validation videos only, then applied once to the test videos.

    FORGERY_PREDICTOR=torch FORGERY_MODEL_PATH=ml/out/forgery_resnet18.pt \
    FORGERY_NORMALIZE_MODE=imagenet FORGERY_DETECT_FACES=true \
    python scripts/aggregation_study.py --val videos_val --test videos_test --out docs/result/aggregation_study.json

Each folder needs real/*.mp4 and fake/*.mp4. Use videos that were NOT used for training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.models.base import aggregate_scores  # noqa: E402
from app.models.registry import get_predictor  # noqa: E402
from app.services.evaluation import best_f1_threshold, compute_metrics  # noqa: E402
from app.services.pipeline import run_pipeline  # noqa: E402

METHODS = ("mean", "median", "topk")


def frame_scores(root: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    predictor = get_predictor()
    labels, scores = [], []
    for name, label in (("real", 0), ("fake", 1)):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in settings.allowed_extensions:
                continue
            try:
                res = run_pipeline(p, predictor)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {p.name}: {exc}")
                continue
            labels.append(label)
            scores.append(np.asarray(res.prediction.frame_scores, dtype=np.float32))
            print(f"{name}/{p.name}: mean frame score {scores[-1].mean():.3f}")
    if not labels:
        raise SystemExit(f"No videos scored under {root}")
    return np.asarray(labels), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--topk-fraction", type=float, default=settings.topk_fraction)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    y_val, f_val = frame_scores(args.val)
    y_test, f_test = frame_scores(args.test)

    report = {}
    for method in METHODS:
        val_p = [aggregate_scores(s, method, args.topk_fraction) for s in f_val]
        test_p = [aggregate_scores(s, method, args.topk_fraction) for s in f_test]
        best = best_f1_threshold(y_val, val_p)
        report[method] = {
            "threshold_from_val": best["threshold"],
            "val_f1_at_threshold": best["f1"],
            "test": compute_metrics(y_test, test_p, best["threshold"]),
        }

    print(f"\n{'method':8} {'thr':>6} {'acc':>6} {'bal_acc':>8} {'f1':>6} {'auc':>6}   (test, n={len(y_test)})")
    for m, r in report.items():
        t = r["test"]
        thr = r["threshold_from_val"]
        print(f"{m:8} {thr:>6} {t['accuracy']:>6} {t['balanced_accuracy']:>8} {t['f1']:>6} {t['auc']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"n_val": int(len(y_val)), "n_test": int(len(y_test)), **report}, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
