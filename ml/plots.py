"""Training curves and result summary figures from ml/out/metrics.json (no GPU / dataset needed).

    python ml/plots.py --metrics ml/out/metrics.json --out docs/result
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("ml/out/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/result"))
    args = ap.parse_args()

    data = json.loads(args.metrics.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    hist = data["history"]
    epochs = [h["epoch"] for h in hist]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in hist], marker="o")
    ax[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    for key, label in (("val_auc", "AUC"), ("val_f1", "F1"), ("val_balanced_accuracy", "Balanced acc.")):
        if key in hist[0]:
            ax[1].plot(epochs, [h[key] for h in hist], marker="o", label=label)
    ax[1].set(title="Validation metrics (frame level)", xlabel="epoch", ylim=(0.5, 1.0))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(args.out / "training_curves.png", dpi=160)
    plt.close(fig)

    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    xs = range(len(keys))
    ax.bar([x - width / 2 for x in xs], [data["test_frame"][k] for k in keys], width, label="frame level")
    ax.bar([x + width / 2 for x in xs], [data["test_video"][k] for k in keys], width,
           label=f"video level (n={data.get('n_test_videos', '?')})")
    ax.set_xticks(list(xs), keys, rotation=20)
    ax.set(ylim=(0.5, 1.02), title="Test metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "test_metrics.png", dpi=160)
    print("saved figures to", args.out)


if __name__ == "__main__":
    main()
