# Rishita Agrawal — ML / Model Lead: Task Tracker

Edit the dates/status to match your Project Abstract Form. Status below reflects what is in the repo.

| Task | Status | Where |
|---|---|---|
| Dataset indexing and leak-free split by video | Done | `ml/split.py` |
| Train ResNet18 baseline (augmentation, validation-selected threshold) | Done | `ml/train_baseline.py`, `docs/result/metrics.json` |
| Model card with limitations | Done | `docs/model_card.md` |
| Train/serve consistency: train on frames cropped by the backend | In progress | `ml/make_training_frames.py` |
| Explainability (Grad-CAM heatmaps) | In progress | `ml/gradcam.py`, `docs/result/gradcam/` |
| Aggregation study (mean / median / topk) on deployed pipeline | In progress | `scripts/aggregation_study.py`, `docs/result/aggregation_study.json` |
| Training curves and result figures | In progress | `ml/plots.py`, `docs/result/*.png` |
| Cross-dataset / compression robustness test | Pending | — |
| Integrate trained model into backend and verify end to end | Pending | `scripts/evaluate_pipeline.py` |

## Notes
Only mark a task Done when you have run it and the output is committed.
