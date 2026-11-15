## ML model (Rishita Agrawal)

**Task:** per-frame binary classification (`real` = 0, `fake` = 1) on face crops; frame scores are averaged into a video score.

| Item | Value |
|---|---|
| Architecture | ResNet18 (ImageNet weights), single-logit head |
| Input | 224x224 RGB face crops, ImageNet normalization |
| Training | AdamW, lr 3e-4, class-weighted BCE, 8 epochs; augmentation: flip, color jitter, JPEG 35-95, blur |
| Split | by video (no frame leakage), stratified per class |
| Threshold | F1-maximizing on validation videos |
| Video-level test result | see `docs/result/metrics.json` (57 test videos, single split) |

**My additions**
- `ml/gradcam.py`: Grad-CAM heatmaps showing which regions drive a "forged" decision
- `ml/make_training_frames.py`: builds training frames with the same crop the API uses
- `scripts/aggregation_study.py`: compares mean / median / top-k aggregation on the deployed pipeline
- `ml/plots.py`: training curves and metric figures in `docs/result/`

**Limitations:** one dataset, one split, small test set; no cross-dataset evaluation. See `docs/model_card.md`.
