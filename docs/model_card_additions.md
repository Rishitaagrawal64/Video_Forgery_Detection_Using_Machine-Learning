<!-- Paste into docs/model_card.md after you have RUN each experiment. Replace every ___ with your real number. -->

## Serving-pipeline check (`scripts/evaluate_pipeline.py`)
| Setup | Videos | Accuracy | F1 | AUC |
|---|---|---|---|---|
| Trained on pre-cropped faces, served with Haar crop | ___ | ___ | ___ | ___ |
| Trained on frames from `ml/make_training_frames.py`, served with Haar crop | ___ | ___ | ___ | ___ |

## Aggregation study (`scripts/aggregation_study.py`, threshold chosen on validation videos)
| Method | Threshold | Test accuracy | Balanced acc. | F1 | AUC |
|---|---|---|---|---|---|
| mean | ___ | ___ | ___ | ___ | ___ |
| median | ___ | ___ | ___ | ___ | ___ |
| topk (30%) | ___ | ___ | ___ | ___ | ___ |

## Explainability
Grad-CAM examples are in `docs/result/gradcam/`. Heatmaps show where the model looked; they are not proof of why a video is fake.
