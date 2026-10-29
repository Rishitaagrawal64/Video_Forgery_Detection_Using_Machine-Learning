# Model card: ResNet18 baseline (`ml/train_baseline.py`)

## What it is
ResNet18 fine-tuned as a binary classifier on face-crop frames (`real` = 0, `fake` = 1). The backend samples
frames uniformly from a video, scores each, averages the scores, and compares the mean to a threshold.

## Serving must match training
| Setting | Training value | Backend setting |
|---|---|---|
| Input | RGB face crops, 224x224 | `FORGERY_FRAME_SIZE=224`, `FORGERY_DETECT_FACES=true` |
| Normalization | ImageNet mean/std | `FORGERY_NORMALIZE_MODE=imagenet` |
| Output | raw logit (sigmoid applied in `TorchPredictor`) | `FORGERY_PREDICTOR=torch` |
| Threshold | selected on validation videos (F1), stored as `threshold` in `metrics.json` | `FORGERY_DECISION_THRESHOLD` |
| Video score | mean of frame probabilities | `FORGERY_AGGREGATION=mean` |

Note: the training frames are pre-cropped faces, the backend uses an OpenCV Haar cascade for cropping.
Different cropping can shift scores, so verify with `scripts/evaluate_pipeline.py` before trusting results.

## Reported results (`docs/result/metrics.json`, test split)
| Level | Accuracy | Balanced acc. | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Frame | 0.927 | 0.941 | 0.991 | 0.919 | 0.953 | 0.987 |
| Video (57 videos) | 0.983 | 0.989 | 1.000 | 0.978 | 0.989 | 0.998 |

Majority-class accuracy on the test videos is 0.807, so the data is imbalanced toward fake.

## Limitations (read before quoting these numbers)
- The video-level test set is only 57 videos, so one misclassified video moves accuracy by about 1.8 points.
- One dataset and one random split. There is no cross-dataset evaluation, and deepfake detectors often drop sharply
  on unseen generators or compression levels.
- The committed `metrics.json` has no `threshold` field (it was produced before that was recorded). Re-run training
  or set the threshold explicitly.
- Detection is not proof of forgery. Treat the output as a probability, not a verdict.
