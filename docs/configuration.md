# Configuration reference

All settings are environment variables with the prefix `FORGERY_`, read from the process environment or from a `.env`
file (copy `.env.example`). Invalid values stop the server at startup with a message naming the setting.
Defaults are shown for a local run; Docker/Kubernetes files may override some of them.

List values (`FORGERY_ALLOWED_EXTENSIONS`, `FORGERY_CORS_ORIGINS`) are written as JSON, e.g.
`FORGERY_ALLOWED_EXTENSIONS='[".mp4", ".mov"]'`.

Settings marked **match training** must equal what `ml/train_baseline.py` used, otherwise predictions are unreliable
(see `model_card.md`). The active values are returned by `GET /api/v1/model`.

## Model and inference
| Variable | Default | Meaning |
|---|---|---|
| `FORGERY_PREDICTOR` | `heuristic` | `heuristic` (placeholder), `torch` (TorchScript) or `keras` |
| `FORGERY_MODEL_PATH` | unset | Model file for `torch`/`keras`. Required for them; checked at startup |
| `FORGERY_NUM_FRAMES` | `16` | Frames sampled per video (1-128). Requests may override it |
| `FORGERY_FRAME_SIZE` | `224` | Square input size in pixels (16-1024). **match training** |
| `FORGERY_NORMALIZE_MODE` | `unit` | `unit` [0,1], `tanh` [-1,1] or `imagenet`. **match training** |
| `FORGERY_DETECT_FACES` | `false` | Crop to the largest face before resizing. **match training**. Requests may override it |
| `FORGERY_AGGREGATION` | `mean` | How frame scores become one video score: `mean`, `median`, `topk` |
| `FORGERY_TOPK_FRACTION` | `0.3` | Share of highest frames averaged when aggregation is `topk` (0-1] |
| `FORGERY_DECISION_THRESHOLD` | `0.5` | Video is forged when its score is at least this. Use the validation threshold |

## Uploads and privacy
| Variable | Default | Meaning |
|---|---|---|
| `FORGERY_MAX_UPLOAD_MB` | `50` | Largest accepted upload (the UI reads this from `/api/v1/model`) |
| `FORGERY_ALLOWED_EXTENSIONS` | `.mp4 .avi .mov .mkv .webm` | Accepted file types (JSON list) |
| `FORGERY_DELETE_AFTER_PROCESSING` | `true` | Delete the video as soon as analysis finishes |
| `FORGERY_STALE_FILE_TTL_HOURS` | `24` | At startup, delete leftover uploads/frames older than this; `0` disables |
| `FORGERY_SAVE_FRAMES` | `false` | Keep the sampled frames on disk (for debugging) |
| `FORGERY_UPLOAD_DIR` | `data/uploads` | Where uploads are stored |
| `FORGERY_FRAMES_DIR` | `data/frames` | Where saved frames go |

## Service
| Variable | Default | Meaning |
|---|---|---|
| `FORGERY_DATABASE_URL` | unset | SQLAlchemy URL (`sqlite:///data/forgery.db`, PostgreSQL). Unset = in-memory jobs |
| `FORGERY_CORS_ORIGINS` | `localhost:3000, localhost:5173` | Allowed browser origins (JSON list) |
| `FORGERY_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `FORGERY_APP_NAME` | `AI Video Forgery Detection API` | Title shown in `/docs` |
| `FORGERY_APP_VERSION` | `0.1.0` | Version reported by `/health` |

Every response carries `X-Request-ID` (a safe client-supplied id is echoed) and `X-Process-Time`.
