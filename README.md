# AI-Generated Video Forgery Detection — Backend & Pipeline

Final-year project (SKIT Jaipur, CSE 2023-27, SDG 16). This repo holds the **backend/pipeline** component:
frame extraction, preprocessing, a FastAPI service, and a pluggable model interface for the CNN.

| Role | Member |
|---|---|
| ML / Model lead | Rishita Agrawal |
| **Backend / Pipeline lead** | **Sumit Todwal** |
| Frontend / UX lead | Sachin Sagar |
| Database / DevOps lead | Tapesh Kumar Vijay |

Mentor: Dr. Meenakshi Nawal

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                           # 17 tests
uvicorn app.main:app --reload                       # http://localhost:8000/docs
```

Try it:

```bash
# CLI: extract + preprocess frames
python scripts/extract_frames.py my_video.mp4 --num-frames 16 --out data/frames/demo
python scripts/extract_frames.py my_video.mp4 --predict

# API
curl -F "file=@my_video.mp4" http://localhost:8000/api/v1/videos          # -> video_id
curl -X POST http://localhost:8000/api/v1/videos/<video_id>/analyze -H "Content-Type: application/json" -d '{"num_frames": 16}'
curl http://localhost:8000/api/v1/jobs/<video_id>
```

Docker: `docker build -t forgery-api . && docker run -p 8000:8000 forgery-api`

## Layout

```
app/
  main.py                  FastAPI app
  api/routes/              health, videos (upload / analyze / jobs / delete)
  core/                    config (env: FORGERY_*), exceptions
  services/
    frame_extractor.py     uniform + fps sampling, video info, save frames
    preprocessing.py       resize, normalize (unit/tanh/imagenet), optional face crop, batching
    pipeline.py            video -> frames -> batch -> prediction
    job_store.py           in-memory job store (swap for DB)
  models/
    base.py                BasePredictor interface (the contract for the CNN)
    heuristic.py           PLACEHOLDER predictor (not a real detector)
    registry.py            predictor selection (FORGERY_PREDICTOR=heuristic|keras)
  schemas/                 Pydantic request/response models
scripts/extract_frames.py  CLI
tests/                     pytest suite (synthetic video fixtures)
docs/                      architecture, API spec, task tracker
```

## Plugging in the real CNN (for Rishita)

Subclass `BasePredictor` in `app/models/base.py`, implement `predict_frames(batch) -> (N,) forged probabilities`
(input is float32 `(N, 224, 224, 3)`), register it in `app/models/registry.py`, then run with
`FORGERY_PREDICTOR=<name>`. A Keras adapter is included: `FORGERY_PREDICTOR=keras FORGERY_MODEL_PATH=model.keras`.

> The default `heuristic` predictor is a placeholder so the pipeline runs end-to-end. It does **not** detect forgeries.

## Configuration

Environment variables (prefix `FORGERY_`) or a `.env` file: `NUM_FRAMES`, `FRAME_SIZE`, `NORMALIZE_MODE`,
`DETECT_FACES`, `SAVE_FRAMES`, `MAX_UPLOAD_MB`, `PREDICTOR`, `MODEL_PATH`.

See [docs/architecture.md](docs/architecture.md), [docs/api-spec.md](docs/api-spec.md) and [docs/TASKS.md](docs/TASKS.md).
