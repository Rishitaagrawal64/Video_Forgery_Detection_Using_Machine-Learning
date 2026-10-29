# AI-Generated Video Forgery Detection (SKIT CSE 2026-27)

CNN-based system that analyses video frames and classifies a video as `REAL` or `AI_GENERATED_FORGED`.

```
React UI  ->  FastAPI (upload -> analyze -> poll)  ->  frame sampling -> preprocessing -> ResNet18 -> aggregation
```

## Quick start (backend)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
python -m uvicorn app.main:app --reload                # Swagger UI: http://localhost:8000/docs
pytest && ruff check .
```
Docker + PostgreSQL: put `POSTGRES_PASSWORD=...` in `.env`, then `docker compose up --build`.

## Quick start (frontend)
```bash
cd frontend && npm ci && npm run dev                   # http://localhost:5173
```
Set `VITE_API_URL` if the API is not at `http://localhost:8000/api/v1`.

## Try the API
```bash
curl -F "file=@sample.mp4" http://localhost:8000/api/v1/videos          # -> {"video_id": "..."}
curl -X POST http://localhost:8000/api/v1/videos/<id>/analyze \
     -H "Content-Type: application/json" -d '{"num_frames": 16, "detect_faces": true}'
curl http://localhost:8000/api/v1/jobs/<id>                             # poll until completed/failed
```
Full contract: `docs/api-spec.md`.

## Use the trained model
```bash
pip install -r ml/requirements.txt
python ml/train_baseline.py --data /path/to/Dataset --out ml/out     # Dataset/{real,fake}/<video>_<n>.png
```
Then in `.env` (these must match training, see `docs/model_card.md`):
```
FORGERY_PREDICTOR=torch
FORGERY_MODEL_PATH=ml/out/forgery_resnet18.pt
FORGERY_NORMALIZE_MODE=imagenet
FORGERY_DETECT_FACES=true
FORGERY_DECISION_THRESHOLD=<"threshold" from ml/out/metrics.json>
```
Check the deployed pipeline end to end on real videos:
```bash
python scripts/evaluate_pipeline.py --data /path/to/videos --out docs/result/pipeline_eval.json   # videos/{real,fake}/
```

## Baseline results
Video-level test AUC 0.998 on 57 videos, from a single split of one dataset. Treat this as a baseline, not proof of
generalization. Details and limitations: `docs/model_card.md`.

## Layout
- `app/` FastAPI backend (routes, services, model adapters), see `docs/architecture.md`
- `ml/` training and dataset splitting, `scripts/` frame extraction and pipeline evaluation
- `frontend/` React + Tailwind UI, `docs/design/` mockups
- `docs/` API spec, database, architecture, model card, task tracker
